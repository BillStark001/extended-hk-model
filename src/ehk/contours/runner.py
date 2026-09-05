"""Recoverable orchestration for sequential contour estimation."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import numpy as np

from .acquisition import (
    AcquisitionResult,
    Action,
    integrated_misclassification_risk,
    propose_action,
    sobol_points,
)
from .gp import MaternGP
from .protocol import ContourProtocol
from .store import Evaluation, EvaluationStore, utc_now

Evaluator = Callable[[Action, ContourProtocol], Evaluation]


class SequentialContourEstimator:
    """Fit, acquire, evaluate, journal, and resume a contour experiment."""

    def __init__(
        self,
        protocol: ContourProtocol,
        store: EvaluationStore,
        *,
        progress: str = "human",
        progress_stream: TextIO | None = None,
    ):
        if store.protocol.fingerprint != protocol.fingerprint:
            raise ValueError("estimator/store protocol mismatch")
        if progress not in {"none", "human", "jsonl"}:
            raise ValueError("progress must be none, human, or jsonl")
        self.protocol = protocol
        self.store = store
        self.progress = progress
        self.progress_stream = progress_stream or sys.stderr

    def _emit(self, event_type: str, **fields: object) -> None:
        event = {"type": event_type, "created_utc": utc_now(), **fields}
        self.store.append_progress(event)
        if self.progress == "jsonl":
            print(json.dumps(event, sort_keys=True), file=self.progress_stream, flush=True)
        elif self.progress == "human":
            detail = " ".join(f"{key}={value}" for key, value in fields.items())
            print(f"[contour] {event_type} {detail}".rstrip(), file=self.progress_stream, flush=True)

    def _initial_actions(self) -> list[Action]:
        design = self.protocol.design
        count = int(design["initial_low"])
        promotions = min(int(design["initial_promotions"]), count)
        points = sobol_points(self.protocol, count, int(design["seed"]))
        actions: list[Action] = []
        for group in self.protocol.groups:
            for point in points:
                actions.append(Action(
                    group,
                    self.protocol.fidelities[0],
                    tuple(float(item) for item in point),
                    0,
                    "new",
                    "initial",
                ))
            for fidelity in self.protocol.fidelities[1:]:
                for point in points[:promotions]:
                    actions.append(Action(
                        group,
                        fidelity,
                        tuple(float(item) for item in point),
                        0,
                        "promote",
                        "initial",
                    ))
        return actions

    def _validation_actions(self) -> list[Action]:
        count = int(self.protocol.raw["validation"].get("count", 0))
        if count <= 0:
            return []
        seed = int(self.protocol.raw["validation"].get("seed", self.protocol.design["seed"] + 1))
        points = sobol_points(self.protocol, count, seed)
        return [
            Action(group, self.protocol.fidelities[-1], tuple(point), 0, "new", "validation")
            for group in self.protocol.groups
            for point in points
        ]

    def _models(self, evaluations: list[Evaluation]) -> dict[str, MaternGP]:
        models: dict[str, MaternGP] = {}
        for group in self.protocol.groups:
            selected = [item for item in evaluations if item.group == group]
            if len(MaternGP._training_rows(selected)) >= 2:
                models[group] = MaternGP(self.protocol).fit(selected)
        return models

    def _action_from_pending(self) -> Action | None:
        pending = self.store.pending_proposals()
        if not pending:
            return None
        return Action.from_dict(dict(pending[0]["action"]))

    def _next_initial(self, evaluations: list[Evaluation]) -> Action | None:
        completed = {item.key for item in evaluations}
        successful_counts = {
            (group, fidelity): sum(
                item.status == "ok" and item.role != "validation"
                and item.group == group and item.fidelity == fidelity
                for item in evaluations
            )
            for group in self.protocol.groups
            for fidelity in self.protocol.fidelities
        }
        for action in self._initial_actions():
            target = (
                int(self.protocol.design["initial_low"])
                if action.fidelity == self.protocol.fidelities[0]
                else int(self.protocol.design["initial_promotions"])
            )
            if successful_counts[(action.group, action.fidelity)] >= target:
                continue
            if action.key(self.protocol) not in completed:
                return action
        return None

    def _evaluate(self, action: Action, evaluator: Evaluator) -> Evaluation:
        started = time.monotonic()
        try:
            result = evaluator(action, self.protocol)
            if not isinstance(result, Evaluation):
                raise TypeError("evaluator must return Evaluation")
            if result.key != action.key(self.protocol):
                raise ValueError("evaluator returned an evaluation for a different action")
            if result.role != action.role:
                raise ValueError("evaluator returned an evaluation with the wrong role")
            result.validate(self.protocol)
            return result
        # Evaluator failures are data-quality records; the sequential run must remain resumable.
        except Exception as error:  # noqa: BLE001
            return Evaluation(
                protocol_sha256=self.protocol.fingerprint,
                group=action.group,
                fidelity=action.fidelity,
                x=action.x,
                replicate=action.replicate,
                role=action.role,
                status="failed",
                runtime_seconds=time.monotonic() - started,
                payload={"error": f"{type(error).__name__}: {error}"},
            )

    def _record_action(
        self,
        action: Action,
        acquisition: AcquisitionResult | None,
        iteration: int,
    ) -> None:
        record: dict[str, object] = {
            "iteration": iteration,
            "evaluation_key": action.key(self.protocol),
            "action": action.as_dict(),
        }
        if acquisition is not None:
            record["acquisition"] = {
                key: value for key, value in asdict(acquisition).items() if key != "action"
            }
        self.store.append_proposal(record)

    def metrics(
        self,
        models: Mapping[str, MaternGP],
        *,
        count: int | None = None,
    ) -> dict[str, object]:
        count = count or int(self.protocol.design["integration_count"])
        points = sobol_points(self.protocol, count, int(self.protocol.design["seed"]) + 7919)
        target = self.protocol.fidelities[-1]
        by_group: dict[str, object] = {}
        for group, model in models.items():
            prediction = model.predict(points, [target] * len(points))
            risk = integrated_misclassification_risk(
                prediction.mean,
                prediction.variance,
                self.protocol.levels,
                self.protocol.level_weights,
            )
            standard = np.sqrt(prediction.variance)
            band_fraction = {
                str(level): float(np.mean(np.abs(prediction.mean - level) <= 1.96 * standard))
                for level in self.protocol.levels
            }
            by_group[group] = {
                "integrated_misclassification_risk": risk,
                "credible_band_fraction": band_fraction,
            }
        return {"groups": by_group}

    def _stop(self, metrics: Mapping[str, object]) -> bool:
        groups = metrics["groups"]
        if len(groups) != len(self.protocol.groups):
            return False
        threshold = float(self.protocol.design["risk_tolerance"])
        return all(
            float(value["integrated_misclassification_risk"]) <= threshold
            for value in groups.values()
        )

    def run(
        self,
        evaluator: Evaluator,
        *,
        max_evaluations: int,
        run_validation: bool = True,
    ) -> dict[str, object]:
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be positive")
        executed = 0
        stop_reason = "evaluation_budget"
        self._emit("resume", stored_evaluations=len(self.store.evaluations()))
        while executed < max_evaluations:
            evaluations = self.store.evaluations()
            action = self._action_from_pending()
            acquisition: AcquisitionResult | None = None
            if action is None:
                action = self._next_initial(evaluations)
            if action is None:
                models = self._models(evaluations)
                current_metrics = self.metrics(models)
                if self._stop(current_metrics):
                    stop_reason = "risk_tolerance"
                    self._emit("stop", reason=stop_reason, metrics=current_metrics)
                    break
                acquisition = propose_action(
                    self.protocol,
                    evaluations,
                    models,
                    seed_offset=len(evaluations),
                )
                action = acquisition.action
            key = action.key(self.protocol)
            if not any(item.key == key for item in evaluations):
                self._record_action(action, acquisition, len(evaluations))
            self._emit(
                "evaluation_start",
                evaluation_key=key,
                group=action.group,
                fidelity=action.fidelity,
                kind=action.kind,
                x=list(action.x),
            )
            result = self._evaluate(action, evaluator)
            self.store.append_evaluation(result)
            executed += 1
            self._emit(
                "evaluation_done",
                evaluation_key=key,
                status=result.status,
                value=result.value,
                censoring=result.censoring,
                runtime_seconds=result.runtime_seconds,
            )

        if run_validation:
            existing = {item.key for item in self.store.evaluations()}
            for action in self._validation_actions():
                if action.key(self.protocol) in existing:
                    continue
                self._record_action(action, None, len(existing))
                result = self._evaluate(action, evaluator)
                self.store.append_evaluation(result)
                existing.add(result.key)
                self._emit("validation_done", evaluation_key=result.key, status=result.status)

        evaluations = self.store.evaluations()
        models = self._models(evaluations)
        metrics = self.metrics(models)
        validation = self.validation_metrics(models, evaluations)
        summary = {
            "protocol_sha256": self.protocol.fingerprint,
            "stop_reason": stop_reason,
            "evaluations": len(evaluations),
            "successful": sum(item.status == "ok" for item in evaluations),
            "failed": sum(item.status != "ok" for item in evaluations),
            "metrics": metrics,
            "validation": validation,
            "updated_utc": utc_now(),
        }
        EvaluationStore._atomic_json(self.store.root / "summary.json", summary)
        self._emit("complete", stop_reason=stop_reason, evaluations=len(evaluations))
        return summary

    def validation_metrics(
        self,
        models: Mapping[str, MaternGP],
        evaluations: list[Evaluation],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for group, model in models.items():
            rows = [
                item for item in evaluations
                if item.group == group and item.role == "validation"
                and item.status == "ok" and item.censoring == "none" and item.value is not None
            ]
            if not rows:
                continue
            prediction = model.predict([item.x for item in rows], [item.fidelity for item in rows])
            residual = np.asarray([item.value for item in rows]) - prediction.mean
            standard = np.sqrt(prediction.variance + np.asarray([item.noise_variance for item in rows]))
            result[group] = {
                "count": len(rows),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "standardized_rmse": float(np.sqrt(np.mean((residual / standard) ** 2))),
                "coverage_95": float(np.mean(np.abs(residual) <= 1.96 * standard)),
            }
        return result

    def export_grid(self, *, resolution: int = 201) -> Path:
        if self.protocol.dimension != 2:
            raise ValueError("grid export currently requires a two-dimensional domain")
        if resolution < 3:
            raise ValueError("resolution must be at least 3")
        models = self._models(self.store.evaluations())
        x_axis = np.linspace(*self.protocol.domain[0], resolution)
        y_axis = np.linspace(*self.protocol.domain[1], resolution)
        xx, yy = np.meshgrid(x_axis, y_axis)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        arrays: dict[str, object] = {
            "x": x_axis,
            "y": y_axis,
            "levels": np.asarray(self.protocol.levels),
            "protocol_sha256": np.asarray(self.protocol.fingerprint),
        }
        for index, group in enumerate(self.protocol.groups):
            prediction = models[group].predict(
                points,
                [self.protocol.fidelities[-1]] * len(points),
            )
            arrays[f"mean_{index}"] = prediction.mean.reshape(xx.shape)
            arrays[f"std_{index}"] = np.sqrt(prediction.variance).reshape(xx.shape)
            arrays[f"group_{index}"] = np.asarray(group)
        path = self.store.root / "posterior_grid.npz"
        np.savez_compressed(path, **arrays)
        return path
