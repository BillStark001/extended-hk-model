from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ehk.contours import (
    Action,
    ContourProtocol,
    Evaluation,
    EvaluationStore,
    LandscapeBarrierGapEvaluator,
    MaternGP,
    PairedMultimetricEvaluator,
    SequentialContourEstimator,
    common_terminal_time,
    hellinger_gap,
    normalized_time_l1_gap,
    propose_action,
    terminal_density_l2_gaps,
    total_variation_from_counts,
)
from ehk.contours.cli import main as contour_main


def protocol_dict(**design_updates: object) -> dict[str, object]:
    design = {
        "seed": 17,
        "candidate_count": 8,
        "integration_count": 16,
        "initial_low": 4,
        "initial_promotions": 2,
        "risk_tolerance": 1e-4,
        "exploration_weight": 0.05,
        "fidelity_costs": {"41": 1.0, "81": 4.0},
    }
    design.update(design_updates)
    return {
        "schema_version": 1,
        "name": "synthetic-circle",
        "coordinates": ["log10_alpha", "log10_q"],
        "domain": [[-3.0, 0.0], [-3.0, 0.0]],
        "levels": [-1.0, 0.0],
        "level_weights": [1.0, 3.0],
        "groups": ["hk/random"],
        "fidelities": [41, 81],
        "response": {"kind": "deterministic_scalar", "transform": "identity"},
        "simulator": {"name": "synthetic", "epsilon": 0.4},
        "design": design,
        "validation": {"count": 2, "seed": 99},
    }


def evaluation(
    protocol: ContourProtocol,
    x: tuple[float, float],
    value: float,
    *,
    fidelity: int = 81,
    replicate: int = 0,
    role: str = "train",
    censoring: str = "none",
    censor_bound: float | None = None,
) -> Evaluation:
    return Evaluation(
        protocol.fingerprint,
        "hk/random",
        fidelity,
        x,
        replicate=replicate,
        role=role,
        value=value,
        censoring=censoring,
        censor_bound=censor_bound,
        runtime_seconds=0.1,
    )


def test_protocol_fingerprint_is_key_order_independent_and_strict() -> None:
    raw = protocol_dict()
    protocol = ContourProtocol(raw)
    reordered = dict(reversed(list(raw.items())))
    assert ContourProtocol(reordered).fingerprint == protocol.fingerprint
    invalid = dict(raw)
    invalid["surprise"] = True
    with pytest.raises(ValueError, match="unknown protocol fields"):
        ContourProtocol(invalid)


def test_terminal_tv_adapter_propagates_multinomial_noise() -> None:
    value, variance = total_variation_from_counts([40, 10, 0], [25, 20, 5])
    assert value == pytest.approx(0.3)
    assert variance > 0


def test_terminal_density_l2_uses_continuous_density_normalization() -> None:
    node, edge = terminal_density_l2_gaps(
        [1.0, 0.0],
        [0.0, 1.0],
        np.asarray([[2.0, 0.0], [0.0, 0.0]]),
        np.zeros((2, 2)),
        cell_width=1.0,
        out_degree=2,
    )
    assert node == pytest.approx(math.sqrt(2.0))
    assert edge == pytest.approx(1.0)


def test_hellinger_gap_has_common_bounded_scale_for_nodes_and_edges() -> None:
    assert hellinger_gap([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert hellinger_gap([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert hellinger_gap(
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.0, 0.0], [0.0, 1.0]]),
    ) == pytest.approx(1.0)


def test_normalized_time_l1_gap_uses_initial_energy_and_trapezoid() -> None:
    gap, initial = normalized_time_l1_gap(
        [0.0, 2.0, 4.0],
        [0.1, 0.2, 0.3],
        [0.1, 0.1, 0.1],
    )
    assert initial == pytest.approx(0.1)
    assert gap == pytest.approx(1.0)


def test_terminal_comparison_requires_same_declared_time() -> None:
    paired = (
        SimpleNamespace(time=np.asarray([0.0, 4.0])),
        SimpleNamespace(time=np.asarray([0.0, 2.0, 4.0])),
    )
    assert common_terminal_time(paired, 4.0) == 4.0
    with pytest.raises(ValueError, match="terminal snapshots must be evaluated"):
        common_terminal_time(
            (paired[0], SimpleNamespace(time=np.asarray([0.0, 3.0]))),
            4.0,
        )


def test_landscape_schedule_preserves_dense_transient_and_terminal_step() -> None:
    steps = LandscapeBarrierGapEvaluator._record_steps(
        {
            "record_schedule": {
                "dense_through": 4,
                "dense_stride": 1,
                "coarse_stride": 3,
            }
        },
        11,
    )
    assert steps == (0, 1, 2, 3, 4, 7, 10, 11)


def test_landscape_adapter_returns_logged_peak_barrier_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "smp-kinetic"
    binary.write_bytes(b"test runtime")
    raw = protocol_dict(initial_low=2, initial_promotions=1)
    raw["fidelities"] = [81]
    raw["design"]["fidelity_costs"] = {"81": 1.0}
    raw["response"] = {
        "kind": "deterministic_scalar",
        "observable": "peak_dominant_barrier_height",
        "methods": ["measure", "fokker_planck"],
        "numerical_floor_by_fidelity": {"81": 1e-10},
        "landscape": {
            "record_schedule": {
                "dense_through": 2,
                "dense_stride": 1,
                "coarse_stride": 2,
            },
            "min_basin_mass": 0.02,
            "min_barrier_height": 1e-4,
            "max_well_displacement": 0.25,
            "dominant_score_margin": 0.05,
            "dominant_switch_persistence": 3,
            "overshoot_tolerance": 1e-4,
        },
    }
    raw["simulator"] = {
        "runtime_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "processes_per_evaluation": 2,
        "epsilon": 0.4,
        "mean_degree": 15,
        "recommendation_count": 10,
        "noise_diffusion": 0.0,
        "dt": 1.0,
        "steps": 4,
        "record_every": 2,
        "confidence_mode": "cell_average",
        "resolution": {
            "population": 500,
            "opinion_min": -1.0,
            "opinion_max": 1.0,
            "opinion_quadrature_points": 5,
            "opinion_quadrature_rule": "gauss_hermite",
            "confidence_quadrature_points": 5,
            "score_max": 45,
            "distance_grid_size": 256,
            "minimum_bandwidth": 0.02,
            "objective_effective_samples": 500,
        },
        "group_parameters": {
            "hk/random": {
                "dynamics": "hk",
                "recommender": {
                    "type": "random",
                    "steepness": 1.0,
                    "random_ratio": 0.0,
                    "opinion_tolerance": 0.4,
                },
            }
        },
    }
    protocol = ContourProtocol(raw)

    def fake_batch(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        cases = args[1]
        processes = args[3]
        assert processes == 2
        assert all(tuple(case[2]) == (0, 1, 2, 4) for case in cases)
        heights = (0.2, 0.5)
        return [
            SimpleNamespace(
                x=np.linspace(-1.0, 1.0, 5),
                time=np.asarray([0.0, 1.0]),
                rho=np.ones((2, 5)),
                velocity=np.full((2, 5), height * case[1].influence),
                max_node_mass_residual=1e-12,
                max_fixed_degree_residual=2e-12,
            )
            for height, case in zip(heights, cases, strict=True)
        ]

    monkeypatch.setattr("ehk.contours.adapters.solve_density_velocity_batch", fake_batch)
    monkeypatch.setattr("ehk.contours.adapters.potential_from_force", lambda x, force: force)
    monkeypatch.setattr(
        "ehk.contours.adapters.quantify_multiwell_series",
        lambda x, time, potential, rho, **kwargs: SimpleNamespace(
            peak_barrier_height=float(potential[0, 0]),
            peak_time=1.0,
            final_barrier_height=float(potential[-1, 0]),
            overshoot_class="no_overshoot",
            dominant_switch=np.asarray([False, False]),
        ),
    )
    evaluator = LandscapeBarrierGapEvaluator(binary)
    result = evaluator(
        Action("hk/random", 81, (-2.0, -1.0), 0, "new", "train"),
        protocol,
    )
    assert result.value == pytest.approx(math.log10(0.3))
    assert result.payload["raw"] == {"measure": 0.2, "fokker_planck": 0.5}


def test_multimetric_adapter_returns_hellinger_energy_and_pathway_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = tmp_path / "smp-kinetic"
    binary.write_bytes(b"test runtime")
    raw = protocol_dict(initial_low=2, initial_promotions=1)
    raw["fidelities"] = [81]
    raw["design"]["fidelity_costs"] = {"81": 1.0}
    raw["response"] = {
        "kind": "deterministic_scalar",
        "observable": "paired_multimetric_validity",
        "methods": ["measure", "fokker_planck"],
        "dynamics": ["hk", "deffuant"],
        "numerical_floor_by_metric": {
            "node_hellinger": 1e-10,
            "edge_hellinger": 1e-10,
            "u_rho_time_gap": 1e-10,
            "u_e_time_gap": 1e-10,
            "pathway_gap": 1e-10,
        },
    }
    raw["simulator"] = {
        "runtime_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "processes_per_evaluation": 4,
        "epsilon": 0.4,
        "mean_degree": 2,
        "recommendation_count": 1,
        "noise_diffusion": 0.0,
        "dt": 1.0,
        "steps": 4,
        "record_every": 2,
        "confidence_mode": "cell_average",
        "resolution": {
            "population": 500,
            "opinion_min": -1.0,
            "opinion_max": 1.0,
            "opinion_quadrature_points": 5,
            "opinion_quadrature_rule": "gauss_hermite",
            "confidence_quadrature_points": 5,
            "score_max": 45,
            "distance_grid_size": 256,
            "minimum_bandwidth": 0.02,
            "objective_effective_samples": 500,
        },
        "dynamics_parameters": {
            dynamics: {
                "recommender": {
                    "type": "random",
                    "steepness": 1.0,
                    "random_ratio": 0.0,
                    "opinion_tolerance": 0.4,
                }
            }
            for dynamics in ("hk", "deffuant")
        },
    }
    protocol = ContourProtocol(raw)

    pathways = (0.5, 0.2, 1.0, 0.4)

    def fake_batch(*args: object, **kwargs: object) -> list[SimpleNamespace]:
        cases = args[1]
        return [
            SimpleNamespace(
                time=np.asarray([0.0, 2.0, 4.0]),
                node_energy=np.asarray(
                    [0.1, 0.1 + 0.1 * (index % 2 == 0) * (index // 2 + 1),
                     0.1 + 0.2 * (index % 2 == 0) * (index // 2 + 1)]
                ),
                edge_energy=np.asarray(
                    [0.2, 0.2 + 0.2 * (index % 2 == 0) * (index // 2 + 1),
                     0.2 + 0.4 * (index % 2 == 0) * (index // 2 + 1)]
                ),
                final_rho=(
                    np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
                    if index % 2 == 0
                    else np.asarray([0.5, 0.5, 0.0, 0.0, 0.0])
                ),
                final_edge=(
                    np.pad(np.asarray([[2.0]]), ((0, 4), (0, 4)))
                    if index % 2 == 0
                    else np.pad(np.asarray([[1.0, 1.0]]), ((0, 4), (0, 3)))
                ),
                pathway=pathway,
                max_node_mass_residual=1e-12,
                max_fixed_degree_residual=2e-12,
            )
            for index, (pathway, case) in enumerate(
                zip(pathways, cases, strict=True)
            )
        ]

    monkeypatch.setattr(
        "ehk.contours.adapters.solve_multimetric_batch", fake_batch
    )
    result = PairedMultimetricEvaluator(binary)((-2.0, -1.0), 81, protocol)
    hk = result["dynamics"]["hk"]
    deffuant = result["dynamics"]["deffuant"]
    assert set(hk["raw_metrics"]) == set(PairedMultimetricEvaluator.metric_names)
    assert hk["raw_metrics"]["u_rho_time_gap"] == pytest.approx(1.0)
    assert hk["raw_metrics"]["u_e_time_gap"] == pytest.approx(1.0)
    assert hk["raw_metrics"]["pathway_gap"] == pytest.approx(0.3)
    assert deffuant["raw_metrics"]["u_rho_time_gap"] == pytest.approx(2.0)
    assert deffuant["raw_metrics"]["u_e_time_gap"] == pytest.approx(2.0)
    assert deffuant["raw_metrics"]["pathway_gap"] == pytest.approx(0.6)
    assert 0.0 <= hk["raw_metrics"]["node_hellinger"] <= 1.0
    assert 0.0 <= hk["raw_metrics"]["edge_hellinger"] <= 1.0
    assert result["dynamics"]["hk"]["terminal_time"] == 4.0


def test_store_is_idempotent_and_ignores_interrupted_final_record(tmp_path: Path) -> None:
    protocol = ContourProtocol(protocol_dict())
    store = EvaluationStore(tmp_path, protocol)
    item = evaluation(protocol, (-2.0, -1.0), 0.25)
    assert store.append_evaluation(item) == item
    assert store.append_evaluation(item) == item
    assert len(store.evaluations()) == 1
    with store.evaluations_path.open("ab") as stream:
        stream.write(b'{"interrupted":')
    assert len(store.evaluations()) == 1
    second = evaluation(protocol, (-1.0, -2.0), 0.5)
    store.append_evaluation(second)
    assert [item.key for item in store.evaluations()] == [item.key, second.key]
    with pytest.raises(ValueError, match="different protocol"):
        changed = protocol_dict()
        changed["name"] = "other"
        EvaluationStore(tmp_path, ContourProtocol(changed))


def test_cli_initializes_and_reports_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    protocol_path = tmp_path / "input.json"
    protocol_path.write_text(json.dumps(protocol_dict()), encoding="utf-8")
    store_path = tmp_path / "run"
    contour_main(["init", str(protocol_path), str(store_path)])
    initialized = json.loads(capsys.readouterr().out)
    assert len(initialized["protocol_sha256"]) == 64
    contour_main(["status", str(store_path)])
    status = json.loads(capsys.readouterr().out)
    assert status["evaluations"] == 0
    assert status["pending"] == 0


def test_gp_supports_known_noise_censoring_and_fidelity() -> None:
    protocol = ContourProtocol(protocol_dict())
    rows = []
    for fidelity in protocol.fidelities:
        bias = 0.2 * (81 / fidelity - 1)
        for x in (-2.8, -2.0, -1.2, -0.3):
            rows.append(evaluation(protocol, (x, -1.5), math.sin(x) + bias, fidelity=fidelity))
    rows.append(evaluation(
        protocol,
        (-2.5, -2.5),
        -3.0,
        censoring="left",
        censor_bound=-2.0,
    ))
    model = MaternGP(protocol).fit(rows)
    prediction = model.predict([[-2.0, -1.5]], [81])
    assert prediction.mean.shape == (1,)
    assert prediction.variance[0] > 0
    assert abs(prediction.mean[0] - math.sin(-2.0)) < 0.35
    covariance = model.posterior_covariance(
        [[-2.0, -1.5]], [81], [[-2.0, -1.5]], [41]
    )
    assert covariance.shape == (1, 1)
    assert np.isfinite(covariance[0, 0])


def test_gp_loo_variance_calibration_never_shrinks_uncertainty() -> None:
    raw = protocol_dict()
    raw["design"]["variance_calibration"] = "loo_quantile"
    raw["design"]["variance_calibration_quantile"] = 0.95
    protocol = ContourProtocol(raw)
    rows = [
        evaluation(
            protocol,
            (float(x), -1.5),
            math.sin(5.0 * float(x)) + 0.1 * float(x),
        )
        for x in np.linspace(-2.9, -0.1, 12)
    ]
    model = MaternGP(protocol).fit(rows)
    assert model.variance_calibration_scale >= 1.0
    assert np.isfinite(model.loo_standardized_rmse)
    prediction = model.predict([[-1.4, -1.5]], [81], include_noise=True)
    assert prediction.variance[0] > 0


def test_acquisition_returns_costed_new_or_promote_action() -> None:
    protocol = ContourProtocol(protocol_dict())
    rows = [
        evaluation(protocol, (-2.8, -2.8), -1.2, fidelity=41),
        evaluation(protocol, (-1.9, -2.0), -0.2, fidelity=41),
        evaluation(protocol, (-1.0, -1.2), 0.4, fidelity=41),
        evaluation(protocol, (-0.2, -0.3), 1.0, fidelity=81),
    ]
    model = MaternGP(protocol).fit(rows)
    result = propose_action(protocol, rows, {"hk/random": model})
    assert result.action.kind in {"new", "promote"}
    assert result.action.key(protocol) not in {item.key for item in rows}
    assert result.score >= 0
    assert result.estimated_cost > 0


def test_runner_resumes_pending_proposal_and_writes_summary(tmp_path: Path) -> None:
    protocol = ContourProtocol(protocol_dict(initial_low=2, initial_promotions=1))
    store = EvaluationStore(tmp_path, protocol)
    action = Action("hk/random", 41, (-2.5, -1.0), 0, "new", "initial")
    store.append_proposal({
        "evaluation_key": action.key(protocol),
        "iteration": 0,
        "action": action.as_dict(),
    })
    called: list[tuple[float, ...]] = []

    def evaluator(action: Action, protocol: ContourProtocol) -> Evaluation:
        called.append(action.x)
        value = (action.x[0] + 1.5) ** 2 + (action.x[1] + 1.5) ** 2 - 1
        return evaluation(
            protocol,
            action.x,
            value,
            fidelity=action.fidelity,
            replicate=action.replicate,
            role=action.role,
        )

    estimator = SequentialContourEstimator(protocol, store, progress="none")
    summary = estimator.run(evaluator, max_evaluations=5, run_validation=False)
    assert called[0] == action.x
    assert summary["successful"] == 5
    assert (tmp_path / "summary.json").exists()
    assert len(store.pending_proposals()) == 0
    assert estimator.export_grid(resolution=11).exists()
