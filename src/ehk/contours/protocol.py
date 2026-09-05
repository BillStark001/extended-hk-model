"""Strict, content-addressed protocol for contour experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def canonical_json(value: object) -> str:
    """Encode a protocol deterministically without lossy float rewriting."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContourProtocol:
    """Validated protocol whose complete JSON content defines data identity."""

    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.validate()

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.raw)

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def domain(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(item[0]), float(item[1])) for item in self.raw["domain"])

    @property
    def dimension(self) -> int:
        return len(self.domain)

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.raw["levels"])

    @property
    def level_weights(self) -> tuple[float, ...]:
        weights = self.raw.get("level_weights", [1.0] * len(self.levels))
        return tuple(float(item) for item in weights)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self.raw["groups"])

    @property
    def fidelities(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.raw["fidelities"])

    @property
    def design(self) -> Mapping[str, Any]:
        return self.raw["design"]

    @property
    def response(self) -> Mapping[str, Any]:
        return self.raw["response"]

    def validate(self) -> None:
        required = {
            "schema_version", "name", "coordinates", "domain", "levels",
            "groups", "fidelities", "response", "simulator", "design",
            "validation",
        }
        missing = sorted(required - self.raw.keys())
        unknown = sorted(self.raw.keys() - required - {"level_weights", "notes"})
        if missing:
            raise ValueError("protocol is missing: " + ", ".join(missing))
        if unknown:
            raise ValueError("unknown protocol fields: " + ", ".join(unknown))
        if self.raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if not str(self.raw["name"]).strip():
            raise ValueError("name must not be empty")
        coordinates = self.raw["coordinates"]
        domain = self.raw["domain"]
        if not isinstance(coordinates, list) or not coordinates or len(coordinates) != len(domain):
            raise ValueError("coordinates and domain must have equal nonzero length")
        for interval in domain:
            if (
                not isinstance(interval, list) or len(interval) != 2
                or not all(math.isfinite(float(item)) for item in interval)
                or float(interval[0]) >= float(interval[1])
            ):
                raise ValueError("each domain interval must be finite and increasing")
        if len(set(map(str, coordinates))) != len(coordinates):
            raise ValueError("coordinate names must be unique")
        levels = self.levels
        if not levels or not all(math.isfinite(item) for item in levels):
            raise ValueError("levels must be nonempty and finite")
        if tuple(sorted(set(levels))) != levels:
            raise ValueError("levels must be unique and strictly increasing")
        weights = self.level_weights
        if len(weights) != len(levels) or any(not math.isfinite(item) or item <= 0 for item in weights):
            raise ValueError("level_weights must be positive and match levels")
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError("groups must be nonempty and unique")
        if not self.fidelities or tuple(sorted(set(self.fidelities))) != self.fidelities:
            raise ValueError("fidelities must be positive, unique, and increasing")
        if any(item <= 0 for item in self.fidelities):
            raise ValueError("fidelities must be positive")
        response = self.response
        if response.get("kind") not in {"deterministic_scalar", "noisy_scalar", "multinomial_tv"}:
            raise ValueError("unsupported response.kind")
        for section in ("response", "simulator", "design", "validation"):
            if not isinstance(self.raw[section], dict):
                raise TypeError(f"{section} must be an object")
        for field in ("seed", "candidate_count", "integration_count", "initial_low", "initial_promotions"):
            if int(self.design.get(field, 0)) < 1:
                raise ValueError(f"design.{field} must be positive")
        if float(self.design.get("risk_tolerance", 0)) <= 0:
            raise ValueError("design.risk_tolerance must be positive")
        if float(self.design.get("exploration_weight", -1)) < 0:
            raise ValueError("design.exploration_weight must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.raw))


def load_protocol(source: str | Path | Mapping[str, Any]) -> ContourProtocol:
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("protocol root must be an object")
    return ContourProtocol(raw)
