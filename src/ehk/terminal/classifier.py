"""Common terminal component functional for atomic opinion measures.

This module is the Python implementation of the versioned classifier contract
also implemented by the microscopic and lifted Go runtimes.  It deliberately
does not use KDE: both an empirical microscopic state and a binned lifted state
are treated as atomic probability measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ClassifierOptions:
    epsilon: float
    occupied_mass: float
    major_mass: float
    position_resolution: float = 0.0
    mass_resolution: float = 0.0


def _validate(
    positions: np.ndarray,
    masses: np.ndarray,
    options: ClassifierOptions,
) -> None:
    if positions.ndim != 1 or masses.ndim != 1:
        raise ValueError("positions and masses must be one-dimensional")
    if positions.size != masses.size:
        raise ValueError("positions and masses must have equal length")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("masses must be finite and non-negative")
    scalar_rules = (
        ("epsilon", options.epsilon, 0.0, True),
        ("occupied_mass", options.occupied_mass, 0.0, True),
        ("major_mass", options.major_mass, 0.0, False),
        ("position_resolution", options.position_resolution, 0.0, True),
        ("mass_resolution", options.mass_resolution, 0.0, True),
    )
    for name, value, lower, inclusive in scalar_rules:
        valid = math.isfinite(value) and (value >= lower if inclusive else value > lower)
        if not valid:
            relation = "non-negative" if inclusive else "positive"
            raise ValueError(f"{name} must be finite and {relation}")


def _category(count: int) -> str:
    return f"k{count}" if count <= 3 else "k4plus"


def classify_atomic_measure(
    positions: Iterable[float],
    masses: Iterable[float],
    options: ClassifierOptions,
) -> dict[str, object]:
    """Classify an atomic measure using confidence-connected components.

    Positions may be unsorted and duplicated.  The output keys and edge-case
    semantics match both Go implementations and the cross-language fixtures.
    """

    position_array = np.asarray(tuple(positions), dtype=float)
    mass_array = np.asarray(tuple(masses), dtype=float)
    _validate(position_array, mass_array, options)

    atoms: dict[float, float] = {}
    for position, mass in zip(position_array, mass_array, strict=True):
        if mass > 0:
            atoms[float(position)] = atoms.get(float(position), 0.0) + float(mass)
    occupied = [
        (position, mass)
        for position, mass in sorted(atoms.items())
        if mass >= options.occupied_mass
    ]
    result: dict[str, object] = {
        "status": "nonterminal",
        "category": "censored",
        "k_all": 0,
        "k_major": 0,
        "components": [],
        "margins": {
            "gap_to_epsilon": -1.0,
            "diameter_to_epsilon": -1.0,
            "mass_to_major_cutoff": -1.0,
        },
    }
    if not occupied:
        return result

    ranges = [[0, 0]]
    ambiguous = False
    gap_margins: list[float] = []
    for index in range(1, len(occupied)):
        gap = occupied[index][0] - occupied[index - 1][0]
        margin = abs(gap - options.epsilon)
        gap_margins.append(margin)
        ambiguous |= (
            options.position_resolution > 0
            and margin <= options.position_resolution
        )
        if gap > options.epsilon:
            ranges.append([index, index])
        else:
            ranges[-1][1] = index

    components: list[dict[str, object]] = []
    diameter_margins: list[float] = []
    mass_margins: list[float] = []
    absorbed = True
    k_major = 0
    largest_mass = 0.0
    for first, last in ranges:
        minimum = occupied[first][0]
        maximum = occupied[last][0]
        mass = sum(item[1] for item in occupied[first:last + 1])
        diameter = maximum - minimum
        diameter_margin = abs(options.epsilon - diameter)
        mass_margin = abs(mass - options.major_mass)
        diameter_margins.append(diameter_margin)
        mass_margins.append(mass_margin)
        absorbed &= diameter <= options.epsilon
        ambiguous |= (
            options.position_resolution > 0
            and diameter_margin <= options.position_resolution
        )
        ambiguous |= (
            options.mass_resolution > 0
            and mass_margin <= options.mass_resolution
        )
        major = mass >= options.major_mass
        k_major += int(major)
        largest_mass = max(largest_mass, mass)
        components.append({
            "minimum": minimum,
            "maximum": maximum,
            "mass": mass,
            "major": major,
        })

    if k_major == 0 and largest_mass > 0:
        k_major = 1
    result.update({
        "k_all": len(components),
        "k_major": k_major,
        "components": components,
        "margins": {
            "gap_to_epsilon": min(gap_margins, default=-1.0),
            "diameter_to_epsilon": min(diameter_margins, default=-1.0),
            "mass_to_major_cutoff": min(mass_margins, default=-1.0),
        },
    })
    if ambiguous:
        result["status"] = "grid_ambiguous"
    elif absorbed:
        result["status"] = "absorbed"
        result["category"] = _category(k_major)
    return result


def classify_opinions(
    opinions: Iterable[float],
    epsilon: float,
    major_mass: float = 0.02,
    *,
    position_resolution: float = 0.0,
    mass_resolution: float = 0.0,
) -> dict[str, object]:
    """Classify an empirical measure with equal mass for every agent."""

    values = np.asarray(tuple(opinions), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("opinions must be a non-empty one-dimensional sample")
    agent_mass = 1.0 / values.size
    return classify_atomic_measure(
        values,
        np.full(values.size, agent_mass),
        ClassifierOptions(
            epsilon=epsilon,
            occupied_mass=0.5 * agent_mass,
            major_mass=major_mass,
            position_resolution=position_resolution,
            mass_resolution=mass_resolution,
        ),
    )


def options_as_dict(options: ClassifierOptions) -> dict[str, float]:
    """Return the exact protocol representation used for fingerprints."""

    return asdict(options)
