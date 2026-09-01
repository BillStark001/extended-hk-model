"""Shared protocol definitions for the multibarrier dynamics comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    PAPER_RATES,
    RecommendationScenario,
    record_schedule,
    select_scenarios,
)

DEFAULT_CONFIGURATIONS = (
    "random",
    "opinion_random_zeta1",
    "opinion_random_zeta4",
    "structure_random_l0_zeta1",
    "structure_random_l0_zeta4",
)
DEFAULT_DYNAMICS = ("hk", "deffuant")
DEFAULT_METHODS = ("measure", "fokker_planck")
DEFAULT_EPSILON_GRIDS = ((0.2, 161), (0.4, 161), (0.8, 161))


@dataclass(frozen=True)
class EpsilonGrid:
    """One confidence radius and its prescribed opinion-grid resolution."""

    epsilon: float
    grid_size: int

    @property
    def key(self) -> str:
        epsilon = f"{self.epsilon:g}".replace(".", "p")
        return f"epsilon_{epsilon}_b{self.grid_size}"


@dataclass(frozen=True)
class GridCell:
    """One indexed point on the paper's alpha/q rate grid."""

    alpha_index: int
    q_index: int
    alpha: float
    q: float
    diagonal_offset: int


@dataclass(frozen=True)
class ScanCase:
    """One independently checkpointed kinetic trajectory."""

    epsilon: float
    grid_size: int
    dynamics: str
    opinion_method: str
    configuration: str
    recsys: str
    steepness: float
    closure_note: str
    alpha_index: int
    q_index: int
    alpha: float
    q: float
    diagonal_offset: int

    @property
    def key(self) -> str:
        return "/".join(
            (
                EpsilonGrid(self.epsilon, self.grid_size).key,
                self.dynamics,
                self.opinion_method,
                self.configuration,
                f"q{self.q_index:02d}_a{self.alpha_index:02d}",
            )
        )

    def payload(self) -> dict[str, object]:
        return asdict(self)


def parse_epsilon_grids(values: list[str] | None) -> tuple[EpsilonGrid, ...]:
    """Parse ``EPSILON:GRID_SIZE`` pairs and enforce odd grids."""

    raw = values or [f"{epsilon}:{grid}" for epsilon, grid in DEFAULT_EPSILON_GRIDS]
    result = []
    for value in raw:
        try:
            epsilon_text, grid_text = value.split(":", maxsplit=1)
            item = EpsilonGrid(float(epsilon_text), int(grid_text))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid epsilon-grid specification {value!r}; expected EPSILON:GRID"
            ) from error
        if not 0 < item.epsilon <= 2:
            raise ValueError("epsilon values must lie in (0, 2]")
        if item.grid_size < 11 or item.grid_size % 2 == 0:
            raise ValueError("epsilon grid sizes must be odd integers >= 11")
        result.append(item)
    if len({item.epsilon for item in result}) != len(result):
        raise ValueError("epsilon values must not be repeated")
    return tuple(result)


def select_grid_cells(
    rates: np.ndarray,
    *,
    selection: str,
    band_offsets: tuple[int, ...] = (-1, 0, 1),
    explicit_cells: tuple[tuple[int, int], ...] = (),
) -> tuple[GridCell, ...]:
    """Select the anti-diagonal band, full grid, or explicit indexed cells.

    An offset is ``q_index + alpha_index - (n_rates - 1)``. Thus offsets
    ``-1, 0, 1`` give the paper-grid anti-diagonal and its two adjacent
    diagonals (28 cells on the 10 by 10 grid).
    """

    values = np.asarray(rates, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("rates must be a non-empty positive finite vector")
    if selection not in {"anti_diagonal_band", "full", "explicit"}:
        raise ValueError(f"unknown grid selection: {selection}")
    size = values.size
    if selection == "full":
        indices = tuple(
            (q_index, alpha_index)
            for q_index in range(size)
            for alpha_index in range(size)
        )
    elif selection == "anti_diagonal_band":
        if not band_offsets or len(set(band_offsets)) != len(band_offsets):
            raise ValueError("band offsets must be non-empty and unique")
        indices = tuple(
            (q_index, alpha_index)
            for q_index in range(size)
            for alpha_index in range(size)
            if q_index + alpha_index - (size - 1) in band_offsets
        )
    else:
        if not explicit_cells:
            raise ValueError("explicit grid selection requires at least one cell")
        if len(set(explicit_cells)) != len(explicit_cells):
            raise ValueError("explicit cells must not be repeated")
        invalid = [
            pair for pair in explicit_cells if min(pair) < 0 or max(pair) >= size
        ]
        if invalid:
            raise ValueError(f"grid cell indices out of range: {invalid}")
        indices = explicit_cells
    return tuple(
        GridCell(
            alpha_index=alpha_index,
            q_index=q_index,
            alpha=float(values[alpha_index]),
            q=float(values[q_index]),
            diagonal_offset=q_index + alpha_index - (size - 1),
        )
        for q_index, alpha_index in sorted(indices)
    )


def build_cases(
    epsilon_grids: tuple[EpsilonGrid, ...],
    dynamics: tuple[str, ...],
    methods: tuple[str, ...],
    scenarios: tuple[RecommendationScenario, ...],
    cells: tuple[GridCell, ...],
) -> tuple[ScanCase, ...]:
    """Expand the factorial protocol into independently resumable cases."""

    if not dynamics or not methods or not scenarios or not cells:
        raise ValueError("all protocol dimensions must be non-empty")
    return tuple(
        ScanCase(
            epsilon=epsilon_grid.epsilon,
            grid_size=epsilon_grid.grid_size,
            dynamics=dynamics_name,
            opinion_method=method,
            configuration=scenario.key,
            recsys=scenario.recsys,
            steepness=scenario.steepness,
            closure_note=scenario.closure_note,
            alpha_index=cell.alpha_index,
            q_index=cell.q_index,
            alpha=cell.alpha,
            q=cell.q,
            diagonal_offset=cell.diagonal_offset,
        )
        for epsilon_grid in epsilon_grids
        for dynamics_name in dynamics
        for method in methods
        for scenario in scenarios
        for cell in cells
    )


__all__ = [
    "DEFAULT_CONFIGURATIONS",
    "DEFAULT_DYNAMICS",
    "DEFAULT_EPSILON_GRIDS",
    "DEFAULT_METHODS",
    "PAPER_RATES",
    "EpsilonGrid",
    "GridCell",
    "ScanCase",
    "build_cases",
    "parse_epsilon_grids",
    "record_schedule",
    "select_grid_cells",
    "select_scenarios",
]
