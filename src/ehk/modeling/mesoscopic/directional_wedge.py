"""Directional-wedge (L1) closure for the mesoscopic EHK model.

The L0 solver retains only node and directed-edge masses.  This module adds
four center-opinion-resolved directed wedge channels.  For an endpoint--center
relation, ``out`` means endpoint -> center and ``in`` means center -> endpoint.
The channel order is ``out/out``, ``out/in``, ``in/out``, and ``in/in``.

Reciprocal dyads are not separately identifiable from a directed pair density.
The implementation therefore uses the sparse-graph L1 closure in which their
inclusion--exclusion correction is higher order in ``1 / population``.  This
limitation is explicit: the state is a directional first-moment closure, not a
complete union-neighborhood candidate law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CHANNELS = ("out_out", "out_in", "in_out", "in_in")
_CHANNEL_DIRECTIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass
class DirectionalWedgeState:
    """Four L1 wedge channels with shape ``(4, B, B, B)``.

    The last three axes are endpoint bin, center bin, endpoint bin.  Values
    are ordered-wedge counts per agent, matching the normalization of the
    directed edge mass.
    """

    values: FloatArray

    def validate(self, grid_size: int) -> None:
        expected = (len(CHANNELS), grid_size, grid_size, grid_size)
        if self.values.shape != expected:
            raise ValueError(
                f"directional wedge has shape {self.values.shape}, expected {expected}"
            )
        if not np.all(np.isfinite(self.values)):
            raise FloatingPointError("directional wedge contains non-finite values")
        if float(np.min(self.values)) < -1e-10:
            raise FloatingPointError("directional wedge contains negative mass")
        if np.any(self.values < 0):
            self.values[self.values < 0] = 0.0

    def union_score_mass(self) -> FloatArray:
        """Return endpoint-block first common-neighbor score mass.

        Summing the four channels is the sparse-graph union-neighborhood
        closure.  Exact finite-``N`` union scores additionally require
        reciprocal-dyad inclusion--exclusion terms.
        """

        return np.sum(self.values, axis=(0, 2))


def independent_directional_wedge(
    rho: FloatArray,
    edge: FloatArray,
) -> DirectionalWedgeState:
    """Return the independent-edge L1 target implied by ``(rho, edge)``."""

    size = rho.size
    per_center = np.zeros((2, size, size), dtype=float)
    # Endpoint -> center: edge[i, r] / rho[r].
    per_center[0] = np.divide(
        edge,
        rho[None, :],
        out=np.zeros_like(edge),
        where=rho[None, :] > 1e-15,
    )
    # Center -> endpoint: edge[r, i] / rho[r], stored as [i, r].
    per_center[1] = np.divide(
        edge.T,
        rho[None, :],
        out=np.zeros_like(edge),
        where=rho[None, :] > 1e-15,
    )

    values = np.empty((len(CHANNELS), size, size, size), dtype=float)
    for channel, (left_direction, right_direction) in enumerate(_CHANNEL_DIRECTIONS):
        left = per_center[left_direction]
        right = per_center[right_direction]
        block = np.einsum("ir,jr,r->irj", left, right, rho, optimize=True)
        # Distinct endpoints may occupy the same opinion bin.  Identifying
        # and removing the one literal self-pair requires a finite-population
        # mark not present in the density state and is O(1 / N) here.
        values[channel] = np.maximum(block, 0.0)
    return DirectionalWedgeState(values)


def mix_after_rewiring(
    state: DirectionalWedgeState,
    rho: FloatArray,
    edge_before: FloatArray,
    edge_after: FloatArray,
) -> DirectionalWedgeState:
    """Mix retained L1 wedges with the new independent-edge target.

    The gross edge-event incidence is not available from the deterministic
    net rewiring flux.  We therefore use direction-specific incident block
    turnover measured by ``abs(edge_after - edge_before)``.  This is an
    explicit moment closure, kept separate from the exact transport step.
    """

    target = independent_directional_wedge(rho, edge_after).values
    change = np.abs(edge_after - edge_before)
    incoming_denominator = np.sum(edge_before, axis=0)
    outgoing_denominator = np.sum(edge_before, axis=1)
    incoming_change = np.divide(
        np.sum(change, axis=0),
        incoming_denominator,
        out=np.zeros_like(rho),
        where=incoming_denominator > 1e-15,
    )
    outgoing_change = np.divide(
        np.sum(change, axis=1),
        outgoing_denominator,
        out=np.zeros_like(rho),
        where=outgoing_denominator > 1e-15,
    )
    # For endpoint -> center wedges, the relation is incident to the center's
    # incoming edges.  For center -> endpoint wedges, it is incident to the
    # center's outgoing edges.
    direction_change = np.stack((incoming_change, outgoing_change))
    direction_change = np.clip(direction_change, 0.0, 1.0)

    output = np.empty_like(state.values)
    for channel, (left_direction, right_direction) in enumerate(_CHANNEL_DIRECTIONS):
        persistence = (1.0 - direction_change[left_direction]) * (
            1.0 - direction_change[right_direction]
        )
        output[channel] = (
            persistence[None, :, None] * state.values[channel]
            + (1.0 - persistence[None, :, None]) * target[channel]
        )
    return DirectionalWedgeState(np.maximum(output, 0.0))


def transport_directional_wedge(
    state: DirectionalWedgeState,
    transport_axis,
) -> DirectionalWedgeState:
    """Apply one conservative opinion transport operator to all three nodes.

    ``transport_axis(values, axis)`` must apply the same one-dimensional
    conservative update used for node and edge endpoints.
    """

    values = state.values
    for axis in (1, 2, 3):
        values = transport_axis(values, axis)
    return DirectionalWedgeState(np.maximum(values, 0.0))
