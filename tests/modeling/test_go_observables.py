from __future__ import annotations

import math

import numpy as np
import pytest

from ehk.modeling.mesoscopic import (
    KineticParameters,
    KineticStopping,
    ObservableResolution,
    ObservableThresholds,
    kinetic_request,
)
from ehk.modeling.mesoscopic.go_kinetic import (
    density_velocity_series,
    multimetric_series,
    observable_series,
)


def resolution() -> ObservableResolution:
    return ObservableResolution(
        population=500,
        opinion_min=-1.0,
        opinion_max=1.0,
        opinion_quadrature_points=5,
        confidence_quadrature_points=5,
        score_max=45,
        distance_grid_size=256,
        minimum_bandwidth=0.01,
        objective_effective_samples=10_000,
    )


def test_request_translation_is_explicit_and_uses_measure_name() -> None:
    parameters = KineticParameters(
        dynamics="deffuant",
        opinion_method="measure",
        recsys="structure_random_l1",
        recommendation_steepness=2.5,
        grid_size=21,
        steps=20,
        record_every=4,
    )
    request = kinetic_request(
        "case", parameters, resolution(), ObservableThresholds(0.5, 0.5)
    )
    assert request["dynamics"]["opinion_method"] == "measure"
    assert request["recommender"]["steepness"] == 2.5
    assert request["resolution"] == {
        "opinion_quadrature_points": 5,
        "opinion_quadrature_rule": "gauss_hermite",
        "confidence_quadrature_points": 5,
        "score_max": 45,
        "distance_grid_size": 256,
    }
    assert request["observables"]["pathway"] is True
    assert request["observables"]["node_energy"] is False
    assert request["observables"]["edge_energy"] is False
    assert request["stopping"]["mode"] == "fixed_steps"
    assert isinstance(request["initial"]["probabilities"], np.ndarray)
    assert request["snapshots"]["record_steps"].shape == (0,)
    assert not any(
        request["snapshots"][name]
        for name in (
            "rho", "edge", "velocity", "rewiring_flux",
            "node_potential", "edge_potential",
        )
    )
    assert request["snapshots"]["final_rho"] is False
    assert request["snapshots"]["final_edge"] is False
    assert request["snapshots"]["final_node_potential"] is False
    assert request["snapshots"]["final_edge_potential"] is False


def test_request_can_select_energy_and_adaptive_stopping() -> None:
    parameters = KineticParameters(grid_size=11, steps=100)
    stopping = KineticStopping(
        mode="state_and_energy",
        minimum_steps=20,
        check_every=5,
        patience_steps=15,
        state_l1_tolerance=1e-6,
        energy_absolute_tolerance=1e-8,
        energy_relative_tolerance=1e-7,
    )
    request = kinetic_request(
        "energy",
        parameters,
        resolution(),
        observable_fields=("node_energy", "edge_energy"),
        final_snapshot_fields=("node_potential", "edge_potential"),
        stopping=stopping,
    )
    assert request["observables"]["node_energy"] is True
    assert request["observables"]["edge_energy"] is True
    assert request["observables"]["pathway"] is False
    assert request["snapshots"]["final_node_potential"] is True
    assert request["snapshots"]["final_edge_potential"] is True
    assert request["stopping"] == {
        "mode": "state_and_energy",
        "minimum_steps": 20,
        "check_every": 5,
        "patience_steps": 15,
        "state_l1_tolerance": 1e-6,
        "energy_absolute_tolerance": 1e-8,
        "energy_relative_tolerance": 1e-7,
    }


def test_observable_response_preserves_unreached_first_passage() -> None:
    response = {
        "request_id": "case",
        "result": {
            "series": {
                "time": np.asarray([0.0, 1.0]),
                "polarization": np.asarray([0.1, 0.2]),
                "subjective": np.asarray([0.3, 0.4]),
                "homophily": np.asarray([0.0, 0.6]),
                "homophily_raw": np.asarray([0.4, 0.8]),
            },
            "summary": {
                "pathway": 0.25,
                "polarization_first_passage": {"reached": False, "time": None},
                "homophily_first_passage": {"reached": True, "time": 1.0},
            },
        },
    }
    result = observable_series(response)
    assert math.isnan(result.polarization_first_passage)
    assert result.homophily_first_passage == 1.0
    np.testing.assert_array_equal(result.polarization, [0.1, 0.2])


def test_request_can_select_only_landscape_snapshot_fields() -> None:
    parameters = KineticParameters(grid_size=5, steps=4)
    request = kinetic_request(
        "landscape",
        parameters,
        resolution(),
        record_steps=(0, 4),
        snapshot_fields=("rho", "velocity"),
        final_snapshot_fields=("rho", "edge"),
    )
    assert request["snapshots"]["rho"] is True
    assert request["snapshots"]["velocity"] is True
    assert request["snapshots"]["edge"] is False
    assert request["snapshots"]["rewiring_flux"] is False
    assert request["snapshots"]["final_rho"] is True
    assert request["snapshots"]["final_edge"] is True


def test_density_velocity_response_validates_snapshot_shapes() -> None:
    parameters = KineticParameters(grid_size=5, steps=4)
    response = {
        "request_id": "landscape",
        "result": {
            "snapshots": {
                "time": [0.0, 4.0],
                "rho": np.ones((2, 5)),
                "velocity": np.zeros((2, 5)),
                "final_rho": np.ones(5) / 5,
                "final_edge": np.ones((5, 5)),
            },
            "diagnostics": {
                "max_node_mass_residual": 1e-12,
                "max_fixed_degree_residual": 2e-12,
            },
        },
    }
    result = density_velocity_series(response, parameters, resolution())
    assert result.rho.shape == (2, 5)
    assert result.final_rho is not None and result.final_rho.shape == (5,)
    assert result.final_edge is not None and result.final_edge.shape == (5, 5)
    assert result.max_fixed_degree_residual == 2e-12


def test_multimetric_response_decodes_energies_and_final_fields() -> None:
    parameters = KineticParameters(grid_size=5, steps=4, record_every=2)
    response = {
        "request_id": "multimetric",
        "result": {
            "series": {
                "time": np.asarray([0.0, 2.0, 4.0]),
                "node_energy": np.asarray([0.03, 0.02, 0.01]),
                "edge_energy": np.asarray([0.06, 0.03, 0.02]),
            },
            "summary": {"pathway": 0.25},
            "snapshots": {
                "time": np.asarray([]),
                "final_rho": np.ones(5) / 5,
                "final_edge": np.ones((5, 5)) * (15 / 25),
            },
            "diagnostics": {
                "max_node_mass_residual": 1e-12,
                "max_fixed_degree_residual": 2e-12,
            },
        },
    }
    result = multimetric_series(response, parameters)
    np.testing.assert_array_equal(result.time, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(result.node_energy, [0.03, 0.02, 0.01])
    assert result.final_rho.shape == (5,)
    assert result.final_edge.shape == (5, 5)
    assert result.pathway == pytest.approx(0.25)
