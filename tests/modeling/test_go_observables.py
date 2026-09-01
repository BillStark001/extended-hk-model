from __future__ import annotations

import math

import numpy as np

from ehk.modeling.mesoscopic import (
    KineticParameters,
    ObservableResolution,
    ObservableThresholds,
    kinetic_request,
)
from ehk.modeling.mesoscopic.go_kinetic import observable_series


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
    assert isinstance(request["initial"]["probabilities"], np.ndarray)
    assert request["snapshots"]["record_steps"].shape == (0,)
    assert not any(
        request["snapshots"][name]
        for name in ("rho", "edge", "velocity", "rewiring_flux")
    )


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
