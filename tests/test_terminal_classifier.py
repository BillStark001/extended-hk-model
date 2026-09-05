import json
from pathlib import Path

import numpy as np
import pytest

from ehk.terminal import (
    ClassifierOptions,
    classify_atomic_measure,
    classify_opinions,
)


FIXTURES = Path(__file__).parent / "fixtures" / "terminal_classifier_cases.json"


def test_cross_language_classifier_fixtures() -> None:
    for case in json.loads(FIXTURES.read_text(encoding="utf-8")):
        result = classify_atomic_measure(
            case["positions"],
            case["masses"],
            ClassifierOptions(**case["options"]),
        )
        assert {
            key: result[key]
            for key in ("status", "category", "k_all", "k_major")
        } == {
            key: case[key]
            for key in ("status", "category", "k_all", "k_major")
        }
        assert len(result["components"]) == len(case["components"])
        for actual, expected in zip(result["components"], case["components"], strict=True):
            assert actual["major"] is expected["major"]
            for key in ("minimum", "maximum", "mass"):
                assert actual[key] == pytest.approx(expected[key], abs=1e-12)
        for key, expected in case["margins"].items():
            assert result["margins"][key] == pytest.approx(expected, abs=1e-12)


def test_empirical_measure_and_atomic_measure_agree() -> None:
    opinions = np.array([-0.72, -0.68, 0.68, 0.72])
    empirical = classify_opinions(opinions, 0.45, 0.05)
    atomic = classify_atomic_measure(
        opinions,
        np.full(4, 0.25),
        ClassifierOptions(0.45, 0.125, 0.05),
    )
    assert empirical == atomic


def test_resolution_marks_threshold_comparison_ambiguous() -> None:
    result = classify_atomic_measure(
        [-0.3, 0.2],
        [0.5, 0.5],
        ClassifierOptions(0.45, 0.001, 0.05, position_resolution=0.1),
    )
    assert result["status"] == "grid_ambiguous"
    assert result["category"] == "censored"
