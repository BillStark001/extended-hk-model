import json
from collections import Counter

import numpy as np

from experiments.theory_guided.terminal_probability.analyze import (
    comparison_case,
    summarize,
    terminal_peak_count,
)
from experiments.theory_guided.terminal_probability.run import main as run_main
from experiments.theory_guided.terminal_probability.scenarios import (
    PAPER_COMPARISON_CASES,
    build_scenarios,
    load_config,
    preset_path,
)


CONFIG = preset_path("paper-figure3")


def test_figure3_config_resolves_twenty_conditions_with_forty_runs_each() -> None:
    config = load_config(CONFIG)
    scenarios = build_scenarios(config)
    groups = Counter(
        (
            row["HKParams"]["Influence"],
            row["HKParams"]["RewiringRate"],
            row["RecsysFactoryType"],
            row["RecSysParams"]["Steepness"],
        )
        for row in scenarios
    )
    assert len(scenarios) == 4 * 5 * 40
    assert len(groups) == 20
    assert set(groups.values()) == {40}


def test_common_random_numbers_and_exact_recommender_parameters() -> None:
    config = load_config(CONFIG)
    scenarios = build_scenarios(config)
    first_cell = scenarios[:5]
    assert len({json.dumps(row["RNG"], sort_keys=True) for row in first_cell}) == 1
    assert [row["RecsysFactoryType"] for row in first_cell] == [
        "Random",
        "OpinionRandom",
        "OpinionRandom",
        "StructureRandom",
        "StructureRandom",
    ]
    assert [row["RecSysParams"]["Steepness"] for row in first_cell] == [
        1.0, 1.0, 4.0, 1.0, 4.0,
    ]
    assert all(
        row["NodeCount"] == 500 and row["NodeFollowCount"] == 15
        for row in first_cell
    )


def test_human_presets_have_expected_sizes() -> None:
    expected = {"smoke": 5, "paper-figure3": 800}
    for preset, count in expected.items():
        assert len(build_scenarios(load_config(preset_path(preset)))) == count


def test_dry_run_accepts_named_preset_without_binary(capsys) -> None:
    run_main(["paper-figure3", "--binary", "/does/not/exist", "--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert output["scenario_count"] == 800
    assert output["rate_cell_count"] == 4


def test_comparison_cases_match_spectrum_and_micro_coordinates() -> None:
    assert [
        comparison_case(case.alpha, case.rewiring)
        for case in PAPER_COMPARISON_CASES
    ] == [case.key for case in PAPER_COMPARISON_CASES]


def test_terminal_component_classifier_separates_censoring_from_failure() -> None:
    one_cluster = np.linspace(-0.05, 0.05, 500)
    two_clusters = np.concatenate([
        np.linspace(-0.65, -0.55, 250),
        np.linspace(0.55, 0.65, 250),
    ])
    assert terminal_peak_count(one_cluster, 0.45) == 1
    assert terminal_peak_count(two_clusters, 0.45) == 2

    summary = summarize([
        {
            "configuration": "random", "case": "balanced",
            "alpha": 0.05, "q": 0.05, "status": "absorbed", "category": "k2",
        },
        {
            "configuration": "random", "case": "balanced",
            "alpha": 0.05, "q": 0.05, "status": "censored", "category": "censored",
        },
        {
            "configuration": "random", "case": "balanced",
            "alpha": 0.05, "q": 0.05, "status": "unfinished", "category": "",
        },
    ])
    row = summary.iloc[0]
    assert row["p_k2"] == 0.5
    assert row["p_censored"] == 0.5
    assert row["failure_fraction"] == 1 / 3
    assert sum(row[column] for column in (
        "p_k1", "p_k2", "p_k3", "p_k4plus", "p_censored"
    )) == 1.0
