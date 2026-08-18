# Terminal-probability workflow

This task owns the configuration-matched comparison between the joint linear
spectrum and finite SMP simulations. It follows the repository's other
`experiments/theory_guided/<task>` packages: reusable scenario definitions live
in `scenarios.py`, machine paths are registered in `paths.py`, and each stage
has a module entry point.

The workflow does **not** identify the linear proxy with an exact terminal
committor. The spectrum predicts a largest amplified Fourier mode above an
onset threshold; the microscopic row independently counts major terminal KDE
peaks. Comparing the two rows tests that mapping.

## Entry points

| Module | Purpose | Starts Go simulations? |
| --- | --- | --- |
| `run` | Resolve a preset/config and call the production SMP runner | yes, unless `--dry-run` |
| `analyze` | Classify completed full-run directories and aggregate probabilities | no |
| `spectrum` | Compute the linear mode-race proxy and render the two-row figure | no |

Run all commands from the `extended-hk-model` repository root. An editable
installation (`pip install -e .`) and `smp_bindings` from
`social-media-models` are expected, as elsewhere in this repository.

## Presets

Inspect the installed presets without starting anything:

```bash
python -m experiments.theory_guided.terminal_probability.run --list-presets
```

| Preset | Resolved simulations | Intended use |
| --- | ---: | --- |
| `smoke` | 5 | Runtime/output check: one short run per recommender column |
| `paper-comparison4` | 800 | Four displayed rate cases × five columns × 40 replicates |
| `paper-grid10` | 20,000 | Full 10×10 logarithmic rate grid × five columns × 40 replicates |

`paper-comparison4` reuses the corresponding `paper-grid10` RNG namespaces,
so it is an exact initial-condition subset rather than merely a new ensemble
at the same four parameter pairs.

The full grid uses

```text
alpha, q = 10^(-3), 10^(-8/3), ..., 10^0
```

and the five columns Random, OpinionRandom at `zeta=1,4`, and exact microscopic
StructureRandom at `zeta=1,4`. Within each rate cell and replicate, all five
columns share the same root RNG specification (common random numbers).

## Validate and run

Dry-run the production preset. This validates and prints the resolved first
and last scenarios without creating an output directory:

```bash
python -m experiments.theory_guided.terminal_probability.run \
  paper-grid10 --dry-run
```

The production command agreed for the theory paper is:

```bash
python -m experiments.theory_guided.terminal_probability.run \
  paper-grid10 \
  --output-dir /Volumes/DataT1/ehk_theory_run/go_simulator/terminal_probability_grid10 \
  --concurrency 8
```

For an actual five-run integration check, replace `paper-grid10` with `smoke`
and choose a disposable output directory. Unlike `--dry-run`, that command
does start the Go simulator.

The default output root can be changed with
`EHK_TERMINAL_PROBABILITY_OUTPUT_DIR`. `--output-dir` has priority. The SMP
binary follows the repository-wide `SMP_BINARY_PATH` setting and can also be
overridden by `--binary`.

Before starting simulations, `run` writes:

- `sweep_config.json`: exact selected preset/custom configuration;
- `resolved_scenarios.jsonl`: complete SMP metadata for every planned run;
- `resolved_sweep.json`: provenance, counts, binary, and config hash;
- `logfile.log`: production runner log.

An existing output directory is accepted only when its saved configuration is
identical, which keeps resume behavior from mixing two experiments.

## Analyze completed simulations

The analysis reads the resolved manifest and only the last opinion row from
each completed SMP accumulative state. Missing or unfinished runs remain an
explicit `p_incomplete` category rather than being silently dropped.

```bash
python -m experiments.theory_guided.terminal_probability.analyze \
  /Volumes/DataT1/ehk_theory_run/go_simulator/terminal_probability_grid10 \
  --jobs 8 --require-complete
```

It writes `analysis/microscopic_terminal_runs.csv`,
`analysis/microscopic_terminal_summary.csv`, and `analysis_metadata.json`.
Use `--limit N` only to debug the loading/classification pipeline; a limited
summary is not a paper result.

## Compute and plot the spectrum

Generate the upper row and a lower-row placeholder under the repository-local
output tree:

```bash
python -m experiments.theory_guided.terminal_probability.spectrum
```

After the microscopic sweep is complete, fill the lower row with:

```bash
python -m experiments.theory_guided.terminal_probability.spectrum \
  --microscopic-summary \
    /Volumes/DataT1/ehk_theory_run/go_simulator/terminal_probability_grid10/analysis/microscopic_terminal_summary.csv \
  --output-dir outputs/experiments/theory_guided/terminal_probability/comparison
```

The spectrum command accepts `--population`, `--horizon`, `--time-steps`,
`--threshold`, `--modes`, and `--grid-size`. Defaults are the declared paper
proxy settings and are recorded in `run_metadata.json`.

## Custom configurations

Pass `--config path/to/config.json` instead of a preset. A configuration must
define exactly one parameter-space form:

- `rates`: a Cartesian grid, with either explicit `values` or logarithmic
  `minimum`, `maximum`, and `levels` for influence and rewiring;
- `cases`: a list of named `(influence, rewiring)` pairs.

Start from one of the versioned files in `configs/`. Keep configuration and
case keys lowercase and filesystem-safe; they become part of each SMP
`UniqueName`.
