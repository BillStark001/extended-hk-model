# Counterfactual potential landscape

This directory replaces the observational, time-window force proxy with the frozen-state counterfactual probe implemented by `social-media-models`.

See [`RESULTS.md`](RESULTS.md) for the completed run's numerical summary.

The default run covers all 11 records used by `f_mech_map_baseline.py` and `f_supp_mech_map.py`: baseline, PbS, SbP, influence, retweet, opinion recommendation, and phases 1--5.

The active steps come from the paper's existing definition (`active_threshold=0.98`, `min_inactive_value=0.75`). Each record is evaluated at `t_n = 0.0, 0.1, ..., 1.0`, mapped to the nearest integer simulation step. Both requested and realized normalized times are stored.

This is an 11-point frozen-state counterfactual analysis. It is intentionally different from the old plot's ten time-window KDE estimates (`[0,.1), ... ,[.9,1]`): every new curve asks what force a hypothetical opinion would experience at one exact frozen network/feed state.

## Run

Build the probe binary and make the local bindings importable:

``` bash
cd ../social-media-models
make build-probe
pip install -e .

cd ../extended-hk-model
python -m experiments.theory_guided.social_force_probe.run
```

The default input directory is `outputs/experiments/paper/mechanism_cases/raw`. Set `EHK_SOCIAL_FORCE_INPUT_DIR` in `.env` or use `--workspace` when the records live elsewhere.

Defaults:

- opinion grid: `[-1, 1]` with step `0.01` (201 points);
- 20 recommendation replicates per anchor agent;
- all 500 agents as anchors;
- a fixed common probe RNG for every scenario.

The fixed common RNG makes the run reproducible and reduces Monte Carlo noise in cross-scenario comparisons. Probe randomness is isolated from simulation randomness.

Useful overrides:

``` bash
python -m experiments.theory_guided.social_force_probe.run \
  --scenario pbs \
  --replicates 40 \
  --grid-step 0.01 \
  --probe-binary ../social-media-models/smp-probe
```

## Outputs

`outputs/experiments/theory_guided/social_force_probe/` contains:

- `counterfactual_probe_<scenario>.npz` for all 11 mechanism records;
- `manifest.json`;
- `figures/counterfactual_probe_landscape.png`;
- `figures/counterfactual_probe_landscape.pdf`.

Each NPZ stores total, neighbor, and recommendation force means/variances, concordant counts, sample/activity counts, and the integrated zero-mean potential. The Go evaluator's `f_probe` is the actual expected one-step drift and therefore already includes HK `Influence` (`alpha`). The paper's existing NOD landscape removes that update-rate coefficient. Both scales are stored:

- `f_probe_*` and `drift_potential`: actual one-step drift scale;
- `social_force_* = f_probe_* / alpha` and `potential`: the paper-comparable coefficient-free scale used by the generated figure.

The displayed potential follows

```
F_CF(x, t_n) = f_probe(x, t_n) / alpha
V_CF(x, t_n) = - integral F_CF(x, t_n) dx.
```

## Mechanism-map landscape selection

Both mechanism-map entry points accept `--landscape {observational,probe}`. This option changes only the potential estimator: the NOD/FOD panels continue to show the original trajectories of all agents. Probe mode uses the total counterfactual force for NOD potential and its neighbor component for FOD potential.

To redraw all paper mechanism maps with probe potentials:

``` bash
python -m experiments.paper.mechanism_cases.plot_baseline \
  --version 0 --landscape probe
python -m experiments.paper.mechanism_cases.plot_baseline \
  --version 1 --landscape probe
python -m experiments.paper.mechanism_cases.plot_supplement \
  --landscape probe
```

The outputs replace the corresponding PDF/PNG pairs under `outputs/experiments/paper/mechanism_cases/figures/`. Omit `--landscape probe` (or select `observational`) to reproduce the legacy time-window KDE potentials.

To draw a standalone comparison from saved NPZ files without rerunning the probe:

``` bash
python -m experiments.theory_guided.social_force_probe.plot \
  outputs/experiments/theory_guided/social_force_probe/counterfactual_probe_pbs.npz \
  outputs/experiments/theory_guided/social_force_probe/counterfactual_probe_sbp.npz
```

The landscape programs are separated by scientific role:

- `experiments.paper.mechanism_cases.plot_baseline`: main empirical baseline/network figure;
- `experiments.paper.mechanism_cases.plot_supplement`: supplementary observational KDE landscapes;
- `theory.force_landscape.plot`: analytic ideal-state landscapes.

The paper figures go to `outputs/experiments/paper/mechanism_cases/figures/`; the analytic figure goes to `outputs/theory/force_landscape/figures/`.
