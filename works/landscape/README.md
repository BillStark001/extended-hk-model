# Counterfactual potential landscape

This directory replaces the observational, time-window force proxy with the
frozen-state counterfactual probe implemented by `social-media-models`.

See [`RESULTS.md`](RESULTS.md) for the completed run's numerical summary.

The default run covers all 11 records used by
`f_mech_map_baseline.py` and `f_supp_mech_map.py`: baseline, PbS, SbP,
influence, retweet, opinion recommendation, and phases 1--5.

The active steps come from the paper's existing definition
(`active_threshold=0.98`, `min_inactive_value=0.75`). Each record is evaluated
at `t_n = 0.0, 0.1, ..., 1.0`, mapped to the nearest integer simulation step.
Both requested and realized normalized times are stored.

This is an 11-point frozen-state counterfactual analysis. It is intentionally
different from the old plot's ten time-window KDE estimates
(`[0,.1), ... ,[.9,1]`): every new curve asks what force a hypothetical opinion
would experience at one exact frozen network/feed state.

## Run

Build the probe binary and make the local bindings importable:

```bash
cd ../social-media-models
make build-probe
pip install -e .

cd ../extended-hk-model
python -m works.landscape.counterfactual_probe
```

The default input directory is the repository-local `run_mech`. Use
`--workspace` when the same records live elsewhere; this command does not
depend on `sim_ws.json` or the general experiment configuration.

Defaults:

- opinion grid: `[-1, 1]` with step `0.01` (201 points);
- 20 recommendation replicates per anchor agent;
- all 500 agents as anchors;
- a fixed common probe RNG for every scenario.

The fixed common RNG makes the run reproducible and reduces Monte Carlo noise
in cross-scenario comparisons. Probe randomness is isolated from simulation
randomness.

Useful overrides:

```bash
python -m works.landscape.counterfactual_probe \
  --scenario pbs \
  --replicates 40 \
  --grid-step 0.01 \
  --probe-binary ../social-media-models/smp-probe
```

## Outputs

`fig/landscape/` contains:

- `counterfactual_probe_<scenario>.npz` for all 11 mechanism records;
- `manifest.json`;
- `counterfactual_probe_landscape.png`;
- `counterfactual_probe_landscape.pdf`.

Each NPZ stores total, neighbor, and recommendation force means/variances,
concordant counts, sample/activity counts, and the integrated zero-mean
potential. The Go evaluator's `f_probe` is the actual expected one-step drift
and therefore already includes HK `Influence` (`alpha`). The paper's existing
NOD landscape removes that update-rate coefficient. Both scales are stored:

- `f_probe_*` and `drift_potential`: actual one-step drift scale;
- `social_force_* = f_probe_* / alpha` and `potential`: the paper-comparable
  coefficient-free scale used by the generated figure.

The displayed potential follows

```text
F_CF(x, t_n) = f_probe(x, t_n) / alpha
V_CF(x, t_n) = - integral F_CF(x, t_n) dx.
```

## Mechanism-map landscape selection

Both mechanism-map entry points accept
`--landscape {observational,probe}`. This option changes only the potential
estimator: the NOD/FOD panels continue to show the original trajectories of
all agents. Probe mode uses the total counterfactual force for NOD potential
and its neighbor component for FOD potential.

To redraw all paper mechanism maps with probe potentials:

```bash
python -m works.landscape.f_mech_map_baseline \
  --version 0 --landscape probe
python -m works.landscape.f_mech_map_baseline \
  --version 1 --landscape probe
python -m works.landscape.f_supp_mech_map \
  --landscape probe
```

The outputs replace the corresponding PDF/PNG pairs under `fig/landscape/`.
Omit `--landscape probe` (or select `observational`) to reproduce the legacy
time-window KDE potentials.

To draw a standalone comparison from saved NPZ files without rerunning the
probe:

```bash
python -m works.landscape.plot_counterfactual_landscape \
  fig/landscape/counterfactual_probe_pbs.npz \
  fig/landscape/counterfactual_probe_sbp.npz
```

The landscape programs now live together:

- `f_mech_map_baseline.py`: main empirical baseline/network figure;
- `f_supp_mech_map.py`: supplementary observational KDE landscapes;
- `f_mech_explain.py`: analytic ideal-state landscapes.

They also write their PDF/PNG pairs beneath `fig/landscape/`.
