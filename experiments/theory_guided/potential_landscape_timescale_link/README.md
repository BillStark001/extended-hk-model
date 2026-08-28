# Potential-landscape link to the macroscopic time-scale scan

This targeted experiment reuses the seven recommendation scenarios and the
`B=81`, 4000-step recording protocol of the macroscopic time-scale scan. It
solves two seven-case designs:

1. `common_rates`: every scenario at `alpha=q=0.1`;
2. `transition_center`: `alpha=0.1`, with `q` set by each scenario's fitted
   raw `q/alpha` transition center.

Each completed case is an atomic NPZ checkpoint containing `rho`, `velocity`,
the coefficient-free force and potential, pathway indices, and quantitative
landscape diagnostics. Re-running the command validates the protocol and
skips completed cases.

```bash
PYTHONPATH=src:. python -m \
  experiments.theory_guided.potential_landscape_timescale_link.run \
  --transition-offsets /path/to/macroscopic_timescale_ratio/transition_offsets.csv \
  --output-dir /path/to/potential_landscape_timescale_link \
  --jobs 4
```

Regenerate analysis without solving:

```bash
PYTHONPATH=src:. python -m \
  experiments.theory_guided.potential_landscape_timescale_link.analyze \
  --output-dir /path/to/potential_landscape_timescale_link
```
