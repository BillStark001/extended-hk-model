# Paper-aligned macroscopic time-scale scan

This experiment reruns the paper Figure 2 rate grid at `B=81` for seven
recommendation closures and evaluates the operator-resolved competition ratio
at two macro-progress levels,

`Gamma(u) = [d I_h / dt]_rewiring,+ / [d I_p / dt]_opinion,+`,

where `u = I_p + I_h`.  The requested diagnostics are `Gamma(0)` and
`Gamma(0.1)`.  The latter is interpolated between densely recorded early
states; it is not the older integrated-window ratio.

The L1, zeta=4 column is explicitly a mean-power closure diagnostic.  The
state retains the first structural-score moment and evaluates the nonlinear
weight as `E[S]^4`; it does not retain `E[S^4]`.

Run the scan (safe to resume in the same output directory):

```bash
PYTHONPATH=src:. python -m experiments.theory_guided.macroscopic_timescale_ratio.run \
  --grid-size 81 --jobs 8 --output-dir /path/to/artifacts/macroscopic_timescale_ratio
```

Regenerate tables and figures without solving trajectories:

```bash
PYTHONPATH=src:. python -m experiments.theory_guided.macroscopic_timescale_ratio.analyze \
  --output-dir /path/to/artifacts/macroscopic_timescale_ratio
```

Each completed cell is atomically checkpointed under `cells/`.  A protocol
digest prevents accidental resume across incompatible parameters.
