# Paper-aligned macroscopic time-scale scan

This experiment reruns a seven-closure rate grid at `B=161` by default and
evaluates the operator-resolved initial competition ratio

`Gamma(u) = [d I_h / dt]_rewiring,+ / [d I_p / dt]_opinion,+`,

where `u = I_p + I_h`.  The reported diagnostic is `Gamma(0)`, evaluated at
the common random-mixing initial state.  Configuration transition centers are
fit to the same continuous `I_w` shown in the heatmaps; the older binary
first-passage response is no longer mixed into this plot.

The L1, zeta=4 column is explicitly a mean-power closure diagnostic.  The
state retains the first structural-score moment and evaluates the nonlinear
weight as `E[S]^4`; it does not retain `E[S^4]`.

Run the scan (safe to resume in the same output directory):

```bash
PYTHONPATH=src:. python -m experiments.theory_guided.macroscopic_timescale_ratio.run \
  --grid-size 161 --dynamics hk --opinion-method measure --epsilon 0.45 \
  --jobs 8 --output-dir /path/to/artifacts/macroscopic_timescale_ratio
```

Regenerate tables and figures without solving trajectories:

```bash
PYTHONPATH=src:. python -m experiments.theory_guided.macroscopic_timescale_ratio.analyze \
  --output-dir /path/to/artifacts/macroscopic_timescale_ratio
```

Each completed cell is atomically checkpointed under `cells/`.  A protocol
digest prevents accidental resume across incompatible parameters.

Audit an interrupted output directory without starting any worker:

```bash
PYTHONPATH=src:. python -m experiments.theory_guided.macroscopic_timescale_ratio.run \
  --grid-size 161 --jobs 8 --output-dir /path/to/artifacts/macroscopic_timescale_ratio \
  --dry-run
```

Rerunning the original command validates the protocol and skips every valid
cell checkpoint.  A `Ctrl-C` terminates only this experiment's worker pool;
already renamed JSON checkpoints remain usable.
