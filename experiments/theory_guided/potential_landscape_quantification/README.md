# Potential-landscape quantification

This workflow turns the mesoscopic effective-potential curves into numerical
observables.  For every recorded time it identifies the dominant pair of
stable wells and reports their positions, separation, depths, local
curvatures, the intervening barrier, and its curvature.  Persistent
first-passage times are reported for initial double-well formation and for
barrier heights 0.01, 0.05, and half the final barrier.

It reuses the four parameter cases and L0/L1 closures from
`theory/mesoscopic/l1_comparison.py`, the finite-volume solver, the pathway
indices, and the common coefficient-free potential convention.  The default
recording schedule is every step through time 200 and every 20 steps
thereafter, avoiding the old `t=20` lower bound on formation times.

Run the paper-scale experiment with:

```bash
PYTHONPATH=src:. python -m \
  experiments.theory_guided.potential_landscape_quantification.run
```

Use `--grid-size 21 --steps 20 --cases balanced --closures l0` for a quick
workflow check.
