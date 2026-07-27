# Counterfactual probe results

The completed run uses 201 counterfactual opinion values, 11 frozen times,
500 anchor agents, and 20 recommendation replicates per anchor (10,000 samples
per curve point).

| Record | `Influence` | `active_step` | max `|F_CF|` |
|---|---:|---:|---:|
| baseline | 0.05 | 352 | 0.4453 |
| PbS baseline | 0.05 | 786 | 0.4496 |
| SbP baseline | 0.005 | 2572 | 0.4490 |
| +influence | 0.1 | 441 | 0.4471 |
| +retweet | 0.05 | 3249 | 0.4491 |
| opinion rec. | 0.05 | 605 | 0.4494 |
| phase 1 | 0.005 | 2254 | 0.2358 |
| phase 2 | 0.005 | 13192 | 0.4472 |
| phase 3 | 0.1 | 301 | 0.4491 |
| phase 4 | 1.0 | 679 | 0.4487 |
| phase 5 | 1.0 | 14 | 0.4486 |

The exact integer steps corresponding to `t_n = 0, 0.1, ..., 1` are stored in
each NPZ and in `manifest.json`.

The raw one-step drift scales differ by approximately tenfold, as expected
from the tenfold `Influence` difference. After conversion to the paper's
coefficient-free social-force convention, both reach nearly identical maximum
absolute force (`0.4496` for PbS and `0.4490` for SbP).

The counterfactual landscapes distinguish the pathway timing:

- PbS has effectively reached a stable double well by `t_n = 0.1`, with final
  minima near `x = -0.49` and `x = 0.39`.
- SbP forms and deepens the wells gradually; its final minima are near
  `x = -0.41` and `x = 0.43`.

Unlike the old time-window KDE estimator, the frozen-state probe evaluates
hypothetical opinions even in sparsely populated regions. Its late-time
landscape therefore retains deep wells instead of automatically flattening
where observational support disappears. Any paper text attributing late-time
well collapse to the model should be revisited before replacing the existing
figure; that behavior can be an estimator-support effect.

The four mechanism-map figures were redrawn with these probe potentials. Their
NOD/FOD panels retain the legacy all-agent trajectories; only the adjacent
potential panels switch estimators.

Validation:

- all result values are finite;
- every force point has 10,000 samples;
- `f_probe = f_neighbor + f_recommendation` to within `5e-15` across all
  scenarios;
- zero-mean potential normalization is accurate to within `3e-17`;
- all four PDF/PNG mechanism-map pairs were visually inspected after redraw.
