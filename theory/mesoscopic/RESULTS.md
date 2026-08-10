# Audited mesoscopic results

All values below were recomputed after two numerical corrections: distance KDE
tails are no longer truncated at 0 and 2, and first threshold crossings are
linearly interpolated rather than rounded to the recording interval.

## Five recommendation kernels

Each resolution contains 320 deterministic trajectories: five kernels and an
8-by-8 influence/rewiring grid. These are pair-closure counterfactuals, not
finite-N simulation replicates.

| kernel | mean Iw 61/81/101 | mean final Ip 61/81/101 |
|---|---|---|
| Random | 0.714 / 0.695 / 0.648 | 0.383 / 0.423 / 0.390 |
| Opinion | 0.709 / 0.650 / 0.668 | 0.644 / 0.870 / 0.797 |
| OpinionM9 | 0.710 / 0.641 / 0.665 | 0.605 / 0.871 / 0.756 |
| Structure | 0.709 / 0.673 / 0.641 | 0.496 / 0.622 / 0.563 |
| StructureM9 | 0.709 / 0.674 / 0.642 | 0.493 / 0.607 / 0.553 |

The robust result is the ordering of mean final polarization at every grid:
the Opinion family is highest, the Structure pair proxy is intermediate, and
Random is lowest. Fine pathway boundaries are not converged. At 81 bins every
kernel has 26 PbS and 38 SbP labels. Exactly one observed crossing receives
precedence; both thresholds are reached in 56 Random, 64 Opinion, 64 OpinionM9,
58 Structure, and 58 StructureM9 cells, and the others are gray in the
precedence heatmap. At 61 bins Opinion changes one label; at 101 bins Random
has 32/32 and both Structure variants have 33/31. The pure-versus-M9
differences are also smaller than the grid sensitivity and should not be
treated as resolved effects.

The `alpha=1` column is a conservative discrete remapping/copying limit, not a
controlled continuous-time point. It is hatched in the combined plot. `I_h`
can also rise because opinion motion makes existing edge endpoints concordant,
so first-passage order is not direct evidence of graph-component fragmentation.

Outputs:

```text
outputs/theory/mesoscopic/recsys_noise_1e-5/
outputs/theory/mesoscopic/recsys_noise_1e-5_grid61/
outputs/theory/mesoscopic/recsys_noise_1e-5_grid101/
outputs/theory/mesoscopic/recsys_grid_convergence/grid_convergence.csv
outputs/theory/mesoscopic/figures/f_kinetic_recsys_comparison_noise_1e-5.{pdf,png}
```

## Independent spectrum check

`spectrum_check.py` injects a relative cosine perturbation of amplitude `1e-6`
into the full nonlinear periodic HK velocity, evaluates the PDE right-hand
side, and projects it onto the injected mode. It does not use the analytical
growth-rate functions in the numerical measurement. For a 1024-point grid,
modes 1--8, and both Random and Opinion kernels, the maximum absolute error is
`2.258e-4`.

```text
outputs/theory/mesoscopic/spectrum_check/spectrum_check.csv
outputs/theory/mesoscopic/spectrum_check/f_spectrum_validation.{pdf,png}
outputs/theory/mesoscopic/spectrum_check/run_metadata.json
```

## Spectrum-selected nonlinear states

`spectrum_steady_states.py` evolves the same periodic nonlinear PDE from the
three initial modes `m=1,2,3`. The 1024-point dealiased RK4 calculation shows
that `m=2` is the fastest of the three for both kernels and settles into two
equal peaks. The `m=1` branch generates an unequal two-peak profile because its
symmetry sector contains the faster `m=2` harmonic. The `m=3` branch generates
six alternating peaks because `m=6` grows faster than `m=3`; Random reaches the
`L1` RHS threshold `2e-8` at `t=376`, while Opinion is still slowly evolving at
the horizon `t=1000` (`L1` residual `1.44e-6`).

The solver retains the exact symmetry sector of each injected mode to prevent
amplified floating-point subharmonics from switching branches, while leaving
all admissible nonlinear harmonics unconstrained. It applies no positivity
clipping or mass renormalization. Relative to 768 points, the 1024-point
Random/Opinion profiles differ in `L1` by `2.44e-3/2.27e-3` for `m=1`, less
than `1.6e-9` for `m=2`, and `1.96e-2/1.07e-2` for `m=3`; fine peak weights on
the `m=3` late-time branch should therefore not be treated as converged.

```text
outputs/theory/mesoscopic/spectrum_steady_states/steady_states.npz
outputs/theory/mesoscopic/spectrum_steady_states/steady_summary.csv
outputs/theory/mesoscopic/spectrum_steady_states/f_spectrum_steady_states.{pdf,png}
outputs/theory/mesoscopic/spectrum_steady_states/run_metadata.json
```

## Verification

The solver and CLI regression tests cover conservative evolution, parameter
guards, failure on material invariant errors, the KDE support convention, and
interpolated crossing times:

```sh
PYTHONPATH=src:. python -m pytest \
  tests/modeling/test_mesoscopic_solver.py \
  tests/unit/test_mesoscopic_cli_utils.py \
  tests/unit/test_spectrum_steady_states.py -q
```

The complete numerical provenance for each current scan is in its
`run_metadata.json` sidecar.
