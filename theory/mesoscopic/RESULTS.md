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
| Random | 0.577 / 0.576 / 0.572 | 0.853 / 0.801 / 0.829 |
| Opinion | 0.659 / 0.651 / 0.653 | 0.786 / 0.832 / 0.799 |
| OpinionM9 | 0.648 / 0.643 / 0.643 | 0.840 / 0.857 / 0.848 |
| L0-Structure | 0.575 / 0.573 / 0.569 | 0.867 / 0.869 / 0.875 |
| L0-StructureM9 | 0.575 / 0.573 / 0.569 | 0.866 / 0.869 / 0.875 |

The only robust family ordering is that L0-Structure has the largest mean final
polarization at every grid. Random and Opinion reverse order under spatial
refinement and cannot be ranked. Fine pathway boundaries are not converged: on
the 81-cell grid, the PbS/SbP/unresolved counts are 32/32/0 for Random,
26/33/5 for Opinion, 26/38/0 for OpinionM9, and 32/32/0 for both L0-Structure
variants. The pure--M9 difference is resolved for Opinion but negligible for
L0-Structure: their 61/81/101-cell mean absolute `Iw` differences are
0.0114/0.0090/0.0102 and 0.00034/0.00034/0.00045, respectively.

The PDE remains well defined at `alpha=1`, but matching that point (or `q=1`)
to a unit-step microscopic update is not a controlled small-step limit. The
`alpha=1` column is hatched in the combined plot. A complete 320-trajectory
rerun at `dt=0.5` changes no pathway classification and changes family mean
final `Ip` by at most `1.7e-4`. `I_h`
can also rise because opinion motion makes existing edge endpoints concordant,
so first-passage order is not direct evidence of graph-component fragmentation.

Outputs:

```text
outputs/theory/mesoscopic/recsys_pde_noise_1e-5/
outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid61/
outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid101/
outputs/theory/mesoscopic/recsys_pde_noise_1e-5_dt0.5/
outputs/theory/mesoscopic/recsys_pde_grid_convergence/grid_convergence.csv
outputs/theory/mesoscopic/figures_pde/f_kinetic_recsys_comparison_pde_noise_1e-5.{pdf,png}
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

## Joint alpha/q spectrum

`joint_spectrum.py` extends the scalar `q=0` calculation to the complete
periodic opinion--edge tangent sector. Since the product-edge state is not
stationary for `q>0`, it reports both frozen eigenvalues along the analytic
rewiring base orbit and time-ordered singular rates over a fixed `T=100`
horizon. This is a deterministic Section 3.6 operator calculation, not a
microscopic ensemble or a Section 3.7 committor.

At `alpha=0.05`, increasing `q` from 0 to 0.3 lowers the fastest resolved
joint rate from `0.0662` to `0.0516` for Random, from `0.0489` to `0.0347` for
Opinion, and from `0.0495` to `0.0353` for OpinionM9. Nevertheless, edge-led
channels remain: for Random at `(alpha,q)=(0.05,0.3)`, modes 4 and 8 have
positive joint rates `0.0121/0.0265` while rho-only seeds have rates
`-0.00951/-0.00171`.

The default 64-point scan includes modes 1--20 and hatches alpha/q cells whose
fastest mode lands on the cutoff. A 20/40/80 time-slice check changes selected
rates by at most `4.7e-5`. Random 64/96-grid differences are small for the
low-to-moderate-alpha resolved band but material at some high modes and
`alpha=0.3`; those locations are diagnostics rather than converged phase
boundaries.

```text
outputs/theory/mesoscopic/joint_spectrum/joint_instantaneous_eigenvalues.csv
outputs/theory/mesoscopic/joint_spectrum/joint_finite_time_singular_values.csv
outputs/theory/mesoscopic/joint_spectrum/joint_alpha_q_summary.csv
outputs/theory/mesoscopic/joint_spectrum/joint_time_step_validation.csv
outputs/theory/mesoscopic/joint_spectrum/f_joint_instantaneous_spectrum.{pdf,png}
outputs/theory/mesoscopic/joint_spectrum/f_joint_finite_time_spectrum.{pdf,png}
outputs/theory/mesoscopic/joint_spectrum/f_joint_alpha_q_spectrum.{pdf,png}
```

The derivation and interpretation are in
[`JOINT_SPECTRUM.md`](JOINT_SPECTRUM.md).

## Verification

The solver and CLI regression tests cover conservative evolution, parameter
guards, failure on material invariant errors, the KDE support convention, and
interpolated crossing times:

```sh
PYTHONPATH=src:. python -m pytest \
  tests/modeling/test_mesoscopic_solver.py \
  tests/unit/test_mesoscopic_cli_utils.py \
  tests/unit/test_joint_spectrum.py \
  tests/unit/test_spectrum_steady_states.py -q
```

The complete numerical provenance for each current scan is in its
`run_metadata.json` sidecar.
