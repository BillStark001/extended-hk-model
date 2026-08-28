# Multibarrier dynamics and solver comparison

This experiment separates two opinion dynamics from two numerical
representations:

- `hk/nonlocal_jump`: deterministic conditional-mean push-forward;
- `deffuant/nonlocal_jump`: full random-neighbor compromise kernel;
- `hk/fokker_planck`: first-moment drift plus the squared deterministic-jump
  second moment;
- `deffuant/fokker_planck`: the same drift plus the full random-neighbor raw
  second moment.

The Deffuant Fokker--Planck coefficient uses

\[
D_{\mathrm{HK}}(x,t)
=\frac{\Delta t\,\alpha^2}{2}
\mathbb{E}[Y-X\mid X=x]^2,
\qquad
D_{\mathrm{D}}(x,t)
=\frac{\Delta t\,\alpha^2}{2}
\mathbb{E}[(Y-X)^2\mid X=x],
\]

under the convention
`partial_t rho = -partial_x(A rho) + partial_xx(D rho)`. Their difference is
proportional to the conditional variance. The nonlocal operators retain every
jump moment and are therefore the reference when
`alpha` is not small; the Fokker--Planck operators are controlled moment
truncations rather than exact substitutes.

Both representations use the same cell-integrated confidence geometry. The
raw second moment and Deffuant jump destinations use cached five-point
Gauss--Legendre quadrature. This cost is paid once for each new
`(B, epsilon, dt * alpha)` tuple in a worker, rather than at every time step.

## Default protocol

The default run uses:

- `epsilon=0.2, B=121`; `epsilon=0.4/0.8, B=81`;
- HK and Deffuant dynamics;
- nonlocal-jump and Fokker--Planck operators;
- Random, OpinionRandom (`zeta=1/4`), and L0-StructureRandom
  (`zeta=1/4`);
- the paper's 10 by 10 alpha/q anti-diagonal and offsets `-1,0,1` (28
  cells).

This is 1,680 independently checkpointed trajectories. Every `.npz` case is
written via an atomic rename. Re-running the identical command verifies the
protocol digest, loads completed cases, and submits only missing cases.

Run the scan, automatic four-way point selection, plots, and timing estimate:

```bash
./experiments/theory_guided/multibarrier_dynamics_comparison/run_experiment.sh
```

An output directory may be passed as the first argument. `JOBS` and
`BENCHMARK_POINTS` are optional environment variables.

For a dry-run protocol check:

```bash
env PYTHONPATH=src:. python -m \
  experiments.theory_guided.multibarrier_dynamics_comparison.run \
  --dry-run --output-dir /path/to/output
```

Use `--grid-selection full` for all 100 paper-grid cells. L1 remains
available by adding `structure_random_l1_zeta1` and
`structure_random_l1_zeta4` to `--configurations`; it is excluded from the
default factorial protocol because it did not materially change the earlier
result and substantially increases state size.

## Multibarrier interpretation

At each record the code identifies every potential minimum, assigns density
mass to adjacent saddle-bounded basins, and ranks each adjacent barrier by

\[
4m_L(1-m_L)\min(\Delta V_L,\Delta V_R).
\]

This suppresses deep but macroscopically irrelevant barriers that isolate a
negligible-mass basin. Wells receive persistent IDs by position/basin-overlap
matching. A peak followed by a decline is classified as barrier annihilation,
dominant-pair switching, relaxation of the same pair, or no robust overshoot.
Thus multiple barriers at `epsilon=0.2` are retained instead of being forced
into one double-well curve.

`selected_comparison_points.csv` contains, for every epsilon, the largest
method disagreement, largest dynamics disagreement, strongest multibarrier
case, and an agreement control. The paired density and barrier figures then
show all four operators at those exact physical points.

The benchmark measures 200-, 800-, and 4,000-step horizons. The target-horizon
measurement is used directly, while the shorter runs expose nonlinearity from
late-state sparsity. Its wall-time estimate assumes ideal worker scaling; BLAS
contention and variation between physical points can change the realized time.
