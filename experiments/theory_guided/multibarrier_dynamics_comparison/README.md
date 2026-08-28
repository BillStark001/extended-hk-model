# Multibarrier dynamics and solver comparison

This experiment separates two opinion dynamics from two numerical
representations:

- `hk/nonlocal_jump`: deterministic conditional-mean push-forward;
- `deffuant/nonlocal_jump`: full random-neighbor compromise kernel;
- `hk/fokker_planck`: the same deterministic push-forward, because a
  zero-conditional-variance jump is already fixed by its first two moments;
- `deffuant/fokker_planck`: a sparse positive closure built only from each
  full jump row's destination mean and variance.

For the physical diffusion diagnostic, the one-step convention is

\[
D_{\mathrm{HK}}(x,t)=0,
\qquad D_{\mathrm{D}}(x,t)
=\frac{\Delta t\,\alpha^2}{2}
\operatorname{Var}(Y-X\mid X=x).
\]

The earlier backward-Euler/upwind implementation combined a synchronous jump
with a continuous generator step and added grid-dependent dissipation. The
current Fokker--Planck update instead matches zeroth, first, and second
*discrete destination moments* of the reference kernel to roundoff, with at
most four nonzeros per source row. This makes the comparison conservative,
positive, time-semantics matched, and fast. The nonlocal Deffuant operator
retains the third and higher conditional moments and is therefore the
reference when `alpha` is not small.

Both representations use the same cell-integrated confidence geometry. The
raw second moment and Deffuant jump destinations use cached five-point
Gauss--Legendre quadrature. This cost is paid once for each new
`(B, epsilon, dt * alpha)` tuple in a worker, rather than at every time step.

## Default protocol

The default run uses:

- `epsilon=0.2/0.4/0.8, B=161`;
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
matching. The dominant pair is tie-aware: a challenger must lead the incumbent
macro score by 5% for three consecutive records before a switch is confirmed.
Near-degenerate symmetric barriers therefore do not produce identity chatter.
A peak followed by a decline is classified as barrier annihilation,
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
