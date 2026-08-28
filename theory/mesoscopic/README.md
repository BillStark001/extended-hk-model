# Mesoscopic density solver

This package solves the no-repost/no-history pair closure of the extended HK
model on a uniform cell-centered grid. Opinion updating can use the full
nonlocal kernel or a sparse discrete-time Fokker--Planck closure which matches
the full kernel's first two destination moments row by row on that same grid.
The state contains node probability mass
`rho[i]` and directed edges per agent `edge[i, j]`, with invariants

```text
sum(rho) = 1
sum(edge) = mean_degree
sum_j edge[i, j] = mean_degree * rho[i]
```

The microscopic and scan configurations use a fixed integer out-degree. Every
step validates nonnegativity and all three invariants. Negative roundoff below
the tolerance is zeroed, but a material error is not renormalized or rescaled.
Parameter validation requires `dt * rewiring <= 1` and
`dt * influence <= 1`. Both opinion transitions are positive row-stochastic
matrices. The moment closure has at most four nonzeros per source row;
exogenous no-flux diffusion remains a backward-Euler M-matrix solve.

## Model scope

- Random recommendation is exact under continuum random mixing.
- Opinion recommendation is a soft Gaussian opinion kernel.
- L0-Structure is an outgoing-common-neighbor pair proxy. Exact directed
  common-neighbor top-k ranking needs wedge state, candidate masks, and score
  order statistics.
- OpinionM9 and StructureM9 use one Random and nine core slots at the standard
  `recsys_count=10`; finite-slot eligibility is retained.
- `I_p`, `I_s`, and `I_w` follow the microscopic statistics conventions.
  Distance KDEs retain four minimum-bandwidths of tail outside `[0, 2]`.
  `I_h` uses the uniform baseline `epsilon - epsilon**2 / 4`.
- First threshold crossings are linearly interpolated between recorded samples.

## Commands

Run from the repository root with the source tree on `PYTHONPATH`:

```sh
PYTHONPATH=src:. python -m theory.mesoscopic.single_run
PYTHONPATH=src:. python -m theory.mesoscopic.phase_scan
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan --jobs 4
PYTHONPATH=src:. python -m theory.mesoscopic.spectrum_check
PYTHONPATH=src:. python -m theory.mesoscopic.joint_spectrum
PYTHONPATH=src:. python -m theory.mesoscopic.spectrum_steady_states --jobs 3
PYTHONPATH=src:. python -m theory.mesoscopic.l1_comparison
```

`l1_comparison` holds every numerical choice fixed and compares the existing
pair-only `structure_random_l0` kernel with `structure_random_l1`.  L1 evolves
the four directed wedge channels `out/out`, `out/in`, `in/out`, and `in/in`
through the same conservative finite-volume transport used by `rho` and
`edge`.  Rewiring uses a documented turnover-to-independent-target moment
closure.  L0 remains a supported kernel.  L1 accepts only structural-score
power one; powers such as four require additional score moments and are not
silently approximated by the first wedge moment.

The paper's resolution check is:

```sh
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan \
  --grid-size 61 --output-dir outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid61 \
  --jobs 2 --skip-plots
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan \
  --grid-size 101 --output-dir outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid101 \
  --jobs 2 --skip-plots
PYTHONPATH=src:. python -m theory.mesoscopic.convergence_report \
  --scan-dir 61=outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid61 \
  --scan-dir 81=outputs/theory/mesoscopic/recsys_pde_noise_1e-5 \
  --scan-dir 101=outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid101 \
  --output-dir outputs/theory/mesoscopic/recsys_pde_grid_convergence
```

The default five-kernel scan uses 81 cells, `D0=1e-5`, 4,000 time steps, and
the eight rates `0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1` for both influence
and rewiring. It stores index trajectories rather than full edge-field
histories. Each analysis writes `run_metadata.json` with its command,
parameters, package versions, git state, and SHA-256 hashes of the relevant
solver and analysis sources.

Outputs live under `outputs/theory/mesoscopic/`; numerical arrays and metadata
are stored with each analysis, and plots are placed in the shared `figures/`
directory or the corresponding spectrum directory. The nonlinear-state script
uses a 1024-point dealiased periodic solver for modes `m=1,2,3`, retaining each
mode's exact symmetry sector. It does not clip or renormalize the density, and
checks its own infinitesimal growth rates against the analytical spectrum
before writing the figure. Independent modes can run in parallel with
`--jobs`. See [`RESULTS.md`](RESULTS.md) for the audited numerical summary.

`joint_spectrum.py` is the Section 3.6 extension for nonzero rewiring. It
linearizes the complete periodic pair closure in the joint
`(opinion mode, relative-edge field)` sector along the analytic rewiring base
orbit. It writes all frozen eigenvalues and fixed-horizon time-ordered singular
values for Random, Opinion, and OpinionM9. It deliberately contains no
finite-system committor or terminal-outcome logic. See
[`JOINT_SPECTRUM.md`](JOINT_SPECTRUM.md) for the derivation and scope.
