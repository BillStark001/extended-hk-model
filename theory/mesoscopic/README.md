# Mesoscopic analyses driven by the Go runtime

This package contains analysis and plotting code only. All kinetic evolution,
including runs that request `rho`, `edge`, velocity, or rewiring-flux
snapshots, is performed by the external Go runtime. `phase_scan` and
`recommender_scan` normally request online scalar series through the
binary-array protocol. The kinetic state contains
node probability mass
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
- Opinion recommendation is the microscopic triangular similarity score raised
  to the configured steepness.
- L0-Structure is an outgoing-common-neighbor pair proxy. Exact directed
  common-neighbor top-k ranking needs wedge state, candidate masks, and score
  order statistics.
- `recommendation_random_ratio=0.1` mixes one-tenth Random mass into the ranked
  kernel; finite-list eligibility is retained explicitly.
- `I_p`, `I_s`, and `I_w` follow the microscopic statistics conventions.
  Distance KDEs retain four minimum-bandwidths of tail outside `[0, 2]`.
  `I_h` uses the uniform baseline `epsilon - epsilon**2 / 4`.
- First threshold crossings are linearly interpolated between recorded samples.

## Commands

Run from the repository root with the source tree on `PYTHONPATH`:

```sh
KINETIC_BINARY=/absolute/path/to/smp-kinetic
SMP_KINETIC_BINARY="$KINETIC_BINARY" PYTHONPATH=src:. python -m theory.mesoscopic.single_run
PYTHONPATH=src:. python -m theory.mesoscopic.phase_scan \
  --kinetic-binary "$KINETIC_BINARY"
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan \
  --kinetic-binary "$KINETIC_BINARY" --jobs 4
PYTHONPATH=src:. python -m theory.mesoscopic.spectrum_check
PYTHONPATH=src:. python -m theory.mesoscopic.joint_spectrum
PYTHONPATH=src:. python -m theory.mesoscopic.spectrum_steady_states --jobs 3
SMP_KINETIC_BINARY="$KINETIC_BINARY" PYTHONPATH=src:. python -m theory.mesoscopic.l1_comparison
```

`l1_comparison` holds every numerical choice fixed and compares the Go
pair-only `structure_random_l0` kernel with Go `structure_random_l1`. The latter
uses the retained directional-wedge score and an explicit capped-Poisson
score-moment closure, so its steepness remains configurable.

The paper's resolution check is:

```sh
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan \
  --kinetic-binary "$KINETIC_BINARY" \
  --grid-size 61 --output-dir outputs/theory/mesoscopic/recsys_pde_noise_1e-5_grid61 \
  --jobs 2 --skip-plots
PYTHONPATH=src:. python -m theory.mesoscopic.recommender_scan \
  --kinetic-binary "$KINETIC_BINARY" \
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
parameters, package versions, git state, the external binary identity, and
SHA-256 hashes of the request adapter and analysis sources.

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
