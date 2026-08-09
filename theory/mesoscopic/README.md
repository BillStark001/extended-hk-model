# Mesoscopic density solver

This package solves the no-repost/no-history closure of the extended HK model on a uniform opinion grid. It evolves:

- node opinion mass `rho[i]`;
- directed edge mass `edge[i, j]`;
- the conditional mean opinion velocity `velocity[i]`;
- one-for-one rewiring gain/loss;
- optional independent reflecting diffusion `D0`.

The stored invariants are

```
sum(rho) = 1
sum(edge) = mean_degree
sum_j edge[i, j] = mean_degree * rho[i]
```

Run the two parameter choices already used by the microscopic mechanism experiments:

``` sh
python -m theory.mesoscopic.single_run
```

Add a small background diffusion:

``` sh
python -m theory.mesoscopic.single_run \
  --noise 1e-5 \
  --tag noise_1e-5 \
  --output-dir outputs/theory/mesoscopic/noise_1e-5
```

Run the paper's complete 8-by-8 influence/rewiring grid with the Python mesoscopic solver only:

``` sh
python -m theory.mesoscopic.phase_scan
```

Compare Random, Opinion, OpinionM9, Structure, and StructureM9 on that same grid:

``` sh
python -m theory.mesoscopic.recommender_scan
```

The default sweep uses `p=0`, `k_h=0`, `D0=1e-5`, 4,000 time steps, and four worker processes. It stores only index trajectories, not 64 full edge-field histories.

The command writes compressed field trajectories and `summary.csv` under `outputs/theory/mesoscopic/`, and PDF/PNG figures under `outputs/theory/mesoscopic/figures/`. The output tree is ignored by Git.

`I_p`, `I_s`, and `I_w` follow the current full-model statistics code. `I_h` uses the corrected uniform-random baseline `epsilon - epsilon**2 / 4`.

The random recommendation kernel is exact under the continuum random-mixing closure. Opinion recommendation uses a soft opinion-ranking kernel. Structure recommendation uses an outgoing-common-neighbor pair closure; the exact directed in/out common-neighbor ranking would require an additional triplet density. The supported systems are `random`, `opinion`, `opinionm9`, `structure`, and `structurem9`. The M9 variants use finite slots: with the standard ten recommendations they contain one random and nine ranked slots; the pure variants contain ten ranked slots.
