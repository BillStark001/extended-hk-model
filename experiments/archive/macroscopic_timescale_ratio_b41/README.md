# Archived B=41 macroscopic time-scale ratio

This runner is retained for provenance and regression testing only.  Its
`B=41` results are not paper results: the pathway boundary is displaced from
the converged Figure 2 grids, and L1 steepness 4 uses a first-moment
mean-power approximation.  Use the active paper-aligned experiment instead.

This experiment evaluates opinion updating and network rewiring as separate
one-step counterfactual operators at the same mesoscopic state.  It defines

```text
Gamma_s = integral [rewiring -> I_h response]_+ dt
          / integral [opinion -> I_p response]_+ dt
```

over the pre-ordering window ending when `I_p + I_h = s`.  The default is
`s=0.25`, below either pathway-classification threshold of 0.5.  Opinion
counterfactuals use the frozen drift field without diffusion; rewiring uses
the exact conservative edge flux already computed by the solver.

The workflow scans the five recommendation configurations from
`theory/mesoscopic/recommender_scan.py` plus directional-wedge L1
StructureRandom at steepness 1 and 4.  The L1 state retains only the first
score moment.  Its explicitly named mean-power closure therefore evaluates
higher steepness as `C * E[S]^zeta`; it does not claim to recover
`C * E[S^zeta]`.  At `zeta=1` it is exactly the strict L1 rule.

The experiment compares `Gamma_0.25` with the raw `q/alpha` ratio using:

- one global PbS/SbP threshold;
- balanced accuracy and ROC AUC;
- leave-one-recommender-configuration-out threshold transfer;
- the dispersion of separately fitted thresholds across configurations;
- Spearman association with the continuous pathway index `I_w`.

It also fits a common-slope logistic pathway transition with one horizontal
offset per recommendation configuration.  `transition_offsets.csv` and
`transition_offsets.{png,pdf}` reduce each point cloud to that fitted scalar;
zero offset denotes the equal-configuration transition center.

Run the full experiment with:

```bash
PYTHONPATH=src:. python -m \
  experiments.archive.macroscopic_timescale_ratio_b41.run
```

To extend an existing output directory after adding configurations, pass
`--resume`; matching `(configuration, alpha, q)` cells are reused from its
`summary.csv`.

For a smoke run, add `--rates 0.01 0.1 --configurations random --steps 20
--grid-size 11 --jobs 1 --skip-plots`.
