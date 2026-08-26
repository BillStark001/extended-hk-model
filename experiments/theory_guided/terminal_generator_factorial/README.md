# Terminal-generator factorial

This workflow makes two choices orthogonal instead of hiding them inside
different historical scripts:

| | unsplit | fast-slow |
| --- | --- | --- |
| pair state | `(rho, E)` | `(rho, E)` with conditional fast rewiring |
| score moments | `(rho, E, W, S2, S4)` | the same lifted state with conditional fast rewiring |

All four cells call `ehk.modeling.terminal_generator` and therefore use the
same initialization, synchronous opinion kernel, tau-leap rewiring law,
terminal classification, and random-number conventions.  In a `fast_slow`
cell, conditional fast absorption is applied only when `q / alpha` reaches the
configured threshold; the row records whether it was actually applied.

From the repository root:

```sh
PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run smoke

PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run \
  historical-12 --jobs 8

PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run \
  paper-figure3 --jobs 8
```

`historical-12` describes the old 4-case x 3-recommender x 40-path protocol,
but now expands it to all four factorial cells.  It is intentionally not run
as part of tests.  The smoke preset exercises all cells with two short paths.
`paper-figure3` uses the exact four rates and five recommender columns of the
matched main-paper experiment; its full factorial expansion contains 3,200
paths and is likewise configuration-only in this change.

Archived paper artifacts made with the old pair Langevin generator and the
later lifted synchronous generator are useful provenance, but they are not a
strict state-level comparison because their event laws differ.  Use this
workflow for new controlled comparisons.
