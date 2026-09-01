# Go base-layer terminal probabilities

This workflow sends explicit `base`-layer requests to the external
`smp-lifted` runtime. The historical Python pair/score-moment factorial has
been removed: this repository constructs requests, supervises long-lived Go
batch processes, and records responses, but performs no stochastic state
evolution itself.

The directory name is retained only to keep the module discoverable. Its old
four-cell output schema and the `historical-12` preset are not supported.

Install and build `social-media-mesoscopic-models`, then run from this
repository root:

```sh
export SMP_LIFTED_BINARY=/absolute/path/to/social-media-mesoscopic-models/bin/smp-lifted

PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run smoke --dry-run

PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run smoke \
  --jobs 1 --workers-per-request 2

PYTHONPATH=src:. python -m \
  experiments.theory_guided.terminal_generator_factorial.run paper-figure3 \
  --jobs 4 --workers-per-request 2
```

Budget CPU as approximately `jobs × workers-per-request`. Each run saves the
complete JSONL requests, complete Go responses, a flat probability table, and
source/binary provenance. The point-probability category order is
`k1,k2,k3,k4plus,censored`. The paper preset uses 80 Go base paths for each of
its 20 conditions; the independently generated microscopic reference has 40
runs per condition.
