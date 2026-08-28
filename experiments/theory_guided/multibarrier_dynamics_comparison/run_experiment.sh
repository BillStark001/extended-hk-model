#!/bin/sh
set -eu

SCRIPT_DIRECTORY=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIRECTORY/../../.." && pwd)
OUTPUT_DIRECTORY=${1:-/Users/billstark001/Desktop/research/extended-hk-model-paper/artifacts/multibarrier_dynamics_comparison}
JOBS=${JOBS:-8}
BENCHMARK_POINTS=${BENCHMARK_POINTS:-6}

cd "$REPOSITORY_ROOT"

/usr/bin/time -lp env \
  PYTHONPATH=src:. \
  PYTHONUNBUFFERED=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  python -m experiments.theory_guided.multibarrier_dynamics_comparison.run \
  --epsilon-grid 0.2:161 \
  --epsilon-grid 0.4:161 \
  --epsilon-grid 0.8:161 \
  --grid-selection anti_diagonal_band \
  --band-offsets -1 0 1 \
  --steps 4000 \
  --early-until 200 \
  --early-every 1 \
  --record-every 20 \
  --dominant-score-margin 0.05 \
  --dominant-switch-persistence 3 \
  --jobs "$JOBS" \
  --output-dir "$OUTPUT_DIRECTORY"

env PYTHONPATH=src:. python -m \
  experiments.theory_guided.multibarrier_dynamics_comparison.select_comparison_points \
  "$OUTPUT_DIRECTORY"

env \
  PYTHONPATH=src:. \
  PYTHONUNBUFFERED=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  python -m experiments.theory_guided.multibarrier_dynamics_comparison.benchmark_methods \
  "$OUTPUT_DIRECTORY" \
  --horizons 200 800 4000 \
  --repeats 1 \
  --max-points "$BENCHMARK_POINTS" \
  --jobs-for-estimate "$JOBS"
