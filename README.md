# Extended Hegselmann-Krause Model for Social Media Echo Chamber Dynamics

This repository contains the simulation code and analysis scripts for the research paper: **Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways**

\[[ArXiv Link](https://arxiv.org/abs/2601.16457)\]

## Overview

This project implements an extended discrete Bounded Confidence Model (BCM) based on the Hegselmann-Krause (HK) model to investigate how different recommendation systems influence echo chamber formation on social media platforms. The model incorporates both content-based and link-based recommendation mechanisms within a dynamic social network framework.

## Notes

There are some concepts renamed in the preprint, including:

- Gradation Index -\> Pathway Index $`I_w`$
- Environment Index -\> Subjective Polarization Index $`I_s`$

## Repository Structure

```
├── (microscopic Go runtime: https://github.com/billstark001/social-media-models)
├── (mesoscopic Go runtimes: https://github.com/billstark001/social-media-mesoscopic-models)
├── src/ehk/                # Reusable package: infrastructure, metrics, micro adapter, models
├── theory/                 # Mesoscopic and analytic theory tasks
├── experiments/
│   ├── theory_guided/      # Runtime-backed theory identification and validation
│   │   └── terminal_probability/ # Preset micro/spectrum terminal workflow
│   └── paper/              # Paper scans, analysis, plots, and illustrations
├── outputs/                # Generated data and figures, grouped by task and ignored by Git
├── data/                   # Versioned fixtures and external-data manifests
├── tests/                  # Unit, modeling, experiment, and integration tests
└── scripts/                # Maintenance scripts
    ├── merge_db.py         # Database merging
    ├── clear_db.py         # Database cleanup
    └── migrate.py          # Migrate legacy caches/events to SMP format
```

## Runtime migration summary

The local runtime/parsing stack was migrated to the shared
[`social-media-models`](https://github.com/billstark001/social-media-models)
project and its Python bindings. Mesoscopic evolution was subsequently moved to
[`social-media-mesoscopic-models`](https://github.com/billstark001/social-media-mesoscopic-models).

- Removed in-repo Go runtime (`ehk-model/`) and legacy parser package (`result_interp/`)
- Simulation entry points now call `smp_bindings.simulation.run_simulations(...)`
- Analysis modules now import records/events directly from `smp_bindings`
- Scenario metadata schema is aligned with SMP keys (see mapping below)

### Metadata Key Mapping (Breaking Changes)

If you have old metadata, use the following field mapping:

- `Decay` -\> `Influence`
- `RetweetRate` -\> `RepostRate`
- `TweetRetainCount` -\> `PostRetainCount`
- Added `DynamicsType` (default `"HK"` in this repository)

### Events Schema Naming Changes

The events DB naming in SMP uses `post`/`repost` terminology:

- Event type `Tweet` -\> `Post`
- Event table `tweet_events` -\> `post_events`
- Event table `view_tweets_events` -\> `view_posts_events`
- Flag `is_retweet` -\> `is_repost`

## Model Architecture

Refer to the [paper](https://arxiv.org/abs/2601.16457) or the [repository documentation](https://github.com/billstark001/social-media-models) for a detailed description of the model architecture and dynamics. Below is a summary of the key components and parameters.

## Installation and Setup

### Prerequisites

- **Go** compatible with the version declared by each Go repository's `go.mod`
- **Python** 3.10 or higher (for orchestration and analysis)
- **C compiler** for the microscopic runtime's SQLite dependency
- **Required Python packages**: See `requirements.txt`

### Installation Steps

1. **Clone the analysis repository and both runtimes**

``` bash
mkdir ehk-workspace
cd ehk-workspace
git clone https://github.com/billstark001/extended-hk-model.git
git clone https://github.com/billstark001/social-media-models.git
git clone https://github.com/billstark001/social-media-mesoscopic-models.git
```

Keeping the three checkouts as siblings matches this repository's default
runtime paths. Other layouts work through the environment variables below.

2. **Build and install the microscopic runtime**

``` bash
cd social-media-models
make build-all
python -m pip install -e .
```

This produces `smp` for simulations and `smp-probe` for frozen-state force
measurements.

3. **Build and install the mesoscopic runtimes**

``` bash
cd ../social-media-mesoscopic-models
python -m pip install -e .
make build
make test
make test-python
```

`make build` produces `bin/smp-kinetic` and `bin/smp-lifted`. On macOS,
`GO_TAGS=accelerate make build test` selects the optional Accelerate backend;
the default build is dependency-free pure Go. The command-line equivalents are
also available through the Python build helper:

``` bash
smp-mesoscopic-build --command kinetic --backend purego
smp-mesoscopic-build --command lifted --backend purego
```

The editable source installation is intentional: the build helper compiles the
Go source in this checkout. A `pip install` of the bindings alone does not
install prebuilt solver binaries.

4. **Install this analysis package**

``` bash
cd ../extended-hk-model
python -m pip install -r requirements.txt
python -m pip install -e .
```

5. **Configure workspace paths**

Create a `sim_ws.json` file defining workspace directories:

``` json
{
  "gradation": "/path/to/gradation/workspace",
  "epsilon": "/path/to/epsilon/workspace",
  "replicate": "/path/to/replicate/workspace",
  "mech": "/path/to/mechanism/workspace"
}
```

6. **Set environment variables**

Create a `.env` file:

``` bash
SMP_BINARY_PATH=/absolute/path/to/social-media-models/smp
SMP_KINETIC_BINARY=/absolute/path/to/social-media-mesoscopic-models/bin/smp-kinetic
SMP_LIFTED_BINARY=/absolute/path/to/social-media-mesoscopic-models/bin/smp-lifted
SIMULATION_WS_PATH=/absolute/path/to/sim_ws.json
SIMULATION_STAT_DIR=/path/to/statistics/output
SIMULATION_INSTANCE_NAME=gradation  # or epsilon, replicate, mech
STAT_THREAD_COUNT=6
EHK_THEORY_MESOSCOPIC_OUTPUT_DIR=/path/to/theory/output
EHK_SOCIAL_FORCE_INPUT_DIR=/path/to/mechanism/raw
EHK_SOCIAL_FORCE_OUTPUT_DIR=/path/to/probe/output
EHK_TERMINAL_PROBABILITY_OUTPUT_DIR=/path/to/terminal/output
```

Every registered input or output path has the repository-local default shown in `theory/paths.py`, `experiments/paths.py`, or `experiments/paper/paths.py`; a matching environment variable can override it through `.env`.

## Migrating Existing Cache Data

If your `snapshot-*.msgpack` / `events.db` were generated by the legacy runtime, run the migration script before analysis:

``` bash
python scripts/migration/migrate.py /path/to/run_dir --dynamics HK
```

Useful options:

- `--dry-run`: report what would be migrated without writing changes
- `--dynamics`: set `DynamicsType` when wrapping legacy snapshots (default `HK`)

The script scans each direct child directory under the target path and migrates:

- `snapshot-*.msgpack`
- `events.db`

## Running Simulations

### Quick Start Example

``` bash
# Run a single simulation with specific parameters (HK via SMP runtime)
$SMP_BINARY_PATH /path/to/output '{"UniqueName":"test","DynamicsType":"HK","Tolerance":0.45,"Influence":0.05,"RewiringRate":0.05,"RepostRate":0.3,"RecsysFactoryType":"OpinionM9","RecsysCount":10,"PostRetainCount":3,"MaxSimulationStep":15000}'
```

### Batch Simulations

Use the paper experiment runner for all pre-configured scenarios.

Basic usage:

``` bash
python -m experiments.paper.run <scenario> [--workspace <name>] [--concurrency <n>]
```

Available scenarios:

- `gradation`: Parameter sweep over influence, rewiring, repost rates, and recsys types
- `epsilon`: Tolerance threshold analysis
- `replicate`: Replication-based statistical validation
- `mech`: Mechanism pathway analysis

The theory-paper terminal-probability comparison has its own preset runner,
because it uses weighted-random recommenders and common random numbers across
five matched columns:

``` bash
python -m experiments.theory_guided.terminal_probability.run --list-presets
python -m experiments.theory_guided.terminal_probability.run paper-figure3 --dry-run
```

See
[`experiments/theory_guided/terminal_probability/README.md`](experiments/theory_guided/terminal_probability/README.md)
for production, analysis, and spectrum commands.

Examples:

``` bash
# Gradation study
python -m experiments.paper.run gradation --concurrency 6

# Epsilon study
python -m experiments.paper.run epsilon --concurrency 6

# Replication study
python -m experiments.paper.run replicate --concurrency 8

# Mechanism study
python -m experiments.paper.run mech --concurrency 4
```

Arguments:

- `scenario` (required): one of `gradation`, `epsilon`, `replicate`, `mech`
- `--workspace` (optional): workspace key resolved from `sim_ws.json`
- `--concurrency` (optional, default: `4`): max number of concurrent simulations

### Simulation Output

Each simulation produces:

- **Graph snapshots** (`graph-{step}.msgpack`): Network structure at key steps
- **Accumulative state** (`acc-state-{timestamp}.lz4`): Compressed time-series data
- **Event database** (`events.db`): SQLite database with detailed agent events
- **Final state** (`finished-{timestamp}.msgpack`): Complete end state

## Data Analysis

### Mesoscopic analysis orchestration

This repository no longer contains a mesoscopic time-evolution or stochastic
terminal solver. Every current trajectory and terminal-probability experiment
calls the Go `smp-kinetic` or `smp-lifted` runtime from
`social-media-mesoscopic-models`; the small module under
`src/ehk/modeling/mesoscopic/` only translates requests and decoded responses.
The Go kinetic runtime exposes explicit `measure` and `fokker_planck` paths,
returns online observables, and can return only the requested `rho`, `edge`,
velocity, and rewiring-flux snapshots. A request for `final_rho` or
`final_edge` captures only the terminal array without retaining a trajectory;
this is the preferred contract for terminal-density comparisons. Install
`smp_meso_bindings`, build the binaries there, and set `SMP_KINETIC_BINARY`
when the sibling checkout is not at its standard location. Requests can also
be sent directly to the Go JSONL interfaces; their complete, explicit schemas
are documented in the
[`social-media-mesoscopic-models` README](https://github.com/billstark001/social-media-mesoscopic-models#lifted-request).

For a saved explicit request, the Go commands support a single JSON request or
a recoverable JSONL batch:

``` bash
/path/to/smp-kinetic run @kinetic-request.json
/path/to/smp-lifted batch < lifted-requests.jsonl > lifted-responses.jsonl
```

Python scans should reuse a long-lived batch process through
`smp_meso_bindings.run_kinetic_batch` or `run_lifted_batch_parallel`; they
should not launch one solver process per parameter point.

``` bash
SMP_KINETIC_BINARY=/path/to/smp-kinetic python -m theory.mesoscopic.single_run
python -m theory.mesoscopic.phase_scan --kinetic-binary /path/to/smp-kinetic
python -m theory.mesoscopic.recommender_scan \
  --kinetic-binary /path/to/smp-kinetic --jobs 4
python -m theory.mesoscopic.spectrum_check
python -m theory.mesoscopic.joint_spectrum
python -m theory.mesoscopic.convergence_report
python -m experiments.theory_guided.macroscopic_timescale_ratio.run \
  --grid-size 161 --dynamics hk --opinion-method measure \
  --epsilon 0.45 --jobs 8 --dry-run
```

The time-scale scan defaults to the paper's current kinetic protocol
(`B=161`, `measure`, `epsilon=0.45`, zero background noise). Its CLI keeps
`hk`/`deffuant`, `measure`/`fokker_planck`, and confidence radius explicit.
Transition offsets fit the same continuous pathway index shown in the
heatmaps; the older first-passage binary fit is retained only in archived
experiments.

The local spectrum and convergence modules are reduced-operator diagnostics
and cache/report tooling; they do not advance the current mesoscopic state and
are not substitutes for either Go runtime.

See `theory/mesoscopic/README.md` for assumptions and reproduction commands,
and `theory/mesoscopic/RESULTS.md` for the corrected multi-resolution results.

### Adaptive contour and boundary estimation

`ehk.contours` implements the shared sequential estimator for continuous
rate-space scans. Its coordinates are normally
`(log10_alpha, log10_q) in [-3,0]^2`; every complete protocol JSON also fixes
the groups, ordered grid fidelities, response transform, solver settings,
target levels, level weights, validation design, and acquisition settings.
The SHA-256 of canonical protocol JSON is part of every evaluation key. A
cache point is reused only when that fingerprint, group, fidelity, coordinates,
and replicate agree exactly.

Initialize and inspect an experiment without running a solver:

```bash
ehk-contour init protocol.json /path/to/contour-run
ehk-contour status /path/to/contour-run
```

The store contains immutable `protocol.json`/`protocol.sha256` plus append-only
`proposals.jsonl`, `evaluations.jsonl`, and `progress.jsonl`. Each proposal is
durably appended before the solver starts and each result is flushed and
`fsync`ed, so an interrupted invocation resumes its pending action. Evaluation
keys are idempotent and conflicting duplicate results are rejected. Scientific
failures remain explicit records rather than missing values.

For the deterministic measure--Fokker--Planck pathway response, run:

```bash
ehk-contour run /path/to/contour-run \
  --adapter operator-gap --binary /path/to/smp-kinetic \
  --max-evaluations 20 --progress jsonl
ehk-contour export-grid /path/to/contour-run --resolution 501
```

The built-in operator adapter evaluates both methods as a two-item Go batch and
returns `log10(abs(I_w_measure-I_w_fokker_planck))`. The solver accumulates
`I_w` online at every numerical step, adds its declared completion term, and
returns only the terminal scalar. Thus `I_w` is independent of the recording
cadence used for other scalar series; `steps`, the time step, and the pathway
observable settings remain estimand fields. Values at or below the protocol's
independently declared numerical floor are left-censored. Do not replace a zero
with an arbitrary small constant.

For a non-`I_w` landscape response, set
`response.observable=peak_dominant_barrier_height`, register the snapshot
schedule and all multiwell thresholds under `response.landscape`, then use
`--adapter landscape-barrier-gap`. This adapter reconstructs
`potential_from_force(velocity / alpha)` and returns the logged absolute gap
between the two methods' peak tracked dominant-barrier heights. Its Go request
selects only `rho` and `velocity`; it does not transfer edge or rewiring-flux
histories. `simulator.processes_per_evaluation=2` can evaluate the paired
methods concurrently. The record schedule, tracking hysteresis, basin-mass and
barrier-height cutoffs are part of the estimand and protocol fingerprint.

`PairedMultimetricEvaluator` is the shared-solve adapter for experiments that
compare several validity fields at once. At one rate-space coordinate it runs
HK/Deffuant times measure/Fokker--Planck, requests trajectory snapshots only
for the observables that need them, and reuses final-only density snapshots for
terminal node/edge continuous-L2 gaps. The same result bundle can also contain
tracked-barrier and pathway gaps, so eight scalar records share four solver
trajectories. Each scalar evaluation remains separately keyed and queryable,
while a per-point atomic checkpoint permits reconstruction after interruption.
For terminal density L2, the adapter now requires both trajectories' final
recorded times to equal the protocol's `steps*dt` and each other before it uses
the final-only snapshots. For barrier and pathway responses it retains and
fits the signed measure-minus-Fokker--Planck difference; `log10(abs(delta))` is
applied only to the reported field. This keeps harmless sign crossings from
becoming singular training targets.

The shared-field production design can combine an anisotropic regular safety
net with balanced batch acquisition. Candidate locations are shortlisted by
pointwise uncertainty, then scored by standardized integrated variance
reduction. An equal-weight mean, the currently worst integrated-risk field,
and a rotating per-field quota determine the score; local penalization keeps
the two points in a batch separated. Since every point produces all eight
outputs, quotas influence coordinates without discarding any shared result.
Global held-out points remain unbiased calibration checks, while separately
reported high-rate and signed-zero challenge strata stress the known difficult
regions.

The surrogate is an exact ARD Matérn-5/2 GP over the two coordinates and
inverse grid fidelity. It accepts known observation variance and iteratively
propagates left/right censoring. An optional training-only leave-one-out
quantile scale can inflate—but never shrink—predictive variance when the
stationary mean/kernel understates local interpolation error. Acquisition approximates contour-weighted
multi-level tIMSE, retains a global-variance term to avoid missing islands, and
divides expected reduction by measured or declared fidelity cost. Its action
space contains new points, promotion of an existing point to the next grid,
and replication for noisy responses. The multinomial terminal adapter computes
five-category TV and its sampling variance directly from generator/microscopic
counts; failed runs are excluded from those counts, whereas a successful
nonabsorbed run is the fifth `censored` outcome.

Calibration must be reported in the representation actually fitted. For a
signed response, good interval coverage for the signed difference does not
imply that a single interval after `log10(abs(.))` will cover well near zero;
the transformed posterior can be strongly non-Gaussian. A short fitted length
scale or failed region-specific challenge coverage is a trigger for local or
nonstationary refinement, and a narrow displayed zero trough is not by itself
evidence for an equally narrow physical regime.

Initial low-fidelity Sobol points and nested promotions precede acquisition.
Held-out Sobol validation points are tagged `role=validation` and are never
used to fit or stop the GP. The summary reports integrated sign-error risk,
credible-band fractions, validation RMSE/coverage, failures, and the stopping
reason. `posterior_grid.npz` only increases rendering resolution; it does not
create observations. A production conclusion still requires held-out coverage,
high-fidelity contour stability, and an acceptable `B=121` to `B=161` shift.

Grid fidelity is never averaged as if it were a replicate. Group aggregation
must be declared before the final scan (for example small multiples, pointwise
quantiles, or worst case); averaging already extracted contour coordinates and
choosing representative groups after seeing the result are unsupported.

### Event Database Schema

The event database tracks three event types:

1. **Rewiring Events**: Network structure changes
2. **Post events**: Posts and reposts with opinion values
3. **View-post events**: Detailed content exposure records

### Analysis Scripts

``` python
# Load and analyze simulation results
from smp_bindings import load_events_db, get_events_by_step_range

db = load_events_db("path/to/events.db")
events = get_events_by_step_range(db, 0, 1000, type_="Rewiring")

# Calculate opinion polarization metrics
from ehk.metrics.polarization import DistanceCalculator
polarization = DistanceCalculator(sample_count=len(opinions)).calculate(graph, opinions)
```

## Experimental Configurations

The `experiments/paper/scenarios.py` module defines the paper scenarios, while reusable metadata construction lives in `ehk.micro.scenarios`:

- **all_scenarios_grad**: 12 simulations × 8 rewiring rates × 8 influence rates × 4 repost rates × 4 recommendation systems
- **all_scenarios_eps**: 100 simulations × 16 tolerance values
- **all_scenarios_rep**: 100 replications × 2 recommendation systems
- **all_scenarios_mech**: 9 mechanism-focused configurations

## Extending the Model

### Adding Custom Recommendation Systems

Recommendation implementations now live in the external microscopic runtime.
Implement the generic `model.SMPModelRecommendationSystem[O, P]` interface
under `social-media-models/recsys`, then register its factory in
`social-media-models/simulation/scenario-metadata.go`. See that repository's
current interfaces and built-in recommenders instead of copying the historical
HK-specific API.

### Modifying Agent Behavior

Edit the HK dynamics implementation in the `social-media-models` repository if you need to modify core agent behavior.

## Citation

If you use this code in your research, please cite:

``` bibtex
@misc{zhao2026segregation,
    title={Segregation Before Polarization: How Recommendation Strategies Shape Echo Chamber Pathways}, 
    author={Junning Zhao and Kazutoshi Sasahara and Yu Chen},
    year={2026},
    eprint={2601.16457},
    archivePrefix={arXiv},
    primaryClass={cs.SI},
    url={https://arxiv.org/abs/2601.16457}, 
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

This research investigates the critical role of algorithmic recommendations in fostering polarization on social media platforms. The findings provide theoretical frameworks for stage-dependent interventions, suggesting platforms could dynamically adjust algorithms to mitigate polarization without resorting to censorship.
