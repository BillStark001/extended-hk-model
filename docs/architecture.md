# Repository architecture

Reusable implementation belongs in `src/ehk`. The package contains infrastructure, shared metrics, the thin SMP runtime adapter, and mathematical models; it does not select paper scenarios or output filenames.

Pure mathematical and numerical studies belong in `theory`. Runtime-backed identification and validation belong in `experiments/theory_guided`. Formal paper scans, aggregation, and scientific plots belong in `experiments/paper`.

Each task registers its own environment variable and repository-local default through `ehk.common.settings.register_path`. Resolution is lazy, so importing `ehk` never reads `.env`, opens a workspace manifest, or creates directories.

Generated files belong in `outputs/<category>/<study>/`. Task entry points create their output directories when they run. Test fixtures and external-data manifests belong in `data/fixtures` and `data/manifests`, respectively.

Dependencies flow from `ehk.common` into `ehk.metrics`, `ehk.modeling`, and `ehk.micro`, then into `theory` and `experiments`. Reusable modules must not import research task modules. Mesoscopic models must not import SMP bindings, SQLAlchemy, or Matplotlib.
