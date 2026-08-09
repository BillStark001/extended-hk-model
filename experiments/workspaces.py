"""Resolve experiment workspaces without import-time I/O."""

from __future__ import annotations

import json
from pathlib import Path

from ehk.common.settings import environment_value
from experiments.paths import WORKSPACE_MANIFEST


def instance_name(name: str | None = None, default: str | None = None) -> str:
    resolved = name or environment_value("SIMULATION_INSTANCE_NAME") or default
    if not resolved:
        raise ValueError("simulation instance name is not defined")
    return resolved


def workspace_dir(name: str | None = None, default: str | None = None) -> Path:
    selected = instance_name(name, default)
    manifest_path = WORKSPACE_MANIFEST.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        workspaces: dict[str, str] = json.load(handle)
    if selected not in workspaces:
        raise ValueError(f"workspace path for {selected!r} is not defined")
    return Path(workspaces[selected]).expanduser().resolve()
