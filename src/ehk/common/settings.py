"""Lazy environment-backed path settings for research tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PathSetting:
    """An environment variable paired with a repository-relative default."""

    environment: str
    default: Path

    def resolve(self) -> Path:
        load_dotenv()
        raw_value = os.environ.get(self.environment)
        value = Path(raw_value).expanduser() if raw_value else self.default
        if not value.is_absolute():
            value = REPOSITORY_ROOT / value
        return value.resolve()


def register_path(environment: str, default: str | Path) -> PathSetting:
    """Register a path without reading the environment at import time."""

    default_path = Path(default)
    if not default_path.is_absolute():
        default_path = REPOSITORY_ROOT / default_path
    return PathSetting(environment=environment, default=default_path)


def environment_value(name: str, default: str | None = None) -> str | None:
    """Read a dotenv-aware non-path setting on demand."""

    load_dotenv()
    return os.environ.get(name, default)
