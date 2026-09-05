"""Crash-tolerant append-only storage for proposals and evaluations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import ContourProtocol, canonical_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluation_key(
    protocol_sha256: str,
    group: str,
    fidelity: int,
    x: Iterable[float],
    replicate: int,
) -> str:
    identity = {
        "protocol_sha256": protocol_sha256,
        "group": group,
        "fidelity": fidelity,
        "x": [float(item) for item in x],
        "replicate": replicate,
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evaluation:
    protocol_sha256: str
    group: str
    fidelity: int
    x: tuple[float, ...]
    replicate: int = 0
    role: str = "train"
    status: str = "ok"
    value: float | None = None
    noise_variance: float = 0.0
    censoring: str = "none"
    censor_bound: float | None = None
    runtime_seconds: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=utc_now)

    @property
    def key(self) -> str:
        return evaluation_key(
            self.protocol_sha256, self.group, self.fidelity, self.x, self.replicate
        )

    def validate(self, protocol: ContourProtocol) -> None:
        if self.protocol_sha256 != protocol.fingerprint:
            raise ValueError("evaluation protocol fingerprint mismatch")
        if self.group not in protocol.groups or self.fidelity not in protocol.fidelities:
            raise ValueError("evaluation group or fidelity is outside protocol")
        if len(self.x) != protocol.dimension:
            raise ValueError("evaluation coordinate dimension mismatch")
        for value, (lower, upper) in zip(self.x, protocol.domain, strict=True):
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError("evaluation coordinate is outside protocol domain")
        if self.replicate < 0:
            raise ValueError("replicate must be non-negative")
        if self.role not in {"initial", "train", "validation", "imported"}:
            raise ValueError("unsupported evaluation role")
        if self.status not in {"ok", "failed", "cancelled"}:
            raise ValueError("unsupported evaluation status")
        if not math.isfinite(self.noise_variance) or self.noise_variance < 0:
            raise ValueError("noise_variance must be finite and non-negative")
        if not math.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be finite and non-negative")
        if self.censoring not in {"none", "left", "right"}:
            raise ValueError("unsupported censoring")
        if self.status == "ok":
            if self.censoring == "none" and (
                self.value is None or not math.isfinite(self.value)
            ):
                raise ValueError("successful uncensored evaluation needs finite value")
            if self.censoring != "none" and (
                self.censor_bound is None or not math.isfinite(self.censor_bound)
            ):
                raise ValueError("censored evaluation needs finite censor_bound")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["x"] = list(self.x)
        record["evaluation_key"] = self.key
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Evaluation:
        fields = dict(record)
        stored_key = fields.pop("evaluation_key", None)
        fields["x"] = tuple(float(item) for item in fields["x"])
        value = cls(**fields)
        if stored_key is not None and stored_key != value.key:
            raise ValueError("stored evaluation key is invalid")
        return value


class EvaluationStore:
    """One protocol plus append-only proposal/evaluation JSONL journals."""

    def __init__(self, root: str | Path, protocol: ContourProtocol):
        self.root = Path(root)
        self.protocol = protocol
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize_protocol()

    @property
    def evaluations_path(self) -> Path:
        return self.root / "evaluations.jsonl"

    @property
    def proposals_path(self) -> Path:
        return self.root / "proposals.jsonl"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.jsonl"

    def _initialize_protocol(self) -> None:
        path = self.root / "protocol.json"
        expected = self.protocol.as_dict()
        if path.exists():
            found = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(found) != canonical_json(expected):
                raise ValueError("store already contains a different protocol")
        else:
            self._atomic_json(path, expected)
        fingerprint = self.root / "protocol.sha256"
        text = self.protocol.fingerprint + "\n"
        if fingerprint.exists() and fingerprint.read_text(encoding="utf-8") != text:
            raise ValueError("protocol.sha256 does not match protocol.json")
        if not fingerprint.exists():
            self._atomic_text(fingerprint, text)

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        EvaluationStore._atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _append(path: Path, record: Mapping[str, Any]) -> None:
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.fstat(descriptor).st_size
            prefix = b""
            if size and os.pread(descriptor, 1, size - 1) not in {b"\n", b"\r"}:
                data = os.pread(descriptor, size, 0)
                trailing = data[data.rfind(b"\n") + 1:]
                recovery = {
                    "_journal_recovery": "discard_truncated_line",
                    "discarded_sha256": hashlib.sha256(trailing).hexdigest(),
                }
                prefix = b"\n" + (canonical_json(recovery) + "\n").encode("utf-8")
            pending = memoryview(prefix + encoded)
            while pending:
                written = os.write(descriptor, pending)
                if written <= 0:
                    raise OSError("append-only journal write made no progress")
                pending = pending[written:]
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        data = path.read_bytes()
        lines = data.splitlines(keepends=True)
        result: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            if not line.endswith((b"\n", b"\r")):
                if index == len(lines) - 1:
                    break  # an interrupted final append is ignored, never rewritten
                raise ValueError(f"corrupt JSONL line {index + 1} in {path}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                if index + 1 < len(lines):
                    try:
                        recovery = json.loads(lines[index + 1])
                    except json.JSONDecodeError:
                        recovery = None
                    digest = hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()
                    if (
                        isinstance(recovery, dict)
                        and recovery.get("_journal_recovery") == "discard_truncated_line"
                        and recovery.get("discarded_sha256") == digest
                    ):
                        index += 2
                        continue
                raise ValueError(f"invalid JSONL line {index + 1} in {path}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL line {index + 1} in {path} is not an object")
            if "_journal_recovery" in value:
                raise ValueError(f"orphan journal recovery marker at line {index + 1} in {path}")
            result.append(value)
            index += 1
        return result

    def evaluations(self, *, include_validation: bool = True) -> list[Evaluation]:
        values = [Evaluation.from_record(item) for item in self._read(self.evaluations_path)]
        for value in values:
            value.validate(self.protocol)
        if not include_validation:
            values = [item for item in values if item.role != "validation"]
        return values

    def append_evaluation(self, evaluation: Evaluation) -> Evaluation:
        evaluation.validate(self.protocol)
        by_key = {item.key: item for item in self.evaluations()}
        previous = by_key.get(evaluation.key)
        if previous is not None:
            if canonical_json(previous.as_record()) != canonical_json(evaluation.as_record()):
                raise ValueError(f"evaluation key collision for {evaluation.key}")
            return previous
        self._append(self.evaluations_path, evaluation.as_record())
        return evaluation

    def append_proposal(self, record: Mapping[str, Any]) -> None:
        value = dict(record)
        value.setdefault("created_utc", utc_now())
        value["protocol_sha256"] = self.protocol.fingerprint
        self._append(self.proposals_path, value)

    def append_progress(self, event: Mapping[str, Any]) -> None:
        value = dict(event)
        value.setdefault("created_utc", utc_now())
        value["protocol_sha256"] = self.protocol.fingerprint
        self._append(self.progress_path, value)

    def pending_proposals(self) -> list[dict[str, Any]]:
        completed = {item.key for item in self.evaluations()}
        pending: dict[str, dict[str, Any]] = {}
        for proposal in self._read(self.proposals_path):
            if proposal.get("protocol_sha256") != self.protocol.fingerprint:
                raise ValueError("proposal protocol fingerprint mismatch")
            key = str(proposal["evaluation_key"])
            if key not in completed:
                pending[key] = proposal
        return list(pending.values())
