from __future__ import annotations

import argparse
import pathlib
import sqlite3
from dataclasses import dataclass
from typing import Callable

import msgpack


@dataclass
class MigrateStats:
  migrated: int = 0
  skipped: int = 0
  failed: int = 0


def _is_snapshot_migrated(path: pathlib.Path) -> bool:
  data = path.read_bytes()
  inner = msgpack.unpackb(data, raw=False, strict_map_key=False)
  return isinstance(inner, dict) and "DynamicsType" in inner and "Data" in inner


def _load_smp_migrators() -> tuple[Callable[[str, str], None], Callable[[str], None]]:
  try:
    from smp_bindings.migrate import migrate_events_db as smp_migrate_events_db
    from smp_bindings.migrate import migrate_snapshot as smp_migrate_snapshot
  except Exception as exc:  # pragma: no cover - import error path
    raise RuntimeError(
        "failed to import smp_bindings migrators; ensure smp_bindings is installed "
        "or available in PYTHONPATH"
    ) from exc
  return smp_migrate_snapshot, smp_migrate_events_db


def migrate_snapshot(path: pathlib.Path, dynamics_type: str, dry_run: bool) -> str:
  if _is_snapshot_migrated(path):
    return "skip"
  if dry_run:
    return "migrated"
  smp_migrate_snapshot, _ = _load_smp_migrators()
  smp_migrate_snapshot(str(path), dynamics_type)
  return "migrated"


def _table_names(db: sqlite3.Connection) -> set[str]:
  cur = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
  return {row[0] for row in cur.fetchall()}


def _column_names(db: sqlite3.Connection, table: str) -> set[str]:
  cur = db.execute(f"PRAGMA table_info({table})")
  return {row[1] for row in cur.fetchall()}


def migrate_events_db(path: pathlib.Path, dry_run: bool) -> str:
  db = sqlite3.connect(path)
  try:
    tables = _table_names(db)
    if dry_run:
      return "migrated" if _would_migrate_events_db(db, tables) else "skip"
    if not _would_migrate_events_db(db, tables):
      return "skip"
  finally:
    db.close()

  _, smp_migrate_events_db = _load_smp_migrators()
  smp_migrate_events_db(str(path))
  return "migrated"


def _would_migrate_events_db(db: sqlite3.Connection, tables: set[str]) -> bool:
  if "tweet_events" in tables and "post_events" not in tables:
    return True
  if "post_events" in tables:
    columns = _column_names(db, "post_events")
    if "is_retweet" in columns and "is_repost" not in columns:
      return True
  if "rewiring_events" in tables:
    if "agent_id" not in _column_names(db, "rewiring_events"):
      return True
  if "view_tweets_events" in tables and "view_posts_events" not in tables:
    return True
  if "events" in tables:
    cur = db.execute(
        "SELECT 1 FROM events WHERE type IN ('Tweet', 'ViewTweets') LIMIT 1"
    )
    if cur.fetchone() is not None:
      return True
  return False


def migrate_record_dir(
    record_dir: pathlib.Path,
    dynamics_type: str,
    dry_run: bool,
) -> MigrateStats:
  stats = MigrateStats()

  snapshots = sorted(record_dir.glob("snapshot-*.msgpack"))
  events_db = record_dir / "events.db"

  for snapshot_path in snapshots:
    try:
      result = migrate_snapshot(snapshot_path, dynamics_type=dynamics_type, dry_run=dry_run)
      if result == "migrated":
        stats.migrated += 1
        print(f"[migrated] snapshot: {snapshot_path}")
      else:
        stats.skipped += 1
        print(f"[skip] snapshot already migrated: {snapshot_path}")
    except Exception as exc:
      stats.failed += 1
      print(f"[error] snapshot {snapshot_path}: {exc}")

  if events_db.exists():
    try:
      result = migrate_events_db(events_db, dry_run=dry_run)
      if result == "migrated":
        stats.migrated += 1
        print(f"[migrated] events db: {events_db}")
      else:
        stats.skipped += 1
        print(f"[skip] events db already migrated: {events_db}")
    except Exception as exc:
      stats.failed += 1
      print(f"[error] events db {events_db}: {exc}")

  return stats


def iter_record_dirs(root_dir: pathlib.Path):
  for p in sorted(root_dir.iterdir()):
    if p.is_dir():
      yield p


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Migrate simulation caches in each child directory under a root directory. "
          "Targets snapshot-*.msgpack and events.db."
      )
  )
  parser.add_argument(
      "target_dir",
      type=pathlib.Path,
      help="Directory whose direct child folders contain simulation caches",
  )
  parser.add_argument(
      "--dynamics",
      default="HK",
      help="DynamicsType used when wrapping snapshots (default: HK)",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Only report files that would be migrated, without writing changes",
  )
  args = parser.parse_args()

  root_dir: pathlib.Path = args.target_dir
  if not root_dir.exists() or not root_dir.is_dir():
    print(f"[error] target_dir is not a valid directory: {root_dir}")
    raise SystemExit(1)

  total = MigrateStats()
  for record_dir in iter_record_dirs(root_dir):
    stats = migrate_record_dir(
        record_dir,
        dynamics_type=args.dynamics,
        dry_run=args.dry_run,
    )
    total.migrated += stats.migrated
    total.skipped += stats.skipped
    total.failed += stats.failed

  print(
      f"[summary] migrated={total.migrated} skipped={total.skipped} failed={total.failed}"
  )
  if total.failed > 0:
    raise SystemExit(2)


if __name__ == "__main__":
  main()