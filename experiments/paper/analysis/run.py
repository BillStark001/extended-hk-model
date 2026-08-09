import argparse
import os
import sys
from experiments.paper.analysis.tasks import get_mode_names, run_mode
DEFAULT_STAT_CONCURRENCY = 6


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m experiments.paper.analysis.run",
      description="Generate scenario statistics and upsert them into stats DB.",
  )
  parser.add_argument(
      "mode",
      choices=get_mode_names(),
      help="Statistic mode to run.",
  )
  parser.add_argument(
      "instance_name",
      help="Simulation workspace instance name defined in sim_ws.json.",
  )
  parser.add_argument(
      "-c",
      "--concurrency",
      type=int,
      default=None,
      help=(
          "Worker process count. Resolve order: CLI arg > STAT_THREAD_COUNT env > "
          f"default ({DEFAULT_STAT_CONCURRENCY})."
      ),
  )
  return parser


def _resolve_concurrency(cli_value: int | None) -> int:
  if cli_value is not None:
    return max(cli_value, 1)

  env_value_raw = os.environ.get("STAT_THREAD_COUNT")
  if env_value_raw is not None and env_value_raw.strip() != "":
    try:
      return max(int(env_value_raw.strip()), 1)
    except ValueError:
      return DEFAULT_STAT_CONCURRENCY

  return DEFAULT_STAT_CONCURRENCY


def main(argv: list[str] | None = None):
  if argv is None:
    argv = sys.argv[1:]
  parser = _build_parser()
  args = parser.parse_args(argv)

  concurrency = _resolve_concurrency(args.concurrency)
  run_mode(args.mode, args.instance_name, concurrency)


if __name__ == "__main__":
  main()
