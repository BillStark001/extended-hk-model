from typing import Callable, Dict, List, TypeAlias

import multiprocessing.util
import traceback
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor, as_completed

from sqlalchemy.orm import Session, undefer
from tqdm import tqdm

from ehk.common.storage.sqlite import create_db_engine_and_session, sync_sqlite_table
from ehk.micro.scenarios import ScenarioMetadata
from experiments.paper.analysis.records import ScenarioStatistics


short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


class AnyAttr:
  pass


def _stop_executor_workers(executor: ProcessPoolExecutor):
  processes = getattr(executor, '_processes', None)  # 3.12.3
  if not processes:
    return

  worker_processes = [proc for proc in processes.values() if proc is not None]

  for proc in worker_processes:
    try:
      if proc.is_alive():
        proc.terminate()
    except Exception:
      pass

  for proc in worker_processes:
    try:
      proc.join(timeout=0.2)
    except Exception:
      pass

  for proc in worker_processes:
    try:
      if proc.is_alive():
        proc.kill()
    except Exception:
      pass

  for proc in worker_processes:
    try:
      proc.join(timeout=0.2)
    except Exception:
      pass


def try_get_stats(sess: Session, name: str, origin: str) -> ScenarioStatistics | None:
  ret = sess.query(ScenarioStatistics).filter(
      ScenarioStatistics.name == name,
      ScenarioStatistics.origin == origin,
  ).options(undefer('*')).first()
  if not ret:
    return None
  ret2 = AnyAttr()
  for column in ScenarioStatistics.__table__.columns:
    attr = getattr(ret, column.name)
    setattr(ret2, column.name, attr)
  return ret2  # type: ignore


StatisticsGetterFunc: TypeAlias = Callable[
    [ScenarioMetadata, str, str, ScenarioStatistics | None],
    ScenarioStatistics | None,
]


def generate_stats(
    get_statistics: StatisticsGetterFunc,
    scenario_base_path: str,
    stats_db_path: str,
    origin: str,
    scenarios: List[ScenarioMetadata],
    ignore_exist: bool = True,
    concurrency: int = 1,
):

  logger = multiprocessing.util.log_to_stderr()
  logger.setLevel('INFO')

  stats_engine, stats_session = create_db_engine_and_session(
      stats_db_path,
      ScenarioStatistics.Base,
  )

  sync_sqlite_table(
      stats_engine,
      ScenarioStatistics,
      extra_columns='error',
  )

  executor = ProcessPoolExecutor(
      max_workers=max(concurrency, 1),
      # max_tasks_per_child=32,
  )
  _interrupted = False
  futures: Dict[Future[ScenarioStatistics | None], str] = {}

  try:
    for s in tqdm(scenarios):
      exist_stats = try_get_stats(stats_session, s['UniqueName'], origin)
      if ignore_exist and exist_stats is not None:
        continue
      # else, upsert
      f = executor.submit(
          get_statistics, s, scenario_base_path, origin, exist_stats)
      futures[f] = s['UniqueName']

    n = len(futures)
    print(f'{n} Futures submitted.')

    for f in tqdm(
        as_completed(futures), total=n, bar_format=short_progress_bar,
    ):
      f_name = futures[f]
      try:
        res = f.result()
        if res is not None:
          stats_session.merge(res)  # this upserts the record
          stats_session.commit()
      except CancelledError:
        pass  # future was cancelled on interrupt, skip silently
      except Exception as e:
        stats_session.rollback()
        print(f"Exception in worker: {e}")
        print(f"Scenario: {f_name}")
        print("Traceback:")
        print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))

  except KeyboardInterrupt:
    _interrupted = True
    print("\nKeyboardInterrupt: cancelling all pending tasks and shutting down...")
    for f in futures:
      f.cancel()
    _stop_executor_workers(executor)
    executor.shutdown(wait=False, cancel_futures=True)
    raise

  finally:
    if not _interrupted:
      executor.shutdown(wait=True)
    try:
      stats_session.close()
    except Exception:
      pass