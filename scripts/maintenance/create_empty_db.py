import sys

from ehk.common.storage.sqlite import create_db_engine_and_session
from experiments.paper.analysis.records import ScenarioStatistics


if __name__ == "__main__":
  
  if len(sys.argv) != 2:
    print("Usage: python create_empty_db.py <db_path>")
    sys.exit(1)
  
  stats_db_path = sys.argv[1]
  
  stats_engine, stats_session = create_db_engine_and_session(
      stats_db_path,
      ScenarioStatistics.Base,
  )
  
  stats_session.close()
  stats_engine.dispose()
  print(f"Empty database created at {stats_db_path}")