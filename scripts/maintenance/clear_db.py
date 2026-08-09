from typing import List
import sys

from ehk.common.storage.sqlite import create_db_session
from experiments.paper.analysis.records import ScenarioStatistics

if __name__ == "__main__":
  stats_db_path = sys.argv[1]
  flawed_path = sys.argv[2]

  # Open the statistics database.
  session = create_db_session(stats_db_path)

  # Load the names marked as flawed.
  with open(flawed_path, 'r', encoding="utf-8") as f:
    flawed_name_list = [x.strip() for x in f.readlines()]

  for flawed_name in flawed_name_list:
    print(flawed_name)
    # LIKE provides substring matching; doubled percent signs escape formatting.
    ret: List[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
        ScenarioStatistics.name.like(f"%{flawed_name}%")
    ).all()
    for s in ret:
      session.delete(s)
      print('Deleted:', s.name)
    session.commit()

  session.close()
