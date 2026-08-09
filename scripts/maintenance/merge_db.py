import argparse
import sqlite3
from typing import List, Tuple, Optional


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> Optional[str]:
  """Get the table schema DDL"""
  cur = conn.execute(
      "SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (
          table_name,)
  )
  row = cur.fetchone()
  return row[0] if row else None


def get_column_names(conn: sqlite3.Connection, table_name: str) -> List[str]:
  """Get table column names"""
  cur = conn.execute(f"PRAGMA table_info({table_name})")
  return [row[1] for row in cur.fetchall()]


def get_data(
    conn: sqlite3.Connection, table_name: str, columns: List[str]
) -> Tuple[List[Tuple], List[str]]:
  """Get data excluding the primary key 'id' and return column names"""
  select_cols = ', '.join(columns)
  cur = conn.execute(f"SELECT {select_cols} FROM {table_name}")
  return cur.fetchall(), columns


def merge_tables(
    db_paths: List[str], merged_db_path: str, table_name: str
) -> None:
  """Merge the same table from multiple SQLite databases into a new database"""
  # Connect to the first database to get the table schema
  conn1 = sqlite3.connect(db_paths[0])
  schema = get_table_schema(conn1, table_name)
  if not schema:
    print(f"Schema for table {table_name} not found.")
    conn1.close()
    return

  column_names = get_column_names(conn1, table_name)
  columns_without_id = [col for col in column_names if col.lower() != 'id']
  # column_names_map = {c: i for i, c in enumerate(columns_without_id)}
  placeholder = ', '.join('?' for _ in columns_without_id)
  insert_sql = f"INSERT INTO {table_name} ({', '.join(columns_without_id)}) VALUES ({placeholder})"
  
  keys = set()
  # for row in get_data(conn1, table_name, columns_without_id)[0]:
  #   key = (row[column_names_map['name']], row[column_names_map['origin']])
  #   keys.add(key)

  # Create the merged database
  merged_conn = sqlite3.connect(merged_db_path)
  merged_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
  merged_conn.execute(schema)

  # Merge data from each database
  for db_path in db_paths:
    conn = sqlite3.connect(db_path)
    
    column_names_local = get_column_names(conn, table_name)
    columns_without_id_local = [col for col in column_names_local if col.lower() != 'id']
    column_names_map_local = {c: i for i, c in enumerate(columns_without_id_local)}
    
    data_local, _ = get_data(conn, table_name, columns_without_id_local)
    data_orig = []
    for row in data_local:
      key = (row[column_names_map_local['name']], row[column_names_map_local['origin']])
      if key not in keys:
        data_orig.append(tuple(row[column_names_map_local[c]] if c in column_names_map_local else None for c in columns_without_id))
        keys.add(key)
    merged_conn.executemany(insert_sql, data_orig)
    conn.close()

  merged_conn.commit()
  merged_conn.close()
  conn1.close()
  print(f"Merging completed. Result is saved to {merged_db_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Merge the same table from multiple SQLite databases into a new database, reassigning primary key id."
  )
  parser.add_argument(
      'db_paths',
      metavar='DB',
      type=str,
      nargs='+',
      help='Paths to the SQLite databases to merge (at least 2)'
  )
  parser.add_argument(
      '--merged-db',
      required=True,
      type=str,
      help='Path to the output merged database'
  )
  parser.add_argument(
      '--table',
      required=True,
      type=str,
      help='Name of the table to merge'
  )
  args = parser.parse_args()

  if len(args.db_paths) < 2:
    print("Please specify at least two database files to merge.")
    return

  merge_tables(args.db_paths, args.merged_db, args.table)


if __name__ == '__main__':
  main()


# python merge_db.py
#   db1.sqlite db2.sqlite db3.sqlite
#   --merged-db merged.sqlite
#   --table your_table
