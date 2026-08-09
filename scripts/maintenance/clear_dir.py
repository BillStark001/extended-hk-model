import argparse
import os
from typing import List, Optional


def get_files_with_prefix(files: List[str], prefix: str) -> List[str]:
  """Return all files in the list that start with the given prefix."""
  return [f for f in files if f.startswith(prefix)]


def get_latest_file(files: List[str], dir_path: str) -> Optional[str]:
  """
  Given a list of filenames, return the filename of the latest file
  according to modified time.
  """
  if not files:
    return None
  files_with_mtime = [(f, os.path.getmtime(
      os.path.join(dir_path, f))) for f in files]
  files_with_mtime.sort(key=lambda x: x[1], reverse=True)
  return files_with_mtime[0][0]


def remove_files(files: List[str], dir_path: str, keep_file: Optional[str]) -> None:
  """Delete all files in the list except for keep_file."""
  for f in files:
    if f != keep_file:
      try:
        os.remove(os.path.join(dir_path, f))
        print(f"Deleted file: {os.path.join(dir_path, f)}")
      except Exception as e:
        print(f"Failed to delete {os.path.join(dir_path, f)}: {e}")


def main():
  parser = argparse.ArgumentParser(
      description="Clean up files in subfolders according to rules.")
  parser.add_argument("target_dir", type=str, help="Target directory to scan")
  parser.add_argument("int_arg", type=int, nargs="?",
                      default=None, help="Optional integer argument")
  args = parser.parse_args()

  target_dir: str = args.target_dir
  int_arg: Optional[int] = args.int_arg

  if not os.path.isdir(target_dir):
    print(f"{target_dir} is not a valid directory.")
    return

  # Walk through subfolders only (not recursively)
  for root, dirs, files in os.walk(target_dir):
    if root == target_dir:
      # Only process subdirectories, not the top-level directory itself
      for subdir in dirs:
        dir_path = os.path.join(root, subdir)
        process_subfolder(dir_path, int_arg)
      break  # Don't go deeper


def process_subfolder(dir_path: str, int_arg: Optional[int]) -> None:
  """Process a single subfolder according to the rules."""
  files = os.listdir(dir_path)

  # Process acc-state
  acc_files = get_files_with_prefix(files, "acc-state")
  if len(acc_files) > 1:
    latest_acc = get_latest_file(acc_files, dir_path)
    remove_files(acc_files, dir_path, latest_acc)

  # Process snapshot
  snap_files = get_files_with_prefix(files, "snapshot")
  if len(snap_files) > 1:
    latest_snap = get_latest_file(snap_files, dir_path)
    remove_files(snap_files, dir_path, latest_snap)

  # Process finished and graph-<int_arg>
  finished_files = get_files_with_prefix(files, "finished")
  if int_arg is not None and finished_files:
    graph_files = get_files_with_prefix(files, f"graph-{int_arg}")
    if graph_files:
      for f in finished_files:
        try:
          os.remove(os.path.join(dir_path, f))
          print(f"Deleted file: {os.path.join(dir_path, f)}")
        except Exception as e:
          print(f"Failed to delete {os.path.join(dir_path, f)}: {e}")


if __name__ == "__main__":
  main()
