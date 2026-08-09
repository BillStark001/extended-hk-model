import argparse
import os
from typing import List, Optional


def get_files_with_prefix(files: List[str], prefix: str) -> List[str]:
  """Return all files in the list that start with the given prefix."""
  return [f for f in files if f.startswith(prefix)]


def get_files_with_mtime(files: List[str], dir_path: str):
  """
  Given a list of filenames, return the filename of the latest file
  according to modified time.
  """
  if not files:
    return []
  files_with_mtime = [(f, os.path.getmtime(
      os.path.join(dir_path, f))) for f in files]
  files_with_mtime.sort(key=lambda x: x[1], reverse=True)
  return files_with_mtime


def is_record_flawed(base_dir: str, rec_name: str) -> bool:
  dir_path = os.path.join(base_dir, rec_name)
  files = os.listdir(dir_path)
  if len(get_files_with_prefix(files, 'acc-state')) > 2:
    return True
  if len(get_files_with_prefix(files, 'snapshot')) > 2:
    return True

  files_with_mtime = get_files_with_mtime(files, dir_path)
  if len(files_with_mtime) > 1:
    if files_with_mtime[0][1] - files_with_mtime[-1][1] > 60 * 10:
      return True
    t2 = 60 * 5
    if any(files_with_mtime[i][1] - files_with_mtime[i+1][1] > t2 for i in range(len(files_with_mtime) - 1)):
      return True

  return False


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
  args = parser.parse_args()

  target_dir: str = args.target_dir
  if not os.path.isdir(target_dir):
    print(f"{target_dir} is not a valid directory.")
    return

  # Walk through subfolders only (not recursively)
  for root, dirs, files in os.walk(target_dir):
    if root != target_dir:
      continue
    # Only process subdirectories, not the top-level directory itself
    for subdir in dirs:
      flawed = is_record_flawed(root, subdir)
      if flawed:
        print(subdir)
    break  # Don't go deeper


if __name__ == "__main__":
  main()
