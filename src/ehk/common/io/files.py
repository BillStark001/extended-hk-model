from typing import List

import os
import json
import base64
import zlib
import numpy as np
from numpy.typing import NDArray, DTypeLike
import logging

import pandas as pd


def find_and_rename(folder_path: str, suffix: str, rename: str):

  for filename in os.listdir(folder_path):
    if filename.endswith(suffix):
      old_path = os.path.join(folder_path, filename)
      new_path = os.path.join(folder_path, rename)
      if old_path == new_path:
        continue
      try:
        os.rename(old_path, new_path)
        print(f"File renamed from {filename} to {rename}")
        return True
      except OSError as e:
        print(f"Error renaming file: {e}")
        return False

  print(f"No file with suffix {suffix} found in {folder_path}")
  return False


def read_records(pat_file_paths: List[str], full_sim_len: int = 1):
  pat_files_raw = []
  for f in pat_file_paths:
    if not os.path.exists(f):
      continue
    with open(f, 'r', encoding='utf8') as fp:
      pat_files_ = json.load(fp)
      total_len = len(pat_files_)
      used_len = total_len - total_len % full_sim_len
      pat_files_raw += pat_files_[:used_len]

  pat_file_seed: dict = pat_files_raw[0] if pat_files_raw else {}

  # partition keys into 2 groups
  keys_0d = []
  keys_non_0d = []
  for (k, v) in pat_file_seed.items():
    if isinstance(v, (int, float, complex, str)) or v is None:
      keys_0d.append(k)
    else:
      keys_non_0d.append(k)

  vals_0d = pd.DataFrame(
      [
          [x[key] for key in keys_0d]
          for x in pat_files_raw
      ],
      columns=keys_0d
  )
  vals_non_0d = {
      k: [v[k] for v in pat_files_raw]
      for k in keys_non_0d
  }

  return vals_0d, vals_non_0d


def compress_array_to_b64(arr: NDArray) -> str:
  compressed = zlib.compress(arr.tobytes())
  return base64.b64encode(compressed).decode("utf-8")


def decompress_b64_to_array(b64_str: str, dtype: DTypeLike) -> NDArray:
  compressed = base64.b64decode(b64_str.encode("utf-8"))
  return np.frombuffer(zlib.decompress(compressed), dtype=dtype)


def init_logger(name: str | None, path: str | None = None):
  logger = logging.getLogger(name)
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  if path:
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  logger.setLevel(logging.DEBUG)
  return logger
