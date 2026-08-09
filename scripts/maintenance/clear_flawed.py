import argparse
import os
import shutil
def main():
  parser = argparse.ArgumentParser(
      description="Clean up files in subfolders according to rules.")
  parser.add_argument("target_dir", type=str, help="Target directory to scan")
  parser.add_argument("flawed_records", type=str,
                      help="Flawed records file (txt)")
  args = parser.parse_args()

  target_dir: str = args.target_dir
  all_files = os.listdir(target_dir)

  flawed_records_path: str = args.flawed_records
  with open(flawed_records_path, 'r', encoding="utf-8") as f:
    flawed_records = [x for x in (xx.strip() for xx in f.readlines()) if x]

  parsed_flawed_records = set()
  for flawed_record in flawed_records:
    x, y = flawed_record.split('_rw')
    rec_sign = 'rw' + y
    parsed_flawed_records.add(rec_sign)

  parsed_flawed_records_list = sorted(list(parsed_flawed_records))

  print(len(flawed_records), len(parsed_flawed_records_list))
  for xx in parsed_flawed_records_list:
    print(xx)
    for f in (x for x in all_files if xx in x):
      shutil.rmtree(os.path.join(target_dir, f))


if __name__ == "__main__":
  main()
