# hdd-bridge

hdd-bridge watches a fast source directory, finds completed simulation folders, and incrementally mirrors them to a slower destination directory.

## Design

- A subdirectory is eligible when it has at least one file starting with finished.
- Any file starting with lock keeps the subdirectory in place.
- Copying starts only after the newest finished marker is older than the configured delay.
- Destination sync is incremental, so unchanged files are skipped on later scans.
- A completion marker prevents repeated full copies after a successful transfer.

## Usage

``` bash
go run . --src D:/sim-cache --dst E:/sim-archive
go run . --src D:/sim-cache --dst E:/sim-archive --interval 30s --minelapsed 2m
go run . --src D:/sim-cache --dst E:/sim-archive --buffer-mib 8 --fsync
```

## Notes

- Keep source and destination on separate, non-overlapping directories.
- The default settings favor throughput over maximum durability.
- Symlinks are skipped intentionally so a simulation folder cannot escape the mirrored tree.
