# waitpid

waitpid blocks until a target PID exits, then runs a follow-up command. It is useful for serializing scripts that must not overlap.

## Features

- Polls a PID until it disappears.
- Treats Unix zombie processes as exited.
- Supports a timeout with exit code 124.
- Runs commands directly by default so arguments are preserved.
- Provides optional shell mode for single command strings.

## Usage

``` bash
go run . -- 12345 python main.py
go run . --poll 1s 12345 ./build.sh --fast
go run . --shell --timeout 60s 12345 "echo done && dir"
```

## Exit Codes

- 0: the PID exited and the command succeeded.
- 1: monitoring failed or the command could not be started.
- 2: invalid arguments.
- 124: timeout while waiting for the PID.
