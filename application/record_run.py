"""Run a baseline command with an exclusive, inspectable provenance record."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command is required")
    repo = Path(__file__).resolve().parents[1]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "cwd": os.getcwd(),
              "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "tracked_worktree_dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running"}
    # An existing record also protects partial/failed runs from being overwritten.
    with args.manifest.open("x") as handle:
        json.dump(record, handle, indent=2)
    try:
        code = subprocess.run(command).returncode
    except BaseException as exc:
        record.update(status="failed", error=type(exc).__name__)
        raise
    else:
        record.update(status="complete" if code == 0 else "failed", returncode=code)
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        args.manifest.write_text(json.dumps(record, indent=2) + "\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
