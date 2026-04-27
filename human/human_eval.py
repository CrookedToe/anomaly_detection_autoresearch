#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    forwarded_args = sys.argv[1:]
    if "--results-root" not in forwarded_args:
        forwarded_args = ["--results-root", "results/human", *forwarded_args]
    command = [sys.executable, str(REPO_ROOT / "Human_eval"), *forwarded_args]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
