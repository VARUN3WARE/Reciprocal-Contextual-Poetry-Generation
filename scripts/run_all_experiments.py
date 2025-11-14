"""Run all experiment combinations (modes x variants) with a short smoke prompt.

This orchestrator runs a quick short generation for each combination and stores results
in `outputs/experiments/`.
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    # ensure our repo root is on PYTHONPATH so scripts can import src
    env["PYTHONPATH"] = str(ROOT)
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Warning: command failed: {e}; continuing to next run.")


def main():
    modes = ["baseline", "rlhf", "pplm", "hybrid"]
    variants = ["gpt2", "finetuned"]

    prompt = "In the silent garden,"

    for mode in modes:
        for var in variants:
            cmd = [PY, str(ROOT / "scripts" / "run_experiment.py"), "--variant", var, "--mode", mode, "--prompt", prompt, "--num", "3", "--max-len", "40"]
            run(cmd)


if __name__ == "__main__":
    main()
