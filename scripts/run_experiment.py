"""Run a single experiment and save results to outputs/experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch

from src.model_utils import load_tokenizer_and_model
from scripts.experiment_utils import run_experiment, detect_device


def ensure_output_dir() -> Path:
    out = Path("outputs") / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="gpt2", help="Model variant: gpt2 or finetuned")
    p.add_argument("--mode", default="baseline", help="Mode: baseline|rlhf|pplm|hybrid")
    p.add_argument("--prompt", default="In the silent garden,", help="Prompt to condition on")
    p.add_argument("--topic", default=None, help="Topic/keyword for PPLM-like steering")
    p.add_argument("--num", type=int, default=5, help="Number of candidates to generate")
    p.add_argument("--max-len", type=int, default=40, help="Max generation length")
    args = p.parse_args(argv)

    device = detect_device()

    print(f"Loading model {args.variant} on device {device}...")
    model, tokenizer = load_tokenizer_and_model(variant=args.variant, device=device)

    print(f"Running experiment: variant={args.variant} mode={args.mode} prompt='{args.prompt}'")
    results = run_experiment(model, tokenizer, args.prompt, mode=args.mode, topic=args.topic, max_length=args.max_len, num_return_sequences=args.num)

    outdir = ensure_output_dir()
    outpath = outdir / f"results_{args.mode}_{args.variant}.json"
    serial = [{"text": t, "score": float(s)} for t, s in results]
    with outpath.open("w", encoding="utf-8") as f:
        json.dump({"variant": args.variant, "mode": args.mode, "prompt": args.prompt, "results": serial}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {outpath}")
    for i, (t, s) in enumerate(results, 1):
        print(f"\n== Candidate {i} (score={s:.4f}) ==\n{t}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
