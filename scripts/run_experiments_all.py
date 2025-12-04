#!/usr/bin/env python3
"""Run experiments across many models, LoRA adapters, seeds and prompt sizes.

This orchestrator calls `scripts/run_experiments.py` for each combination and
produces a master CSV (`master_results.csv`) with one row per method (Base, RLHF,
PPLM, Hybrid) per run so you can do downstream analysis.

Usage example:
  PYTHONPATH=. python scripts/run_experiments_all.py --seeds 42 7 123 --num-prompts 30 60 100 --model-dirs gpt2 models/gpt2_poetry_small --lora-dirs poetry-lora-final --device cpu --quick
"""
import argparse
import subprocess
import json
import os
from pathlib import Path
import sys
import time
import traceback
import pandas as pd


def run_single(cmd, env=None):
    start = time.time()
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print('Run failed:', e)
        return False, time.time() - start
    return True, time.time() - start


def main():
    parser = argparse.ArgumentParser(description='Orchestrate experiments across many models and adapters')
    parser.add_argument('--seeds', nargs='+', default=['42', '7', '123'], help='Seeds (space separated or comma-separated)')
    parser.add_argument('--num-prompts', nargs='+', default=['30', '100'], help='Prompt counts (space separated or comma-separated)')
    parser.add_argument('--model-dirs', nargs='+', default=['gpt2'], help='Model directories or HF names to evaluate (space separated)')
    parser.add_argument('--lora-dirs', nargs='+', default=['poetry-lora-final'], help='LoRA adapter dirs to try; use "none" to skip')
    parser.add_argument('--device', type=str, default=None, help='Device to pass to run_experiments (cuda|cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick mode to speed up runs')
    parser.add_argument('--out-dir', type=str, default='outputs/experiments/all_runs', help='Output directory for artifacts')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers (not implemented; runs sequentially)')
    args = parser.parse_args()

    # Normalize seeds and nums (allow comma-separated tokens too)
    seeds = []
    for t in args.seeds:
        for p in str(t).split(','):
            if p.strip():
                seeds.append(int(p.strip()))

    nums = []
    for t in args.num_prompts:
        for p in str(t).split(','):
            if p.strip():
                nums.append(int(p.strip()))

    model_dirs = args.model_dirs
    lora_dirs = []
    for t in args.lora_dirs:
        for p in str(t).split(','):
            if p.strip():
                if p.strip().lower() in ('none', 'no'):
                    continue
                lora_dirs.append(p.strip())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []
    env = os.environ.copy()
    # Ensure run_experiments can import local src
    env['PYTHONPATH'] = env.get('PYTHONPATH', '')
    if env['PYTHONPATH']:
        env['PYTHONPATH'] = os.pathsep.join([env['PYTHONPATH'], '.'])
    else:
        env['PYTHONPATH'] = '.'

    total = 0
    for model_dir in model_dirs:
        for lora_dir in (lora_dirs if lora_dirs else [None]):
            for seed in seeds:
                for n in nums:
                    total += 1

    print(f'Running {total} combinations (sequential). This may take a while.')

    combo_idx = 0
    for model_dir in model_dirs:
        model_label = Path(model_dir).name
        for lora_dir in (lora_dirs if lora_dirs else [None]):
            lora_label = Path(lora_dir).name if lora_dir else 'nolora'
            for seed in seeds:
                for n in nums:
                    combo_idx += 1
                    print(f'[{combo_idx}/{total}] model={model_label} lora={lora_label} seed={seed} prompts={n}')
                    out_file = raw_dir / f'results_{model_label}_{lora_label}_s{seed}_n{n}.json'
                    cmd = [sys.executable, 'scripts/run_experiments.py', '--seed', str(seed), '--num-prompts', str(n), '--out', str(out_file)]
                    if args.device:
                        cmd += ['--device', args.device]
                    if args.quick:
                        cmd.append('--quick')
                    if model_dir:
                        cmd += ['--model-dir', model_dir]
                    if lora_dir:
                        cmd += ['--lora-dir', lora_dir]

                    try:
                        ok, elapsed = run_single(cmd, env=env)
                        if not ok:
                            print('Skipping aggregation for failed run', out_file)
                            continue
                        # load the produced JSON
                        if not out_file.exists():
                            print('Expected output not found:', out_file)
                            continue
                        with open(out_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        for run in data.get('runs', []):
                            cfg = run.get('config')
                            met = run.get('metrics', {})
                            row = {
                                'model_dir': model_dir,
                                'model_label': model_label,
                                'lora_dir': lora_dir if lora_dir else '',
                                'lora_label': lora_label,
                                'seed': seed,
                                'num_prompts': n,
                                'method': cfg,
                                'avg_reward': met.get('avg_reward'),
                                'perplexity': met.get('perplexity'),
                                'distinct_2': met.get('distinct_2'),
                                'distinct_3': met.get('distinct_3')
                            }
                            master_rows.append(row)

                    except Exception as e:
                        print('Exception during run:', e)
                        traceback.print_exc()
                        continue

    if master_rows:
        df = pd.DataFrame(master_rows)
        # coerce numeric
        for col in ['avg_reward', 'perplexity', 'distinct_2', 'distinct_3']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        master_csv = out_dir / 'master_results.csv'
        df.to_csv(master_csv, index=False)
        print('Saved master CSV to', master_csv)
    else:
        print('No results were collected.')


if __name__ == '__main__':
    main()
