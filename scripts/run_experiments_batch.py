"""Run experiments across multiple seeds and prompt sizes, aggregate results.

This script calls `scripts/run_experiments.py` for each combination of seed and
num-prompts, saves individual results, and then computes mean/std/95% CI per
method across runs. It also generates aggregated CSV and figures.

Use quick mode when testing to reduce runtime.
"""
import argparse
import subprocess
import json
from pathlib import Path
import time
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_single(seed, num_prompts, quick, out_path):
    cmd = [sys.executable, 'scripts/run_experiments.py', '--seed', str(seed), '--num-prompts', str(num_prompts), '--out', str(out_path)]
    if quick:
        cmd.append('--quick')
    print('Running:', ' '.join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    print('Completed in {:.1f}s'.format(time.time() - start))


def aggregate(results_paths, out_dir: Path):
    records = []
    for p in results_paths:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for run in data.get('runs', []):
            cfg = run['config']
            met = run['metrics']
            records.append({
                'source': str(p.name),
                'config': cfg,
                'avg_reward': met.get('avg_reward'),
                'perplexity': met.get('perplexity'),
                'distinct_2': met.get('distinct_2'),
                'distinct_3': met.get('distinct_3')
            })

    df = pd.DataFrame(records)
    # Group by config and compute stats
    agg = df.groupby('config').agg(['mean', 'std', 'count'])

    summary = {}
    for cfg in agg.index:
        row = agg.loc[cfg]
        m_reward = float(row[('avg_reward', 'mean')])
        s_reward = float(row[('avg_reward', 'std')]) if not math.isnan(row[('avg_reward', 'std')]) else 0.0
        n = int(row[('avg_reward', 'count')])
        ci95 = 1.96 * s_reward / math.sqrt(n) if n > 0 else 0.0

        summary[cfg] = {
            'mean_reward': m_reward,
            'std_reward': s_reward,
            'n': n,
            'ci95_reward': ci95,
            'mean_perplexity': float(row[('perplexity', 'mean')]),
            'std_perplexity': float(row[('perplexity', 'std')]) if not math.isnan(row[('perplexity', 'std')]) else 0.0,
            'mean_distinct_2': float(row[('distinct_2', 'mean')]),
            'mean_distinct_3': float(row[('distinct_3', 'mean')])
        }

    # Save aggregated JSON and CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'aggregated_multi.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary}, f, indent=2)

    # Save CSV of raw records
    df.to_csv(out_dir / 'raw_records.csv', index=False)

    # Create plots with error bars for reward
    configs = list(summary.keys())
    means = [summary[c]['mean_reward'] for c in configs]
    cis = [summary[c]['ci95_reward'] for c in configs]

    plt.figure(figsize=(6,4))
    plt.bar(configs, means, yerr=cis, capsize=6)
    plt.ylabel('Mean Reward (95% CI)')
    plt.title('Aggregated Reward by Method')
    plt.tight_layout()
    plt.savefig(out_dir / 'agg_reward_ci.png', dpi=150)
    plt.close()

    print('Aggregated results saved to', out_dir)
    return out_dir / 'aggregated_multi.json'


def main():
    parser = argparse.ArgumentParser(description='Batch-run experiments across seeds and prompt sizes')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated seeds, e.g. 42,7,123')
    parser.add_argument('--num-prompts', type=str, default='30', help='Comma-separated prompt counts, e.g. 30,60')
    parser.add_argument('--quick', action='store_true', help='Quick mode to speed up runs')
    parser.add_argument('--out-dir', type=str, default='outputs/experiments/batch', help='Output directory for batch artifacts')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    nums = [int(n) for n in args.num_prompts.split(',') if n.strip()]

    out_dir = Path(args.out_dir)
    results_paths = []

    for s in seeds:
        for n in nums:
            out_file = out_dir / f'results_seed{s}_n{n}.json'
            out_file.parent.mkdir(parents=True, exist_ok=True)
            run_single(s, n, args.quick, out_file)
            results_paths.append(out_file)

    aggregate(results_paths, out_dir)


if __name__ == '__main__':
    main()
