"""Robust aggregator for per-run batch JSONs.

Reads all JSON files in the provided input directory, computes per-method
mean/std/count/95% CI for reward, mean perplexity, mean distinct-n, and
writes `aggregated_multi.json` and `raw_records.csv` into the output dir.

Usage: python scripts/aggregate_batch_fix.py --in-dir outputs/experiments/batch --out-dir outputs/experiments/batch
"""
import argparse
import json
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt


def load_records(in_dir: Path):
    records = []
    for p in sorted(in_dir.glob('results_seed*.json')):
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
        except Exception as e:
            print('Failed to read', p, e)
            continue
        for run in data.get('runs', []):
            cfg = run.get('config')
            met = run.get('metrics', {})
            records.append({
                'source': p.name,
                'method': cfg if isinstance(cfg, str) else json.dumps(cfg),
                'avg_reward': float(met.get('avg_reward')) if met.get('avg_reward') is not None else None,
                'perplexity': float(met.get('perplexity')) if met.get('perplexity') is not None else None,
                'distinct_2': float(met.get('distinct_2')) if met.get('distinct_2') is not None else None,
                'distinct_3': float(met.get('distinct_3')) if met.get('distinct_3') is not None else None,
            })
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame):
    summary = {}
    if df.empty:
        return summary
    gb = df.groupby('method')
    for name, group in gb:
        n = int(group['avg_reward'].count())
        mean_reward = float(group['avg_reward'].mean())
        std_reward = float(group['avg_reward'].std(ddof=0)) if n > 1 else 0.0
        ci95 = 1.96 * std_reward / math.sqrt(n) if n > 0 else 0.0

        summary[name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'n': n,
            'ci95_reward': ci95,
            'mean_perplexity': float(group['perplexity'].mean()) if group['perplexity'].notna().any() else None,
            'std_perplexity': float(group['perplexity'].std(ddof=0)) if group['perplexity'].notna().any() else None,
            'mean_distinct_2': float(group['distinct_2'].mean()) if group['distinct_2'].notna().any() else None,
            'mean_distinct_3': float(group['distinct_3'].mean()) if group['distinct_3'].notna().any() else None,
        }
    return summary


def save_outputs(summary, df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'aggregated_multi.json', 'w', encoding='utf-8') as f:
        json.dump({'summary': summary}, f, indent=2)
    df.to_csv(out_dir / 'raw_records.csv', index=False)

    # simple plot
    configs = list(summary.keys())
    if configs:
        means = [summary[c]['mean_reward'] for c in configs]
        cis = [summary[c]['ci95_reward'] for c in configs]
        plt.figure(figsize=(6,4))
        plt.bar(configs, means, yerr=cis, capsize=6)
        plt.ylabel('Mean Reward (95% CI)')
        plt.title('Aggregated Reward by Method')
        plt.tight_layout()
        plt.savefig(out_dir / 'agg_reward_ci.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, default='outputs/experiments/batch')
    parser.add_argument('--out-dir', type=str, default='outputs/experiments/batch')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    df = load_records(in_dir)
    summary = aggregate(df)
    save_outputs(summary, df, out_dir)
    print('Wrote aggregated_multi.json and raw_records.csv to', out_dir)


if __name__ == '__main__':
    main()
