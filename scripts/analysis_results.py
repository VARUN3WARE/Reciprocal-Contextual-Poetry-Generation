import json
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
exp_dir = root / "outputs-gpu" / "experiments"
master_csv = exp_dir / "all_runs" / "master_results.csv"
results_json = exp_dir / "results.json"
aggregated_csv = exp_dir / "aggregated_metrics.csv"

out_summary_csv = exp_dir / "analysis_summary.csv"
out_report = exp_dir / "analysis_report.txt"

# Load master CSV
try:
    df = pd.read_csv(master_csv)
except Exception as e:
    print(f"Failed to read {master_csv}: {e}")
    df = pd.DataFrame()

# Coerce numeric columns
num_cols = ["avg_reward", "perplexity", "distinct_2", "distinct_3"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Grouped stats
grouped = None
if not df.empty:
    grouped = df.groupby("method")[num_cols].agg(["mean", "std", "count"]).round(6)
    grouped.to_csv(out_summary_csv)

# Load results.json (per-run detailed)
try:
    with open(results_json, "r", encoding="utf-8") as f:
        res = json.load(f)
except Exception as e:
    print(f"Failed to read {results_json}: {e}")
    res = {}

# Prepare top/bottom examples per method
examples_dir = exp_dir
if isinstance(res, dict) and "runs" in res:
    runs = res["runs"]
    # Build list of (method, avg_reward, examples)
    recs = []
    for r in runs:
        method = r.get("config") or r.get("metrics", {}).get("method")
        metrics = r.get("metrics", {})
        avg = metrics.get("avg_reward")
        exs = r.get("examples", [])
        # join examples into single text
        joined = "\n----\n".join([e for e in exs if e]) if exs else ""
        recs.append({"method": method, "avg_reward": avg, "examples": joined})
    df_runs = pd.DataFrame(recs)
    # For each method, pick top3 and bottom3 by avg_reward
    methods = df_runs['method'].unique()
    top_files = []
    report_lines = []
    for m in methods:
        sub = df_runs[df_runs['method'] == m].dropna(subset=['avg_reward'])
        if sub.empty:
            continue
        sub_sorted = sub.sort_values('avg_reward', ascending=False)
        top3 = sub_sorted.head(3)
        bot3 = sub_sorted.tail(3)
        # write files
        tf = exp_dir / f"top_examples_{m}.txt"
        bf = exp_dir / f"bottom_examples_{m}.txt"
        with open(tf, 'w', encoding='utf-8') as f:
            f.write(f"Top 3 examples for {m}\n\n")
            for i, row in top3.iterrows():
                f.write(f"avg_reward: {row['avg_reward']}\n")
                f.write(row['examples'] + "\n\n---\n\n")
        with open(bf, 'w', encoding='utf-8') as f:
            f.write(f"Bottom 3 examples for {m}\n\n")
            for i, row in bot3.iterrows():
                f.write(f"avg_reward: {row['avg_reward']}\n")
                f.write(row['examples'] + "\n\n---\n\n")
        report_lines.append(f"Method: {m} | top_avg_reward: {top3['avg_reward'].mean():.6f} | bottom_avg_reward: {bot3['avg_reward'].mean():.6f}")

    # Write textual report
    with open(out_report, 'w', encoding='utf-8') as f:
        f.write("Analysis Report\n================\n\n")
        if grouped is not None:
            f.write("Grouped statistics (see analysis_summary.csv):\n")
            f.write(grouped.to_string())
            f.write("\n\n")
        f.write("Top/bottom examples summary:\n")
        for line in report_lines:
            f.write(line + "\n")

else:
    with open(out_report, 'w', encoding='utf-8') as f:
        f.write("No detailed runs found in results.json to extract examples.\n")

print("Analysis complete. Outputs written:")
print(out_summary_csv)
print(out_report)