import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
exp_dir = root / "outputs-gpu" / "experiments"
master_csv = exp_dir / "all_runs" / "master_results.csv"
out_dir = exp_dir / "by_model"
out_dir.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(master_csv)
except Exception as e:
    print(f"ERROR: cannot read {master_csv}: {e}")
    raise

# coerce numeric
for c in ["avg_reward","perplexity","distinct_2","distinct_3"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

model_dirs = df['model_dir'].unique()
report_rows = []

for m in model_dirs:
    sub = df[df['model_dir'] == m]
    if sub.empty:
        continue
    # aggregate by method
    agg = sub.groupby('method')[['avg_reward','perplexity','distinct_2','distinct_3']].agg(['mean','std','count'])
    out_csv = out_dir / f"summary_{m.replace('/','_')}.csv"
    agg.to_csv(out_csv)
    report_rows.append((m, out_csv))

    # plots: avg_reward and perplexity (means with std)
    means = agg['avg_reward']['mean']
    stds = agg['avg_reward']['std']
    plt.figure(figsize=(6,4))
    means.plot(kind='bar', yerr=stds, capsize=4, color='C0')
    plt.title(f"Avg reward by method — {m}")
    plt.ylabel('avg_reward')
    plt.tight_layout()
    p1 = out_dir / f"avg_reward_{m.replace('/','_')}.png"
    plt.savefig(p1)
    plt.close()

    means_p = agg['perplexity']['mean']
    stds_p = agg['perplexity']['std']
    plt.figure(figsize=(6,4))
    means_p.plot(kind='bar', yerr=stds_p, capsize=4, color='C1')
    plt.title(f"Perplexity by method — {m}")
    plt.ylabel('perplexity')
    plt.tight_layout()
    p2 = out_dir / f"perplexity_{m.replace('/','_')}.png"
    plt.savefig(p2)
    plt.close()

# Build a simple HTML report
html_lines = ["<html><head><meta charset='utf-8'><title>Per-model analysis</title></head><body>",
              "<h1>Per-model analysis</h1>"]
for m, csvp in report_rows:
    html_lines.append(f"<h2>Model: {m}</h2>")
    dfcsv = pd.read_csv(csvp, index_col=0)
    html_lines.append(f"<h3>Summary table</h3>")
    html_lines.append(dfcsv.to_html())
    img1 = f"avg_reward_{m.replace('/','_')}.png"
    img2 = f"perplexity_{m.replace('/','_')}.png"
    html_lines.append(f"<h3>Plots</h3>")
    html_lines.append(f"<img src='{img1}' style='max-width:700px;'><br>")
    html_lines.append(f"<img src='{img2}' style='max-width:700px;'><br>")

html_lines.append("</body></html>")
report_html = out_dir / "report.html"
with open(report_html, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_lines))

print("Wrote per-model summaries and report to:")
print(out_dir)
print(report_html)
