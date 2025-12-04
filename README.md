# Reciprocal Contextual Poetry Generation

This repository contains a minimal, structured implementation of PPLM + a simplified RLHF proxy adapted from the notebooks in this workspace. The goal is to provide a reproducible experiment harness that:

- Implements a simplified PPLM steering using BoW attribute models.
- Implements a lightweight RLHF proxy: a reward model + small supervised fine-tune on preferred examples.
- Runs experiments comparing Base, PPLM, RLHF, and Hybrid approaches and computes simple quantitative metrics.

Structure

- `src/` : core modules (pplm, rlhf, generator, eval)
- `scripts/` : experiment runner
- `poetry_dataset.txt` : dataset (should be present in repo root)
- `outputs/experiments/results.json` : saved experiment outputs

Quick start

1. Create a Python environment and install dependencies:

# Reciprocal Contextual Poetry Generation

Compact repo summary — tools and data to reproduce and compare three complementary approaches for controlled poetry generation:

- PPLM: inference-time attribute steering (BoW or classifier-based steering)
- RLHF (proxy): supervised fine-tuning + learned reward-model reranking (proxy for full RLHF)
- Hybrid: fine-tuned model + PPLM steering at inference time

This repo converts experiments from notebooks into modular scripts and stores run outputs under `outputs-gpu/experiments/`.

## What's included

- `src/` — core library: generator, pplm implementation, reward-model (MLP), evaluation helpers
- `scripts/` — CLI runners: training, single-run experiments, batch sweeps, analysis
- `models-gpu/` — (optional, large) model snapshots and reward-model state_dicts (may not be included in all releases)
- `outputs-gpu/experiments/` — experiment CSVs, per-run JSONs, aggregated analysis (e.g., `all_runs/master_results.csv`, `analysis_summary.csv`)
- `now question_.txt` — short architecture summary + Mermaid diagram (architecture visualization)

## Quick start (minimal)

1. Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a single experiment (example):

```bash
python scripts/run_experiments.py --model-dir models/gpt2_poetry_small --num-prompts 30 --device cpu
```

3. Aggregate batch runs (example sweep):

```bash
python scripts/run_experiments_all.py --model-dirs "gpt2 models/gpt2_poetry_small" --lora-dirs "poetry-lora-final" --seeds "7 42 123" --num-prompts "30 60 100" --device cuda
```

4. Analysis (after runs):

```bash
python scripts/analysis_results.py
python scripts/analysis_by_model.py
# open outputs-gpu/experiments/by_model/report.html
```

## Key scripts (what to use)

- `scripts/train_models.py` — supervised fine-tune and reward-model training
- `scripts/run_experiments.py` — generate samples for Base / PPLM / RLHF / Hybrid and compute metrics
- `scripts/run_experiments_all.py` — orchestrate parameter sweeps and write `all_runs/master_results.csv`
- `scripts/analysis_results.py` / `scripts/analysis_by_model.py` — produce aggregated CSVs, summary stats, and plots
- `scripts/generate_interactive_step.py` — interactive stepwise generator for user feedback

## Models & defaults used in released experiments

- Base generator: GPT-2 family (default experiments use GPT-2 Small or `gpt2_poetry_small` when available)
- LoRA adapter defaults: rank=16, alpha=32, dropout=0.1 (adapter label `poetry-lora-final` in outputs)
- PPLM default step size: 0.02 (configurable via `--pplm-strength`)
- Reward model: small MLP over sentence embeddings (default hidden dims [256,128,64])

Note: the exported CSVs reflect runs using these defaults; if you want to run ablation sweeps (LoRA rank, PPLM step sizes, reward architectures, epochs) pass distinct labels/args to `run_experiments_all.py` so results are recorded.

## Reproducibility tips

- Fix random seeds (scripts support seeds 7, 42, 123 by convention).
- Record exact `transformers`, `tokenizers`, and `torch` versions when reproducing training runs.
- Large model artifacts (safetensors, checkpoints) should be excluded from Git and stored separately (Git LFS or external storage).

## Outputs & evaluation

- Metrics collected per run: `avg_reward`, `perplexity`, `distinct_2`, `distinct_3` (see `all_runs/master_results.csv`).
- Use `scripts/analysis_results.py` to compute grouped means, std, and to produce `analysis_summary.csv`.

## Troubleshooting & notes

- If GPU OOM occurs: use LoRA adapters, gradient accumulation, mixed precision (bf16/AMP), or run on CPU for small tests.
- If pushing to remote fails due to large files, ensure `.gitignore` includes virtualenvs and model folders and remove large tracked files from the index before pushing.

## Next steps / contributions

- Add human preference collection for stronger reward models.
- Run full PPO-based RLHF (e.g., via `trl`) for larger-scale experiments.
- Add CI, unit tests, and packaged example configs for reproducible ablation runs.

---

If you want, I can: (a) add a small `examples/` folder with a ready-to-run config for the exact experiment that produced `analysis_summary.csv`, or (b) produce an SVG of the Mermaid diagram and place it in `docs/`. Tell me which.
