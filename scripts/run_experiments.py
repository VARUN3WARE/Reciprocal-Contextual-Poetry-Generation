"""Run small experiments comparing Base, PPLM, RLHF and Hybrid for poetry generation.

This script is intentionally small-scale so it can run on a laptop/CPU for demo and
produce quick quantitative comparisons. It reads `poetry_dataset.txt` from repo root.
"""
import os
import json
import random
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
try:
    # PEFT is optional at runtime; used to load LoRA adapters
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False
from sentence_transformers import SentenceTransformer

from src.generator_fixed import HybridPoetryGenerator
from src.pplm_fixed import BoWAttributeModel
from src.eval_fixed import perplexity, distinct_n, avg_reward


def load_prompts(dataset_path: str, max_prompts: int = 50):
    texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                texts.append(s)
    random.shuffle(texts)
    return texts[:max_prompts]


def run():
    parser = argparse.ArgumentParser(description='Run poetry generation experiments')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num-prompts', type=int, default=30, help='Number of prompts to sample from dataset')
    parser.add_argument('--quick', action='store_true', help='Quick mode: use fewer prompts and shorter max_length')
    parser.add_argument('--out', type=str, default=None, help='Output results file (JSON)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device to use')
    parser.add_argument('--model-dir', type=str, default=None, help='Optional path to a local finetuned model directory')
    parser.add_argument('--lora-dir', type=str, default=None, help='Optional path to a PEFT/LoRA adapter directory to evaluate (will be loaded on top of base GPT-2)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / 'poetry_dataset.txt'
    if not data_file.exists():
        print('poetry_dataset.txt not found in repo root. Please add dataset.')
        return

    

    # Set device (allow override from CLI)
    device = torch.device(args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Using device:', device)

    # Set seeds if provided
    if args.seed is not None:
        seed = args.seed
        print(f'Setting random seed = {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    print('Loading base tokenizer and model...')
    # Always load base GPT-2 tokenizer/model first
    base_model_name = 'gpt2'
    # If user provided a model-dir, use it as a base checkpoint (e.g., models/gpt2_poetry_small)
    if args.model_dir:
        md = Path(args.model_dir)
        if md.exists():
            print('Using provided --model-dir as base model:', md)
            try:
                base_model = GPT2LMHeadModel.from_pretrained(str(md))
                tokenizer = GPT2Tokenizer.from_pretrained(str(md))
            except Exception:
                print('Failed to load provided model-dir; falling back to official gpt2')
                base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        else:
            print('Provided --model-dir does not exist, falling back to official gpt2')
            base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    else:
        base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)

    base_model = base_model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    embed = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    prompts = load_prompts(str(data_file), max_prompts=args.num_prompts)
    if args.quick:
        prompts = prompts[:5]

    def run_for_model(model_obj, model_label: str):
        """Run the standard set of configs using model_obj (this will be used both for base and LoRA-wrapped models).

        We set both generator.base_model and generator.rlhf_model to the provided model so that the
        Base vs RLHF toggles compare the same underlying weights (no additional finetune applied).
        """
        print(f'\n--- Running experiments on model: {model_label} ---')
        gen = HybridPoetryGenerator(base_model=model_obj, tokenizer=tokenizer, embedding_model=embed, device=device)
        # For deterministic comparison, set the rlhf_model to the same model object (no extra finetune)
        gen.base_model = model_obj
        gen.rlhf_model = model_obj

        outputs_local = []
        configs = [
            ('Base', False, False, None),
            ('RLHF', True, False, None),
            ('PPLM', False, True, 'nature'),
            ('Hybrid', True, True, 'nature'),
        ]

        max_length = 30 if not args.quick else 20

        for name, use_rlhf, use_pplm, theme in configs:
            print('\n=== Running:', name, 'on', model_label)
            texts = []
            for p in prompts:
                out = gen.generate(p, theme=theme, use_rlhf=use_rlhf, use_pplm=use_pplm, max_length=max_length)
                texts.append(out['text'])

            met = {
                'method': name,
                'model_version': model_label,
                'avg_reward': avg_reward(gen.reward_model, embed, texts),
                'perplexity': perplexity(model_obj, tokenizer, texts, device),
                'distinct_2': distinct_n(texts, 2),
                'distinct_3': distinct_n(texts, 3),
            }
            print('Metrics:', met)
            outputs_local.append({'config': name, 'metrics': met, 'examples': texts[:5]})
        return outputs_local

    outputs = {'runs': []}

    # First: run with base model
    base_results = run_for_model(base_model, 'base')
    outputs['runs'].extend(base_results)

    # Next: attempt to load LoRA adapter and run experiments again
    lora_dir = Path(args.lora_dir) if args.lora_dir else (repo_root / 'poetry-lora-final')
    if lora_dir.exists() and _HAS_PEFT:
        try:
            print('\nLoading LoRA adapter from', lora_dir)
            # Attempt to prefer the adapter's base model name if available in adapter_config.json
            adapter_cfg_path = lora_dir / 'adapter_config.json'
            adapter_base = None
            if adapter_cfg_path.exists():
                try:
                    import json as _json
                    with open(adapter_cfg_path, 'r', encoding='utf-8') as _af:
                        acfg = _json.load(_af)
                        adapter_base = acfg.get('base_model_name_or_path')
                except Exception:
                    adapter_base = None

            load_base_name = adapter_base if adapter_base is not None else base_model_name
            print('Using base model for LoRA adapter:', load_base_name)
            base_for_lora = GPT2LMHeadModel.from_pretrained(load_base_name)
            # prefer tokenizer from adapter folder if present, otherwise use base tokenizer
            try:
                tokenizer_lora = GPT2Tokenizer.from_pretrained(str(lora_dir))
            except Exception:
                tokenizer_lora = tokenizer

            lora_model = PeftModel.from_pretrained(base_for_lora, str(lora_dir))
            lora_model = lora_model.to(device)
            lora_results = run_for_model(lora_model, f'lora({load_base_name})')
            outputs['runs'].extend(lora_results)
        except Exception as e:
            print('Warning: failed to load/run LoRA adapter:', e)
            print('Skipping LoRA evaluation.')
    else:
        if not lora_dir.exists():
            print('\nNo LoRA directory found at', lora_dir, '- skipping LoRA evaluation')
        else:
            print('\nPEFT not available in environment; cannot load LoRA adapters. Install `peft` to enable this.')

    # Save results
    out_dir = repo_root / 'outputs' / 'experiments'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out if args.out is not None else out_dir / 'results.json'
    # If out_file is a Path-like string, ensure it's a Path
    out_file = Path(out_file)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=2)

    # Save per-method example files and aggregate CSV
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    rows = []
    for run in outputs['runs']:
        cfg = run['config']
        met = run['metrics']
        rows.append({
            'config': cfg,
            'avg_reward': met.get('avg_reward'),
            'perplexity': met.get('perplexity'),
            'distinct_2': met.get('distinct_2'),
            'distinct_3': met.get('distinct_3')
        })

        # Save examples per method
        examples_path = out_dir / f"examples_{cfg}.txt"
        with open(examples_path, 'w', encoding='utf-8') as ef:
            for ex in run.get('examples', []):
                ef.write(ex.replace('\r', '') + '\n\n' + ('-' * 80) + '\n\n')

    df = pd.DataFrame(rows)
    csv_path = out_dir / 'aggregated_metrics.csv'
    df.to_csv(csv_path, index=False)

    # Auto-generate simple bar plots
    import matplotlib.pyplot as plt

    try:
        plt.figure(figsize=(6, 4))
        plt.title('Average Reward by Method')
        plt.bar(df['config'], df['avg_reward'], color='tab:blue')
        plt.ylabel('Avg Reward')
        plt.tight_layout()
        plt.savefig(figures_dir / 'avg_reward.png', dpi=150)

        plt.figure(figsize=(6, 4))
        plt.title('Perplexity by Method')
        plt.bar(df['config'], df['perplexity'], color='tab:orange')
        plt.ylabel('Perplexity')
        plt.tight_layout()
        plt.savefig(figures_dir / 'perplexity.png', dpi=150)

        plt.figure(figsize=(6, 4))
        plt.title('Distinct-n by Method')
        x = range(len(df['config']))
        plt.bar([p - 0.2 for p in x], df['distinct_2'], width=0.4, label='distinct_2')
        plt.bar([p + 0.2 for p in x], df['distinct_3'], width=0.4, label='distinct_3')
        plt.xticks(x, df['config'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / 'distinct_n.png', dpi=150)
        plt.close('all')
    except Exception as e:
        print('Warning: could not generate plots:', e)

    # Simple ranking: prefer higher avg_reward, tie-breaker lower perplexity, then higher distinct_2
    def rank_row(r):
        return (r['avg_reward'], -r['perplexity'], r['distinct_2'])

    best = max(rows, key=rank_row)
    summary = {
        'best_method': best['config'],
        'metrics': best
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=2)

    print('\nSaved results to', out_file)
    print('Saved aggregated CSV to', csv_path)
    print('Saved figures to', figures_dir)
    print('Saved summary to', out_dir / 'summary.json')


if __name__ == '__main__':
    run()
