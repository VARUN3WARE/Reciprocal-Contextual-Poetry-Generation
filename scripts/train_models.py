#!/usr/bin/env python3
"""Train small supervised GPT-2 on poetry_dataset.txt and train RewardModel with synthetic labels.

Saves:
- models/gpt2_poetry_small/ (HuggingFace format)
- models/reward_model.pt (state_dict)

Usage: PYTHONPATH=. python scripts/train_models.py --epochs 2 --reward-epochs 20
"""
import argparse
from pathlib import Path
import random
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

from src.rlhf import finetune_gpt2_supervised, RewardModel, train_reward_model


def load_texts(dataset_path: Path, max_texts: int = None):
    texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                texts.append(s)
    if max_texts:
        random.shuffle(texts)
        texts = texts[:max_texts]
    return texts


def synthetic_ratings(texts, themes=None):
    # Simple heuristic: if contains any theme word -> higher rating, else lower
    ratings = []
    themes = themes or {'nature': ['tree', 'forest', 'river', 'mountain', 'flower', 'leaf'],
                        'love': ['love', 'heart', 'beloved', 'passion'],
                        'ocean': ['sea', 'wave', 'shore', 'tide']}
    theme_words = set(w for words in themes.values() for w in words)
    import numpy as np
    for t in texts:
        score = 0.3
        txt = t.lower()
        if any(w in txt for w in theme_words):
            score += 0.4
        # boost short lines slightly
        if len(txt.split()) < 12:
            score += 0.1
        score = min(1.0, max(0.0, score + np.random.normal(0, 0.05)))
        ratings.append(float(score))
    return ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--reward-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps to simulate larger batch sizes')
    parser.add_argument('--max-texts', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-name', type=str, default='distilgpt2', help='HuggingFace model name to finetune (e.g., gpt2, gpt2-medium, gpt2-large, distilgpt2)')
    parser.add_argument('--use-bf16', action='store_true', help='Use bfloat16 autocast (H100 optimized)')
    parser.add_argument('--use-bnb', action='store_true', help='Load model in 8-bit via bitsandbytes (requires bitsandbytes)')
    parser.add_argument('--grad-checkpoint', action='store_true', help='Enable gradient checkpointing to save memory')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_file = repo_root / 'poetry_dataset.txt'
    if not data_file.exists():
        print('poetry_dataset.txt not found in repo root. Aborting.')
        return

    device = torch.device(args.device)
    print('Using device:', device)

    print('Loading dataset...')
    texts = load_texts(data_file, max_texts=args.max_texts)
    print(f'Loaded {len(texts)} texts (using up to {args.max_texts})')

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model '{args.model_name}' (use_bnb={args.use_bnb}, use_bf16={args.use_bf16})...")
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Optionally load in 8-bit if requested (requires bitsandbytes)
    try:
        if args.use_bnb:
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print('Model load warning:', e)
        print('Falling back to standard load...')
        model = AutoModelForCausalLM.from_pretrained(model_name)

    print('Starting supervised finetune (this may take a while)...')
    finetune_gpt2_supervised(model, tokenizer, texts, device=device, epochs=args.epochs, lr=5e-5, batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=args.grad_checkpoint, use_bf16=args.use_bf16)

    out_dir = repo_root / 'models' / f"{model_name}_poetry_small"
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Saving finetuned model to', out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print('Preparing reward model training...')
    embed = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    ratings = synthetic_ratings(texts)
    embeddings = embed.encode(texts)

    reward_model = RewardModel()
    print('Training reward model...')
    train_reward_model(reward_model, embeddings, ratings, device=device, epochs=args.reward_epochs)

    rm_path = repo_root / 'models' / 'reward_model.pt'
    torch.save(reward_model.state_dict(), rm_path)
    print('Saved reward model state to', rm_path)


if __name__ == '__main__':
    main()
