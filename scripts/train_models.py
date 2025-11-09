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
    parser.add_argument('--max-texts', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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

    print('Loading smaller DistilGPT2 for CPU-friendly finetuning...')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = 'distilgpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print('Starting supervised finetune (this may take a while)...')
    finetune_gpt2_supervised(model, tokenizer, texts, device=device, epochs=args.epochs, batch_size=args.batch_size)

    out_dir = repo_root / 'models' / 'distilgpt2_poetry_small'
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
