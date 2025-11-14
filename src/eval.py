import math
from typing import List
import torch
import numpy as np


def perplexity(model, tokenizer, texts: List[str], device):
    model.eval()
    losses = []
    with torch.no_grad():
        for t in texts:
            try:
                enc = tokenizer.encode(t, return_tensors='pt')
                # skip empty encodings
                if enc is None or enc.numel() == 0 or enc.size(-1) == 0:
                    continue
                enc = enc.to(device)
                try:
                    outputs = model(input_ids=enc, labels=enc)
                except RuntimeError:
                    # skip problematic examples that cause shape/indexing issues
                    continue
                loss = outputs.loss.item()
                losses.append(loss)
            except Exception:
                # be resilient: skip this text if tokenization/model fails for any reason
                continue
    # exp of average loss
    return float(np.exp(np.mean(losses))) if len(losses) > 0 else float('inf')


def distinct_n(texts: List[str], n: int = 2):
    ngrams = set()
    total = 0
    for t in texts:
        tokens = t.split()
        for i in range(len(tokens) - n + 1):
            ngrams.add(' '.join(tokens[i:i+n]))
            total += 1
    return len(ngrams) / total if total > 0 else 0.0


def avg_reward(reward_model, embedding_model, texts: List[str]):
    if len(texts) == 0:
        return 0.0
    rewards = reward_model.get_reward(texts, embedding_model)
    return float(rewards.mean())
