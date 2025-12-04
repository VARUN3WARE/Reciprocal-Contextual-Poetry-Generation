"""Helpers for running lightweight generation experiments across variants and modes.

Modes implemented (lightweight proxies):
 - baseline: simple sampling generation.
 - rlhf: generate many candidates and rerank by model log-prob (proxy reward).
 - pplm: simple keyword steering by retrying generations until keyword present (proxy PPLM).
 - hybrid: pplm steering + rlhf reranking.

This is intentionally small and safe for quick experiments.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch

from src.model_utils import load_tokenizer_and_model


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_baseline(model, tokenizer, prompt: str, max_length: int = 50, num_return_sequences: int = 5, temperature: float = 1.0) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        max_length=min(max_length, inputs["input_ids"].shape[1] + max_length),
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )
    results = [tokenizer.decode(o[input_len:], skip_special_tokens=True).strip() if (input_len := inputs["input_ids"].shape[1]) is not None else tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]
    return results


def score_by_avg_logprob(model, tokenizer, texts: List[str]) -> List[float]:
    # compute average log-prob per token under the model as a proxy reward
    model.eval()
    scores = []
    for t in texts:
        with torch.no_grad():
            enc = tokenizer(t, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            outputs = model(**enc, labels=enc["input_ids"])
            # outputs.loss is mean negative log-likelihood over tokens
            # convert to average log-prob (higher is better)
            nll = outputs.loss.item()
            avg_logprob = -nll
            scores.append(avg_logprob)
    return scores


def generate_rlhf(model, tokenizer, prompt: str, max_length: int = 50, num_return_sequences: int = 20) -> List[str]:
    # Generate many candidates and rerank by avg log-prob
    candidates = generate_baseline(model, tokenizer, prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    scores = score_by_avg_logprob(model, tokenizer, candidates)
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ranked


def generate_pplm(model, tokenizer, prompt: str, keyword: str, max_length: int = 50, num_return_sequences: int = 5, attempts: int = 10) -> List[str]:
    # Lightweight steering: retry sampling until generated text contains the keyword or attempts exhausted
    results = []
    for _ in range(num_return_sequences):
        found = False
        for _ in range(attempts):
            out = generate_baseline(model, tokenizer, prompt, max_length=max_length, num_return_sequences=1)
            text = out[0]
            if keyword.lower() in text.lower():
                results.append(text)
                found = True
                break
        if not found:
            # fallback: take last sample
            results.append(text)
    return results


def run_experiment(model, tokenizer, prompt: str, mode: str, topic: str | None = None, max_length: int = 50, num_return_sequences: int = 5) -> List[Tuple[str, float]]:
    """Run a single experiment and return list of (text, score) tuples ordered by preference.

    Scores are model-derived average log-prob when available, else 0.
    """
    mode = mode.lower()
    if mode == "baseline":
        texts = generate_baseline(model, tokenizer, prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        scores = score_by_avg_logprob(model, tokenizer, texts)
        return list(zip(texts, scores))

    if mode == "rlhf":
        texts = generate_rlhf(model, tokenizer, prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        scores = score_by_avg_logprob(model, tokenizer, texts)
        return list(zip(texts, scores))

    if mode == "pplm":
        if not topic:
            topic = "poetry"
        texts = generate_pplm(model, tokenizer, prompt, keyword=topic, max_length=max_length, num_return_sequences=num_return_sequences)
        scores = score_by_avg_logprob(model, tokenizer, texts)
        return list(zip(texts, scores))

    if mode == "hybrid":
        if not topic:
            topic = "poetry"
        # pplm steering then rlhf rerank
        steered = generate_pplm(model, tokenizer, prompt, keyword=topic, max_length=max_length, num_return_sequences=num_return_sequences)
        scores = score_by_avg_logprob(model, tokenizer, steered)
        ranked = [t for _, t in sorted(zip(scores, steered), key=lambda x: x[0], reverse=True)]
        return list(zip(ranked, sorted(scores, reverse=True)))

    raise ValueError(f"Unknown mode: {mode}")
