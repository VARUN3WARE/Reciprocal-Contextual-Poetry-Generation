"""Utilities to load tokenizer and model variants used by experiment scripts.

This module centralizes how we select the baseline GPT-2 and the user's
finetuned GPT-2 located in the repository (directory `Fine-Gpt2` by default).

Functions:
 - load_tokenizer_and_model(variant: str, device: str) -> (model, tokenizer)

The function supports `variant='gpt2'` (loads Hugging Face gpt2 from hub)
and `variant='finetuned'` (loads a local directory named `Fine-Gpt2` or a path
specified via the FINETUNED_MODEL_DIR environment variable).
"""
from __future__ import annotations

import os
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def _default_finetuned_dir() -> Path:
    # Allow override from environment for non-standard locations
    env = os.environ.get("FINETUNED_MODEL_DIR")
    if env:
        return Path(env)
    # Repository-local default
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "Fine-Gpt2"


def load_tokenizer_and_model(variant: str = "gpt2", device: str | torch.device = "cpu"):
    """Load tokenizer and model for given variant.

    Args:
        variant: 'gpt2' or 'finetuned'.
        device: torch device string or torch.device.

    Returns:
        (model, tokenizer) tuple. Model is moved to `device`.
    """
    device = torch.device(device if not isinstance(device, torch.device) else str(device))

    if variant.lower() == "gpt2":
        print("Loading baseline GPT-2 from Hugging Face hub...")
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    elif variant.lower() in ("finetuned", "fine", "fine-gpt2"):
        fd = _default_finetuned_dir()
        print(f"Loading finetuned model from {fd}")
        if not fd.exists():
            raise FileNotFoundError(f"Finetuned model directory not found at {fd}. Set FINETUNED_MODEL_DIR env var or place model at this path.")

        # Load tokenizer: if the finetuned dir doesn't include tokenizer files, fall back
        # to the canonical 'gpt2' tokenizer to keep generation working.
        try:
            tok = GPT2Tokenizer.from_pretrained(str(fd))
        except Exception as e_tok:
            print(f"Warning: could not load tokenizer from {fd}: {e_tok}")
            print("Falling back to the baseline 'gpt2' tokenizer for compatibility.")
            tok = GPT2Tokenizer.from_pretrained("gpt2")

        # Load model weights. Try PyTorch load first; if it fails, attempt TF->PT fallback.
        try:
            model = GPT2LMHeadModel.from_pretrained(str(fd))
        except Exception as e_model:
            print("Warning: standard PyTorch model load failed. Trying TF->PyTorch fallback. Error:", e_model)
            try:
                model = GPT2LMHeadModel.from_pretrained(str(fd), from_tf=True)
            except Exception as e_tf:
                # Provide a helpful message about common causes (protobuf missing when reading TF checkpoints)
                raise RuntimeError(
                    f"Failed to load finetuned model from {fd}. Tried both PyTorch and from_tf=True. "
                    f"TF fallback error: {e_tf}. If the model was saved in TensorFlow format, ensure 'protobuf' is installed. "
                    "Alternatively, re-export the model in PyTorch format or set FINETUNED_MODEL_DIR to a directory with tokenizer and PyTorch weights."
                ) from e_tf
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'gpt2' or 'finetuned'.")

    # ensure pad token
    if not getattr(tok, 'pad_token', None):
        tok.pad_token = tok.eos_token

    model = model.to(device)
    return model, tok
