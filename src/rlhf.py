import torch
import torch.nn as nn
import numpy as np
from typing import List
import contextlib


class RewardModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

    def get_reward(self, texts: List[str], embedding_model) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            embeddings = embedding_model.encode(texts)
            embeddings_tensor = torch.FloatTensor(embeddings).to(next(self.parameters()).device)
            rewards = self.forward(embeddings_tensor)
            return rewards.cpu().numpy()


def train_reward_model(reward_model: RewardModel, embeddings: np.ndarray, ratings: List[float], device, epochs: int = 20, lr: float = 1e-3):
    reward_model.to(device)
    reward_model.train()
    X = torch.FloatTensor(embeddings).to(device)
    y = torch.FloatTensor(ratings).to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = reward_model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    return reward_model


def finetune_gpt2_supervised(model, tokenizer, texts: List[str], device, epochs: int = 1, lr: float = 5e-5, batch_size: int = 4, gradient_accumulation_steps: int = 1, gradient_checkpointing: bool = False, use_bf16: bool = False):
    """Simple supervised fine-tune of GPT-2 on positive texts as an RLHF proxy.

    This is intentionally small-scale and fast; it uses causal LM loss and
    performs a few gradient steps to bias the model toward preferred outputs.
    """
    model.to(device)
    model.train()
    # small memory-saver: empty CUDA cache before starting
    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Choose AMP dtype: use bfloat16 on H100 if requested, otherwise fp16 when CUDA available
    use_amp = (device.type == 'cuda') and hasattr(torch.cuda, 'amp')
    use_bf16 = use_bf16 and (device.type == 'cuda')
    scaler = None
    if use_amp and not use_bf16:
        scaler = torch.cuda.amp.GradScaler()

    # Tokenize with truncation to model max length to avoid OOM on long lines
    max_len = getattr(tokenizer, 'model_max_length', 1024)
    tokenized = [tokenizer.encode(t, add_special_tokens=True, max_length=max_len, truncation=True) for t in texts]
    inputs = [torch.tensor(t, dtype=torch.long) for t in tokenized]

    dataset = inputs
    # enable gradient checkpointing if requested (saves memory at cost of compute)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            maxlen = max([b.size(0) for b in batch])
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for j, b in enumerate(batch):
                input_ids[j, : b.size(0)] = b
                attention_mask[j, : b.size(0)] = 1

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Use a robust autocast context that handles different PyTorch versions/signatures
            if use_amp:
                # Try several autocast call signatures and fall back to cpu/non-amp if necessary
                autocast_cm = None
                # Prefer torch.amp.autocast if available
                amp_mod = getattr(torch, 'amp', None)
                cuda_amp_mod = getattr(torch, 'cuda', None) and getattr(torch.cuda, 'amp', None)

                tried = False
                if amp_mod is not None and hasattr(amp_mod, 'autocast'):
                    try:
                        if use_bf16:
                            autocast_cm = amp_mod.autocast(device_type='cuda', dtype=torch.bfloat16)
                        else:
                            autocast_cm = amp_mod.autocast()
                        tried = True
                    except TypeError:
                        # signature may not accept device_type; try dtype-only
                        try:
                            if use_bf16:
                                autocast_cm = amp_mod.autocast(dtype=torch.bfloat16)
                            else:
                                autocast_cm = amp_mod.autocast()
                            tried = True
                        except Exception:
                            autocast_cm = None

                if autocast_cm is None and cuda_amp_mod is not None and hasattr(cuda_amp_mod, 'autocast'):
                    try:
                        # cuda.amp.autocast typically accepts no args or dtype in newer versions
                        if use_bf16:
                            autocast_cm = cuda_amp_mod.autocast(dtype=torch.bfloat16)
                        else:
                            autocast_cm = cuda_amp_mod.autocast()
                        tried = True
                    except TypeError:
                        try:
                            autocast_cm = cuda_amp_mod.autocast()
                            tried = True
                        except Exception:
                            autocast_cm = None

                if autocast_cm is None:
                    # fallback: no autocast available, use nullcontext
                    autocast_cm = contextlib.nullcontext()

                with autocast_cm:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss

            # scale loss for gradient accumulation and (optionally) AMP
            loss = loss / float(gradient_accumulation_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # step optimizer every gradient_accumulation_steps
            step_idx = (i // batch_size) + 1
            if (step_idx % gradient_accumulation_steps) == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

    return model
