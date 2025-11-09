import torch
import torch.nn as nn
import numpy as np
from typing import List


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


def finetune_gpt2_supervised(model, tokenizer, texts: List[str], device, epochs: int = 1, lr: float = 5e-5, batch_size: int = 4, gradient_accumulation_steps: int = 1):
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
    use_amp = (device.type == 'cuda') and hasattr(torch.cuda, 'amp')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Tokenize with truncation to model max length to avoid OOM on long lines
    max_len = getattr(tokenizer, 'model_max_length', 1024)
    tokenized = [tokenizer.encode(t, add_special_tokens=True, max_length=max_len, truncation=True) for t in texts]
    inputs = [torch.tensor(t, dtype=torch.long) for t in tokenized]

    dataset = inputs
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

            if use_amp:
                with torch.cuda.amp.autocast():
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
