import os
from copy import deepcopy
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np
import torch

from .rlhf_fixed import RewardModel, train_reward_model, finetune_gpt2_supervised
from .pplm_fixed import BoWAttributeModel, pplm_generate


class HybridPoetryGenerator:
    def __init__(self, base_model, tokenizer, embedding_model, device, user_id: str = 'user'):
        self.base_model = base_model
        self.rlhf_model = deepcopy(base_model)
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.device = device
        self.user_id = user_id

        # reward model
        self.reward_model = RewardModel()
        self.reward_model.to(device)

        # simple themes
        self.THEME_WORDS = {
            'nature': ['tree', 'forest', 'mountain', 'river', 'sky', 'wind', 'flower', 'leaf'],
            'love': ['heart', 'passion', 'romance', 'beloved', 'embrace'],
            'ocean': ['wave', 'tide', 'sea', 'shore', 'salt', 'deep', 'horizon', 'blue'],
        }
        self.theme_models = {k: BoWAttributeModel(v, tokenizer) for k, v in self.THEME_WORDS.items()}

        self.interaction_history = []
        self.feedback_data = []
        self.stats = {'generations': 0, 'feedback_count': 0, 'rlhf_updates': 0, 'pplm_uses': 0}

    def generate(self, prompt: str, theme: Optional[str] = None, use_rlhf: bool = True, use_pplm: bool = False, pplm_strength: float = 0.02, max_length: int = 40) -> Dict:
        self.stats['generations'] += 1
        model = self.rlhf_model if use_rlhf else self.base_model

        if use_pplm and theme:
            self.stats['pplm_uses'] += 1
            bow = self.theme_models.get(theme, list(self.theme_models.values())[0])
            text = pplm_generate(model, self.tokenizer, prompt, bow, self.device, max_length=max_length, step_size=pplm_strength)
        else:
            text = self._generate_standard(model, prompt, max_length)

        reward = float(self.reward_model.get_reward([text], self.embedding_model)[0])

        interaction = {'timestamp': datetime.now().isoformat(), 'prompt': prompt, 'theme': theme, 'use_rlhf': use_rlhf, 'use_pplm': use_pplm, 'generated_text': text, 'reward': reward}
        self.interaction_history.append(interaction)
        return {'text': text, 'reward': reward, 'method': f"{'RLHF' if use_rlhf else 'Base'}{'+PPLM' if use_pplm else ''}", 'theme': theme}

    def _generate_standard(self, model, prompt: str, max_length: int):
        model.eval()
        with torch.no_grad():
            # Truncate prompt so that generation length does not exceed model's max length
            max_model_len = getattr(self.tokenizer, 'model_max_length', 1024)
            # reserve `max_length` tokens for generation
            max_input_len = max(1, max_model_len - int(max_length) - 1)
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=max_input_len, truncation=True)
            # Put inputs on the same device as the model to avoid device mismatch
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = self.device
            input_ids = input_ids.to(model_device)
            # compute generation max length safely
            gen_max = int(min(input_ids.shape[1] + max_length, max_model_len))
            attention_mask = torch.ones_like(input_ids).to(model_device)
            output = model.generate(input_ids, attention_mask=attention_mask, max_length=gen_max, do_sample=True, top_k=50, temperature=0.9, pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text

    def add_feedback(self, text: str, rating: float, feedback_type: str = 'rating'):
        self.stats['feedback_count'] += 1
        feedback = {'timestamp': datetime.now().isoformat(), 'text': text, 'rating': rating, 'type': feedback_type, 'user_id': self.user_id}
        self.feedback_data.append(feedback)

        # auto-trigger update every 10 feedbacks
        if len(self.feedback_data) >= 10 and self.stats['feedback_count'] % 10 == 0:
            self.update_from_feedback()

    def update_from_feedback(self, finetune_iters: int = 1):
        if len(self.feedback_data) < 5:
            return

        texts = [f['text'] for f in self.feedback_data]
        ratings = [f['rating'] for f in self.feedback_data]

        # update reward model
        embeddings = self.embedding_model.encode(texts)
        train_reward_model(self.reward_model, embeddings, ratings, device=self.device, epochs=20)

        # lightweight supervised fine-tune as an RLHF proxy using top-rated texts
        top_texts = [t for t, r in zip(texts, ratings) if r >= 0.7]
        if len(top_texts) >= 1:
            finetune_gpt2_supervised(self.rlhf_model, self.tokenizer, top_texts, device=self.device, epochs=finetune_iters)
            self.stats['rlhf_updates'] += 1
