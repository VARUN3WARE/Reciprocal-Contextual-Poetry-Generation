import torch
import torch.nn.functional as F
from typing import List


class BoWAttributeModel:
    """Simple BoW attribute model to compute a loss from logits.

    This mirrors the simplified BoW model used in the notebook for PPLM steering.
    """
    def __init__(self, word_list: List[str], tokenizer):
        self.word_list = word_list
        self.tokenizer = tokenizer
        self.target_token_ids = []
        for word in word_list:
            # prefix with space to match tokenization of words
            tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
            self.target_token_ids.extend(tokens)
        self.target_token_ids = list(set(self.target_token_ids))

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Return negative log prob of BoW tokens present in the logits.

        logits: (vocab,) or (1, vocab)
        """
        if logits.dim() == 1:
            probs = F.softmax(logits, dim=-1)
            target_probs = probs[self.target_token_ids].sum()
        else:
            probs = F.softmax(logits, dim=-1)
            target_probs = probs[:, self.target_token_ids].sum(dim=-1)
            target_probs = target_probs.mean()

        return -torch.log(target_probs + 1e-10)


def pplm_generate(
    model,
    tokenizer,
    prompt: str,
    bow_model: BoWAttributeModel,
    device,
    max_length: int = 40,
    step_size: float = 0.02,
    temperature: float = 0.9,
):
    """Simplified PPLM-style generation: perturb logits using BoW loss gradients.

    This implementation follows the notebook's simplified approach and is
    designed for CPU/quick experiments.
    """
    model.eval()
    # Truncate prompt to model's max length to avoid index errors on long inputs
    max_model_len = getattr(tokenizer, 'model_max_length', 1024)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if input_ids is None:
        input_ids = tokenizer.encode('', return_tensors='pt')
    # If prompt longer than model context, keep only the last `max_model_len-1` tokens
    if input_ids.size(1) > max_model_len - 1:
        print(f"[pplm_generate] prompt too long ({input_ids.size(1)} tokens), truncating to last {max_model_len-1} tokens")
        input_ids = input_ids[:, - (max_model_len - 1) :]
    input_ids = input_ids.to(device)
    generated = input_ids
    past_key_values = None

    for _ in range(max_length):
        # Ensure the current context doesn't exceed the model's position embeddings
        context_len = getattr(model.config, 'n_positions', getattr(tokenizer, 'model_max_length', 1024))
        # reserve one token for next generation
        max_context = max(1, context_len - 1)
        if generated.size(1) > max_context:
            # truncate to last max_context tokens and reset past_key_values so model recomputes
            generated = generated[:, -max_context:]
            past_key_values = None

        with torch.no_grad():
            inputs_for_model = generated[:, -1:] if past_key_values else generated
            outputs = model(
                input_ids=inputs_for_model,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            # get logits for the last token position
            unmodified_logits = outputs.logits[:, -1, :].detach()
            past_key_values = outputs.past_key_values

        # compute perturbation by treating logits as requires_grad
        with torch.enable_grad():
            logits_perturb = unmodified_logits.clone().requires_grad_(True)
            loss = bow_model.compute_loss(logits_perturb)
            loss.backward()
            grad = logits_perturb.grad
            if grad is not None:
                modified_logits = unmodified_logits - step_size * grad
            else:
                modified_logits = unmodified_logits

        probs = F.softmax(modified_logits / temperature, dim=-1)
        top_k = min(50, probs.shape[-1])
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices.gather(-1, next_token_idx)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text
