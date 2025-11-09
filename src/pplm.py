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
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    past_key_values = None

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
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
