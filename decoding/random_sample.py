import torch
from torch.nn import functional as F

def sample(logits, temperature, top_k= None, top_p= None):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    
    # Apply temperature scaling
    scaled_logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(scaled_logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        scaled_logits = torch.where(scaled_logits < min_values, float('-inf'), scaled_logits)

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        mask = cumulative_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False

        sorted_logits[mask] = float('-inf')
        scaled_logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)