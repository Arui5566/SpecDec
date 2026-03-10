import torch
from config import eos_tokens_id

def draft_model_generate(draft_model: torch.nn.Module,
                         input_ids: torch.Tensor,
                         k: int = 5)-> torch.Tensor:
    """
    Generate a draft sequence using the provided draft model.(USING GREEDY DECODING)

    Args:
        draft_model: The model used for drafting.
        input_ids: The input token IDs for the draft model.
        k: The number of tokens to draft.

    Returns:
        A tensor containing the drafted token IDs.

    """
    draft = []

    with torch.no_grad():
        for _ in range(k):
            logits = draft_model(input_ids).logits
            pre_token_id = logits[0,-1].argmax().item()
            draft.append(pre_token_id)

            if pre_token_id == eos_tokens_id:
                break

            next_token = torch.tensor([[pre_token_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return torch.tensor(draft, device=input_ids.device)