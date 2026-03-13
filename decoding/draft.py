import torch
from utils import Logger
import time

logger = Logger("DraftModel")

def draft_model_generate(draft_model: torch.nn.Module,
                         input_ids: torch.Tensor,
                         tokenizer = None,
                         eos_tokens_id: int = None,
                         k: int = 5,
                         )-> torch.Tensor:
    """
    Generate a draft sequence using the provided draft model.(USING GREEDY DECODING)

    Args:
        draft_model: The model used for drafting.
        input_ids: The input token IDs for the draft model.
        k: The number of tokens to draft.
        tokenizer: The tokenizer for decoding token IDs.

    Returns:
        A tensor containing the drafted token IDs.

    """
    draft = []
    past_key_values = None

    with torch.no_grad():
        start_time = time.time()

        outputs = draft_model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        pre_token_id = outputs.logits[0, -1].argmax().item()
        draft.append(pre_token_id)

        if pre_token_id == eos_tokens_id:
            logger.info("End of sequence token generated. Stopping draft generation.")
        else:
            for _ in range(k - 1):
                next_token = torch.tensor([[pre_token_id]], device=input_ids.device)

                outputs = draft_model(
                    next_token,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                pre_token_id = outputs.logits[0, -1].argmax().item()
                draft.append(pre_token_id)

                if pre_token_id == eos_tokens_id:
                    logger.info("End of sequence token generated. Stopping draft generation.")
                    break

        end_time = time.time()

        if tokenizer and draft:
            draft_text = tokenizer.decode(draft, skip_special_tokens=True)
            logger.info(f"Draft generated: {draft_text}")

        logger.info(f"Draft generation completed : {end_time - start_time:.2f} seconds.")
    
    return torch.tensor(draft, device=input_ids.device)