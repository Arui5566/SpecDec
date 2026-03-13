import time
import torch
from utils import Logger

logger = Logger("TargtModel")

def target_model_verify(target_model: torch.nn.Module,
                        input_data: torch.Tensor,
                        drafts: torch.Tensor,
                        tokenizer = None) -> torch.Tensor:
    bonus = True
    verify_drafts = []
    k = drafts.shape[-1]
    input_ids = torch.cat([input_data, drafts.unsqueeze(0)], dim=-1)
    logger.info(f'verify {k} drafts: {tokenizer.decode(drafts, skip_special_tokens=True) if tokenizer else drafts.tolist()}')

    with torch.no_grad():
        start_time = time.time()
        logits = target_model(input_ids).logits
        targets = logits[0, -k-1:].argmax(-1)

        for i, draft in enumerate(drafts):
            if tokenizer:
                draft_text = tokenizer.decode([draft.item()], skip_special_tokens=True)
                target_text = tokenizer.decode([targets[i].item()], skip_special_tokens=True)
            else:
                draft_text = str(draft.item())
                target_text = str(targets[i].item())
            if draft == targets[i]:
                verify_drafts.append(draft.item())
                logger.info(f'the {i+1}-th draft token "{draft_text}" is correct.')
            else:
                bonus = False
                correction = targets[i]
                verify_drafts.append(correction.item())
                logger.info(f'the {i+1}-th draft token "{draft_text}" is incorrect. Correcting to "{target_text}".')
                break

        if bonus:
            verify_drafts.append(targets[-1].item())
            if tokenizer:
                target_text = tokenizer.decode([targets[-1].item()], skip_special_tokens=True)
                logger.info(f'All drafts are correct. Bonus token "{target_text}" added.')

        end_time = time.time()
        logger.info(f"Verification completed : {end_time - start_time:.2f} seconds")
        logger.info(f"Tokens verification rate : {len(verify_drafts)/(end_time - start_time):.2f} t/s.")
        
    return torch.tensor(verify_drafts, device=input_data.device)    
