import torch

def target_model_verify(target_model: torch.nn.Module,
                        input_data: torch.Tensor,
                        drafts: torch.Tensor) -> torch.Tensor:
    bonus = True
    verify_drafts = []
    k = drafts.shape[-1]
    print(k)
    print(input_data.shape, drafts.shape)
    input_ids = torch.cat([input_data, drafts.unsqueeze(0)], dim=-1)

    with torch.no_grad():
        logits = target_model(input_ids).logits
        targets = logits[0, -k-1:].argmax(-1)
        print(targets.shape, targets)
        for i, draft in enumerate(drafts):
            if draft == targets[i]:
                verify_drafts.append(draft)
            else:
                bonus = False
                correction = targets[i]
                verify_drafts.append(correction)
        if bonus:
            verify_drafts.append(targets[-1].item())
        
    return torch.tensor(verify_drafts, device=input_data.device)
