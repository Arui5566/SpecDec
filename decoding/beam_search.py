import torch
def beam_search(model,
                input_ids,
                beam_size,
                max_length,
                eos_token_id=None,
                device='cuda'):
    input_ids = input_ids.to(device)

    beams = [(input_ids, 0.0)]  # (sequence, score)
    completed_beams = []

    for step in range(max_length):
        new_beams = []
        for seq, score in beams:
            if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                completed_beams.append((seq, score))
                continue

            with torch.no_grad():
                outputs = model(seq)
                next_token_logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    new_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, new_token], dim=-1)
                    new_score = score + topk_log_probs[0, i].item()

                    new_beams.append((new_seq, new_score))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        if len(completed_beams) >= beam_size:
            break

    if len(completed_beams) == 0:
        completed_beams = beams
    
    completed_beams.sort(key=lambda x: x[1], reverse=True)
    best_seq, best_score = completed_beams[0]
    return best_seq, best_score