import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from decoding import draft_model_generate, target_model_verify
from config import draft_model_name, target_model_name, eos_tokens_id, prompt


def main():
    # Hyper-params
    k = 5
    max_new_tokens = 128
    prompt = '写一篇题为The Hazards of AI的essay。'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    # Load models
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Prepare input
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_input = tokenizer([text], return_tensors="pt")
    current_input_ids = model_input.input_ids.to(draft_model.device)

    generated = []
    finished = False

    while not finished and len(generated) < max_new_tokens:
        # 1) Draft
        drafts = draft_model_generate(
            draft_model,
            current_input_ids,
            tokenizer=tokenizer,
            eos_tokens_id=eos_tokens_id,
            k=min(k, max_new_tokens - len(generated)),
        )
        if drafts.numel() == 0:
            break

        # 2) Verify
        verified = target_model_verify(
            target_model,
            current_input_ids.to(target_model.device),
            drafts.to(target_model.device),
            tokenizer=tokenizer,
        )
        if verified.numel() == 0:
            break

        verified_list = verified.tolist()

        # 3) Append verified tokens to context
        append_ids = verified.unsqueeze(0).to(current_input_ids.device)
        current_input_ids = torch.cat([current_input_ids, append_ids], dim=-1)
        generated.extend(verified_list)

        # Stop condition
        if eos_tokens_id in verified_list:
            finished = True

    if eos_tokens_id in generated:
        generated = generated[: generated.index(eos_tokens_id)]

    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()