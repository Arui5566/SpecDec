import json
import os
import urllib.error
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import draft_model_name, eos_tokens_id, target_model_name
from decoding import draft_model_generate


def edge_verify(input_ids: torch.Tensor, drafts: torch.Tensor, verify_url: str) -> torch.Tensor:
    payload = {
        "input_ids": input_ids.detach().cpu().tolist(),
        "drafts": drafts.detach().cpu().tolist(),
    }
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        verify_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Verify API HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Verify API unreachable: {exc.reason}") from exc

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Verify API returned invalid JSON: {response_body}") from exc

    verified = data.get("verified")
    if not isinstance(verified, list):
        raise RuntimeError(f"Verify API response missing 'verified': {data}")

    return torch.tensor(verified, device=input_ids.device, dtype=input_ids.dtype)


def main() -> None:
    # Hyper-params
    k = 5
    max_new_tokens = 128
    prompt = "写一篇题为The Hazards of AI的英语essay（100 words）。"
    verify_url = os.getenv("VERIFY_URL", "http://127.0.0.1:5000/verify")

    # Tokenizer for prompt formatting and decoding
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    # Only the draft model runs locally; verification is delegated to /verify
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        dtype="auto",
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

        # 2) Verify via edge API
        verified = edge_verify(current_input_ids, drafts, verify_url)
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
