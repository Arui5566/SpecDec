import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decoding import draft_model_generate, target_model_verify
from config import draft_mllm_name, target_mllm_name, prompt, eos_tokens_id

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
draft_mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    draft_mllm_name,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

target_mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    target_mllm_name,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(draft_mllm_name, min_pixels=min_pixels, max_pixels=max_pixels)

tokenizer = processor.tokenizer

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
current_input_ids = inputs.input_ids.to("cuda")

generated = []
finished = False
max_new_tokens = 128
k = 5

while not finished and len(generated) < max_new_tokens:
    # 1) Draft
    drafts = draft_model_generate(
        draft_mllm,
        current_input_ids,
        tokenizer=tokenizer,
        eos_tokens_id=eos_tokens_id,
        k=min(k, max_new_tokens - len(generated)),
    )
    if drafts.numel() == 0:
        break

    # 2) Verify
    verified = target_model_verify(
        target_mllm,
        current_input_ids.to(target_mllm.device),
        drafts.to(target_mllm.device),
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

print("Verify output: ", tokenizer.decode(generated, skip_special_tokens=True))


