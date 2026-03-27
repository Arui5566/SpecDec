from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import draft_mllm_name, prompt, target_mllm_name
import torch

draft_mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    draft_mllm_name,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(draft_mllm_name, min_pixels=min_pixels, max_pixels=max_pixels)

tokenizer = processor.tokenizer

prompt = '''
Identify the open book in the image and generate 5–8 coordinate points that form a closed polygon fully enclosing the book.

Requirements:
- The polygon must tightly enclose the visible area of the open book.
- Provide between 5 and 8 points in clockwise order.
- Each point should be an integer pixel coordinate in the format [x, y].
- All coordinates must lie within the image boundaries.

Output format (strictly follow this JSON format, no extra text):
{
  "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
}
'''

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                # "image": "https://img95.699pic.com/photo/50261/8377.jpg_wh860.jpg",
                "image": "./book.jpg",
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
).to("cuda")

generated_ids = draft_mllm.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)