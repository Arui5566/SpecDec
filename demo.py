from decoding import draft_model_generate, target_model_verify
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import draft_model_name, target_model_name, eos_tokens_id, prompt

tokenizer = AutoTokenizer.from_pretrained(target_model_name)

draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, dtype="auto", device_map="auto")
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, dtype="auto", device_map="auto")

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)

model_input = tokenizer([text], return_tensors="pt").to(draft_model.device)

draft_output = draft_model_generate(draft_model, model_input.input_ids, tokenizer, eos_tokens_id, k=5)

model_input_ids = model_input.input_ids.to(target_model.device)
draft_output = draft_output.to(target_model.device)

verified_output = target_model_verify(target_model, model_input_ids, draft_output, tokenizer)