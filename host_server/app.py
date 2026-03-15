import torch
from decoding import target_model_verify
from flask import Flask, request, jsonify
from config import target_model_name
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained(target_model_name)
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, dtype="auto", device_map="auto")

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if request.method == 'GET':
        return jsonify({
            'message': 'Use POST with Content-Type: application/json and keys input_ids, drafts.'
        }), 200

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            'error': 'Invalid or missing JSON body. Set Content-Type to application/json.'
        }), 400

    input_ids = data.get('input_ids')
    drafts = data.get('drafts')
    if input_ids is None or drafts is None:
        return jsonify({'error': 'Missing input_ids or drafts'}), 400

    input_ids = torch.tensor(input_ids).to(target_model.device)
    drafts = torch.tensor(drafts).to(target_model.device)

    verified = target_model_verify(
        target_model,
        input_ids,
        drafts,
        tokenizer=tokenizer,
    )

    return jsonify({'verified': verified.cpu().tolist()})

