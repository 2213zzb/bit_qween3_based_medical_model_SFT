import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids,attention_mask=model_inputs.attention_mask, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 微调后的模型
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/output/Qwen3-1.7B_1716/checkpoint-360", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/output/Qwen3-1.7B_1716/checkpoint-360",  torch_dtype=torch.float16).to("cuda")

# 预训练模型
# tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-1.7B",  torch_dtype=torch.float16).to("cuda")

test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}
instruction = test_texts['instruction']
input_value = test_texts['input']
messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

response = predict(messages, model, tokenizer)
print(response)
