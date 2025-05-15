import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from accelerate import Accelerator
accelerator = Accelerator()

os.environ["SWANLAB_PROJECT"]="qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048
swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
    })

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            input = data["question"]
            output = f"<think> {data['think']} </think> \n {data['answer']}"
            message = {
                "instruction": PROMPT,
                "input": f"{input}",
                "output": output,
            }
            messages.append(message)
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """
    将数据集进行预处理
    """ 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        return_tensors="pt"  
    ).to(accelerator.device)  
    response = tokenizer(f"{example['output']}", add_special_tokens=False,return_tensors="pt").to(accelerator.device) 
    input_ids = torch.cat([
        instruction.input_ids[0], 
        response.input_ids[0], 
        torch.tensor([tokenizer.pad_token_id], device=accelerator.device)
    ])
    attention_mask = torch.cat([
        instruction.attention_mask[0],
        response.attention_mask[0],
        torch.tensor([1], device=accelerator.device)
    ])

    labels = torch.cat([
        torch.full((instruction.input_ids.shape[1],), -100, dtype=torch.long, device=accelerator.device),
        response.input_ids[0],
        torch.tensor([tokenizer.pad_token_id], device=accelerator.device)
    ])


    if input_ids.shape[0] > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids.cpu().tolist(),  
        "attention_mask": attention_mask.cpu().tolist(),
        "labels": labels.cpu().tolist()
    }

def predict(messages, model, tokenizer):
    device = accelerator.device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = accelerator.prepare(model_inputs)  
    with accelerator.autocast():
        generated_ids = model.generate(
            model_inputs.input_ids.to(device),
            attention_mask=model_inputs.attention_mask.to(device),
            max_new_tokens=MAX_LENGTH,
            synced_gpus=True  
        )

    if accelerator.is_main_process:
        generated_ids = accelerator.pad_across_processes(generated_ids, pad_index=tokenizer.pad_token_id)
        generated_ids = generated_ids[:, len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response
    else:
        return None  
    


# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-1.7B",  torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"
train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

# 得到验证集
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

print("------------------",len(train_dataset),len(eval_dataset),"--------------------")

args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B_1716",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=400,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    # report_to="swanlab",
    run_name="qwen3-1.7B",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)


trainer.train()

# 用测试集的前3条，主观看模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]
    response = predict(messages, model, tokenizer)
    response_text = f"""
    Question: {input_value}
    LLM:{response}
    """

    test_text_list.append(swanlab.Text(response_text))
    print(response_text)

swanlab.log({"Prediction": test_text_list})

swanlab.finish()
