import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="train.jsonl", split="train")

def tokenize_function(examples):
    prompts = [
        f"<|start_header_id|>system<|end_header_id|>\n\n{m[0]['content']}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{m[1]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{m[2]['content']}<|eot_id|>"
        for m in examples["messages"]
    ]
    return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    num_train_epochs=1,
    fp16=False,
    bf16=True,
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
)

trainer.train()
model.save_pretrained("./final_adapter")
print("Training complete and adapter saved.")
