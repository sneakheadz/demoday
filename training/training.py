# train.py (Conceptual Example - Adapt as needed)
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch

# --- Configuration (Get from Env Vars passed by Soperator) ---
MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-...") # IMPORTANT: Use the correct Nemotron Model ID!
DATASET_ID = "huggingface/CodeAlpaca-20k"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/results")
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2")) # Adjust based on VRAM
GRAD_ACCUMULATION_STEPS = int(os.getenv("GRAD_ACCUMULATION_STEPS", "16")) # Adjust effective batch size
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
HF_TOKEN = os.getenv("HF_TOKEN", None) # Pass token if needed

# --- Load Model & Tokenizer ---
print(f"Loading model: {MODEL_ID}")
# Consider bf16 or fp16 based on GPU capability
# May require device_map='auto' if not handled by Soperator/torchrun/deepspeed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN, # Use token if model is gated
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# --- Load & Preprocess Dataset ---
print(f"Loading dataset: {DATASET_ID}")
dataset = load_dataset(DATASET_ID, cache_dir=CACHE_DIR, token=HF_TOKEN)

def format_instruction(example):
    # Adapt formatting based on Nemotron's fine-tuning recommendations
    # This is a generic example
    if example.get("input"):
        return f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    else:
        return f"Instruction: {example['instruction']}\nOutput: {example['output']}"

def preprocess(examples):
    texts = [format_instruction(ex) + tokenizer.eos_token for ex in examples]
    tokenized = tokenizer(texts, truncation=True, padding="longest", max_length=2048) # Adjust max_length
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Preprocessing dataset...")
processed_dataset = dataset['train'].map(
    preprocess,
    batched=True,
    remove_columns=dataset['train'].column_names
)
print(f"Processed dataset size: {len(processed_dataset)}")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    report_to="tensorboard",
    # Consider gradient_checkpointing=True for memory saving
    # Add deepspeed config path if using DeepSpeed
)

# --- Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Train ---
print("Starting Training...")
trainer.train()

# --- Save ---
print(f"Saving final model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete.")
