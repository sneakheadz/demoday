# train.py (Example for Llama-3-8B-Instruct - Adapt as needed!)
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
# Model Selection: Using Llama-3-8B-Instruct
# Note: Requires access approval on Hugging Face and HF_TOKEN
# Resource Needs: More manageable than 340B, but still needs significant VRAM (>24GB recommended)
#                 Consider PEFT (LoRA/QLoRA) for lower VRAM GPUs.
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
DATASET_ID = "huggingface/CodeAlpaca-20k" # Code Alpaca dataset
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/results") # Output directory for model checkpoints
CACHE_DIR = os.getenv("CACHE_DIR", "/cache")   # Cache directory for models/datasets
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "1")) # Number of training epochs
# Adjust batch size based on VRAM - start low
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2")) # PER DEVICE batch size
# Adjust accumulation steps to reach desired effective batch size
GRAD_ACCUMULATION_STEPS = int(os.getenv("GRAD_ACCUMULATION_STEPS", "8")) # Effective batch size = BATCH_SIZE * GRAD_ACCUMULATION_STEPS * num_gpus
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5")) # Common starting LR for 8B fine-tuning
HF_TOKEN = os.getenv("HF_TOKEN", None) # REQUIRED: Hugging Face token with access to Llama 3

# --- Load Model & Tokenizer ---
print(f"Loading model: {MODEL_ID}")
if HF_TOKEN is None:
    print("WARNING: HF_TOKEN environment variable not set. Access to gated Llama 3 model may fail.")

# Use bf16 for performance on compatible GPUs (Ampere+)
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,             # Use token for gated models
    cache_dir=CACHE_DIR,
    trust_remote_code=True,     # May still be needed for certain features/tokenizers
    torch_dtype=compute_dtype,  # Use bfloat16 or float16
    # device_map="auto",        # Consider if using multiple GPUs with accelerate w/o deepspeed/fsdp
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# Llama 3 specific tokenizer settings - Check if pad_token is needed/set
# Llama Tokenizer might handle padding differently, often no explicit pad_token is set by default.
# The DataCollator should handle padding during batching.
# If issues arise, uncomment and check:
# if tokenizer.pad_token is None:
#     print("Setting pad_token to eos_token")
#     tokenizer.pad_token = tokenizer.eos_token # Or another appropriate token if needed
#     model.config.pad_token_id = tokenizer.pad_token_id

# --- Load & Preprocess Dataset ---
print(f"Loading dataset: {DATASET_ID}")
dataset = load_dataset(DATASET_ID, cache_dir=CACHE_DIR, token=HF_TOKEN) # Use token if dataset is private

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! IMPORTANT: Using the