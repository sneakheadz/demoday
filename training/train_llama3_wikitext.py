import argparse
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from accelerate import Accelerator, PartialState # For distributed training awareness

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define script arguments using dataclasses for HfArgumentParser
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_qlora: bool = field(default=True, metadata={"help": "Whether to use QLoRA (4-bit quantization + LoRA)."})
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension (rank)."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling parameter."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability."})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", # Modules for Llama 3
        metadata={"help": "Comma separated list of module names to apply LoRA to."},
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Enable trusting remote code for models like Llama 3."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(default="wikitext", metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: str = field(default="wikitext-103-raw-v1", metadata={"help": "The configuration name of the dataset to use."})
    block_size: int = field(
        default=1024, # Adjust based on GPU memory and model context length capability
        metadata={
            "help": "Optional input sequence length after tokenization. Sequences shorter than this will be padded,"
            " and sequences longer will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )

@dataclass
class ScriptArguments:
    """
    Additional arguments specific to this script setup.
    """
    hf_token: Optional[str] = field(default=None, metadata={"help": "Hugging Face API token"})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Weights & Biases project name"})

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()

    # --- Setup ---
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Initialize Accelerator
    # Accelerator handles device placement and distributed setup
    # It will automatically detect Slurm environment variables like WORLD_SIZE, RANK etc.
    # distributed_state = PartialState() # Use this if you need rank/world_size before full Accelerator init
    # training_args.local_rank = distributed_state.local_process_index # Set local_rank for Trainer
    accelerator = Accelerator(log_with="wandb" if script_args.wandb_project else None)

    # Setup logging (distribute logging messages)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARN,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    # Setup WandB
    if accelerator.is_main_process and script_args.wandb_project:
        import wandb
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=script_args.wandb_project, config={**vars(model_args), **vars(data_args), **vars(training_args)})
        training_args.report_to = ["wandb"] # Ensure Trainer reports to wandb
    else:
        training_args.report_to = [] # Disable reporting on non-main processes or if no project set


    # --- Load Dataset ---
    logger.info(f"Loading dataset: {data_args.dataset_name} ({data_args.dataset_config_name})")
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        token=script_args.hf_token
    )

    # Create validation split if necessary
    if "validation" not in raw_datasets.keys():
        logger.info("Creating validation split.")
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            token=script_args.hf_token
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            token=script_args.hf_token
        )

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer for model: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=script_args.hf_token
    )
    # Set padding token if missing (GPT-like models often use EOS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer does not have a pad token, setting it to EOS token.")

    # --- Preprocess Data ---
    # Inspired by https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not training_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    block_size = min(data_args.block_size, tokenizer.model_max_length)
    if data_args.block_size > tokenizer.model_max_length:
         logger.warning(f"block_size ({data_args.block_size}) > model_max_length ({tokenizer.model_max_length}). Using {tokenizer.model_max_length}")


    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding instead if we wanted
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not training_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    # --- Load Model ---
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    quantization_config = None
    if model_args.use_qlora:
        logger.info("Using QLoRA (4-bit quantization).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
            bnb_4bit_use_double_quant=True,
        )
        # For QLoRA, device_map should be set to distribute the quantized model
        # Accelerator handles device placement when launching, so 'auto' or specific mapping works.
        # We let Trainer/Accelerator handle device map based on launch config
        device_map = {"": accelerator.local_process_index} # Maps the entire model to the current GPU assigned by Accelerator
        # device_map = "auto" # Often works well too with Accelerate

    else:
        logger.info("Loading model in full precision (or default).")
        device_map = None # Trainer will handle distribution if FSDP/DDP is used via accelerate

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map, # Let accelerate handle placement mostly
        trust_remote_code=model_args.trust_remote_code,
        token=script_args.hf_token,
        torch_dtype=torch.bfloat16, # Use bfloat16 for potential speedup and memory savings
    )

    # --- PEFT (LoRA/QLoRA) Setup ---
    if model_args.use_qlora:
        logger.info("Preparing model for k-bit training and applying LoRA.")
        # Prepare model for k-bit training (gradient checkpointing, etc.)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # Define LoRA config
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(",") if model_args.lora_target_modules else None,
            lora_dropout=model_args.lora_dropout,
            bias="none", # Usually recommended for LoRA
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA adapter
        model = get_peft_model(model, lora_config)
        logger.info("LoRA adapter applied.")
        model.print_trainable_parameters()
    elif training_args.gradient_checkpointing:
         # Enable gradient checkpointing even without LoRA if requested
         model.gradient_checkpointing_enable()

    # Required for training (pre-computation)
    model.config.use_cache = False

    # --- Trainer Setup ---
    # Data collator for language modeling (handles padding)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Training ---
    logger.info("*** Starting Training ***")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # --- Save Model ---
    # Saves the tokenizer and the PEFT adapter weights.
    logger.info(f"*** Saving model checkpoint to {training_args.output_dir} ***")
    # Use trainer.save_model() which handles PEFT saving automatically
    # It saves the adapter config and weights in the output_dir
    trainer.save_model() # Saves adapter model automatically if it's a PeftModel

    # Also save trainer state
    trainer.save_state()

    logger.info("Training finished successfully.")

    # Optional: Clean up WandB
    if accelerator.is_main_process and script_args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()
