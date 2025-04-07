# train.py  
import torch  
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling  
  
# Load dataset  
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  
  
# Model and tokenizer  
model_checkpoint = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)  
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)  
  
# Tokenize dataset  
def tokenize_function(examples):  
    return tokenizer(examples["text"])  
  
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])  
tokenized_datasets.set_format("torch")  
  
# Data collator for causal language modeling  
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  
  
# Training arguments  
training_args = TrainingArguments(  
    output_dir="./results",  
    num_train_epochs=1,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,  
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    logging_dir="./logs",  
    logging_steps=10,  
    fp16=True,  
)  
  
# Trainer  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=tokenized_datasets["train"],  
    eval_dataset=tokenized_datasets["validation"],  
    tokenizer=tokenizer,  
    data_collator=data_collator,  
)  
  
# Training  
trainer.train()  
  
# Save model  
trainer.save_model("./final_model")  