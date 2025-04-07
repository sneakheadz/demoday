import os
import torch
from transformers import LLaMAForCausalLM, LLaMATokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Define hyperparameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 2048

# Load dataset
dataset_name = "salesforce/wikitext"
dataset = load_dataset(dataset_name, split="train")

# Define tokenizer
tokenizer = LLaMATokenizer.from_pretrained("llama-3.1-8b")

# Define dataset class
class WikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=MAX_SEQ_LENGTH, padding="max_length", truncation=True)
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

# Create dataset instance
dataset_instance = WikiTextDataset(dataset, tokenizer)

# Create data loader
data_loader = DataLoader(dataset_instance, batch_size=BATCH_SIZE, shuffle=True)

# Load model
model = LLaMAForCausalLM.from_pretrained("llama-3.1-8b")

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss()

# Train model
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

model.save_pretrained("llama-3.1-8b-finetuned")