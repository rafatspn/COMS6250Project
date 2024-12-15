import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import random

# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 2e-5  # Learning rate for AdamW optimizer

# Step 1: Load Data from CSV Files
def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    texts = data['text'].tolist()
    labels = data['class'].tolist()

    # Convert labels to integers if not already
    unique_labels = sorted(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_int[label] for label in labels]

    return texts, labels

# Step 2: Tokenization using BERT Tokenizer
def tokenize_texts(texts, tokenizer, max_len):
    """Tokenize input texts and return input IDs, attention masks."""
    encodings = tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=max_len, 
        return_tensors="pt"
    )
    return encodings["input_ids"], encodings["attention_mask"]


def compute_accuracy_with_tolerance(model, data_loader, tolerance=3):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Check if prediction is within tolerance range
            correct += torch.sum(torch.abs(predicted - labels) <= tolerance).item()
            total += labels.size(0)

    return correct / total

# Step 7: Evaluation Function
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = [x.to(device) for x in batch]

            outputs = model(input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total

def calculate_accuracy(outputs, labels):
    """Calculate accuracy for a batch."""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)

# Step 4: Gradient-Based Text Generation
def generate_text(target_class, tokenizer, model, max_len=MAX_LEN, steps=300, lr=0.1):
    model.eval()

    # Initialize input_ids randomly as a LEAF tensor with requires_grad=True
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, max_len), dtype=torch.float32).to(device)
    input_ids.requires_grad_()  # Make input_ids a leaf tensor with gradients enabled

    optimizer_gen = optim.Adam([input_ids], lr=lr)

    for step in range(steps):
        optimizer_gen.zero_grad()

        # Ensure valid input_ids (rounded and clamped)
        input_ids_rounded = torch.round(input_ids).long().clamp(1, vocab_size - 1)

        # Generate attention mask (1s for valid tokens, 0s for padding)
        attention_mask = (input_ids_rounded != tokenizer.pad_token_id).long()

        # Forward pass through the model
        outputs = model(input_ids_rounded, attention_mask=attention_mask)
        logits = outputs.logits

        # Loss: maximize the logit for the target class
        loss = -logits[0, target_class]
        loss.backward()
        optimizer_gen.step()

        # Clamp input_ids to ensure they remain valid token indices
        with torch.no_grad():
            input_ids.clamp_(1, vocab_size - 1)

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

    # Decode the optimized input_ids to text
    final_input_ids = torch.round(input_ids).long().clamp(1, vocab_size - 1)
    generated_tokens = tokenizer.decode(final_input_ids[0], skip_special_tokens=True)
    return generated_tokens

# Step 3: Load Data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Main Execution (Dataset Reading)
train_file_path = "./ir_data/train.csv"  # Replace with your train CSV file path
test_file_path = "./ir_data/test.csv"    # Replace with your test CSV file path

train_texts, train_labels = load_csv_data(train_file_path)
test_texts, test_labels = load_csv_data(test_file_path)

train_input_ids, train_attention_masks = tokenize_texts(train_texts, tokenizer, MAX_LEN)
test_input_ids, test_attention_masks = tokenize_texts(test_texts, tokenizer, MAX_LEN)

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Step 4: Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 5: Initialize BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Step 6: Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        input_ids, attention_masks, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        correct_train += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")


# Save the Model
model_save_path = "ir_classifier_bert.pth"  # Path to save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate on Test Data
# test_accuracy = evaluate(model, test_loader)
train_accuracy = compute_accuracy_with_tolerance(model, train_loader, 3)
test_accuracy = compute_accuracy_with_tolerance(model, test_loader, 3)
print(f"Train Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Example of Text Generation
# target_class = 1  # Example: Generate text for class 1 (Sports)
# generated_text = generate_text(target_class, tokenizer, model)
# print(f"Generated Text for Class {target_class}:", generated_text)


