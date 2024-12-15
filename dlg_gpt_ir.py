import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import random

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5  # Learning rate for AdamW optimizer
TEXT_GEN_STEPS = 300
TEXT_GEN_LR = 0.1

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Data from CSV Files
def load_csv_data(train_file_path, test_file_path):
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)

    train_texts = train_data['text'].tolist()
    train_labels = train_data['class'].tolist()

    test_texts = test_data['text'].tolist()
    test_labels = test_data['class'].tolist()

    # Filter out test labels that are not in train labels
    valid_labels = set(train_labels)
    filtered_test_texts = []
    filtered_test_labels = []

    for text, label in zip(test_texts, test_labels):
        if label in valid_labels:
            filtered_test_texts.append(text)
            filtered_test_labels.append(label)

    # Convert labels to integers
    unique_labels = sorted(valid_labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    train_labels = [label_to_int[label] for label in train_labels]
    filtered_test_labels = [label_to_int[label] for label in filtered_test_labels]

    num_labels = len(unique_labels)
    return train_texts, train_labels, filtered_test_texts, filtered_test_labels, num_labels

# Step 2: Tokenization using GPT-2 Tokenizer
def tokenize_texts(texts, tokenizer, max_len):
    """Tokenize input texts and return input IDs."""
    encodings = tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=max_len, 
        return_tensors="pt"
    )
    return encodings["input_ids"]
    

def evaluate_with_tolerance(model, dataloader, tolerance=3):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            _, preds = torch.max(outputs, 1)
            
            # Check if prediction is within tolerance
            correct += ((abs(preds - labels) <= tolerance).sum().item())
            total += labels.size(0)

# Step 3: Load Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Use EOS token as padding token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id  # Set pad_token_id for the model
model = model.to(device)

train_path = './ir_data/train.csv'
test_path = './ir_data/test.csv'

# Load data from CSV
train_texts, train_labels, test_texts, test_labels, num_labels = load_csv_data(train_path, test_path)

# Tokenize data
train_input_ids = tokenize_texts(train_texts, tokenizer, MAX_LEN)
test_input_ids = tokenize_texts(test_texts, tokenizer, MAX_LEN)

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_labels)
test_dataset = TensorDataset(test_input_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 5: Define GPT-based Classifier
class GPTClassifier(nn.Module):
    def __init__(self, model):
        super(GPTClassifier, self).__init__()
        self.gpt = model
        self.fc = nn.Linear(self.gpt.config.n_embd, num_labels)  # Output 4 classes

    def forward(self, input_ids):
        # Forward pass with hidden state extraction
        outputs = self.gpt(input_ids=input_ids, attention_mask=(input_ids != tokenizer.pad_token_id).long(), output_hidden_states=True)
        
        # Extract the last hidden state from the outputs
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden states
        cls_hidden_state = hidden_states[:, 0, :]  # First token (similar to [CLS])
        
        # Pass through the classification head
        logits = self.fc(cls_hidden_state)
        return logits

# Initialize the classifier
classifier = GPTClassifier(model).to(device)
optimizer = AdamW(classifier.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Step 8: Gradient-Based Text Generation
def generate_text_for_label(target_class, tokenizer, model, max_len=MAX_LEN, steps=TEXT_GEN_STEPS, lr=TEXT_GEN_LR):
    """
    Generate text that maximizes the probability of a target class label.
    """
    model.eval()

    # Initialize input_ids randomly within the valid range
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size, (1, max_len), dtype=torch.float32, requires_grad=True, device=device)

    optimizer_gen = optim.Adam([input_ids], lr=lr)

    for step in range(steps):
        optimizer_gen.zero_grad()

        # Round and clamp input_ids to ensure valid token indices
        input_ids_rounded = torch.round(input_ids).long().clamp(1, vocab_size - 1)

        # Create attention mask (ignore padding tokens)
        attention_mask = (input_ids_rounded != tokenizer.pad_token_id).long()

        # Forward pass through the classifier
        outputs = classifier(input_ids_rounded)
        logits = outputs  # Logits for all classes

        # Loss: Maximize the logit corresponding to the target class
        loss = -logits[0, target_class]
        loss.backward()
        optimizer_gen.step()

        # Clamp input_ids after each optimization step
        with torch.no_grad():
            input_ids.clamp_(1, vocab_size - 1)

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

    # Decode the optimized token indices back to text
    final_input_ids = torch.round(input_ids).long().clamp(1, vocab_size - 1)
    generated_text = tokenizer.decode(final_input_ids[0], skip_special_tokens=True)
    return generated_text

# Training Loop (unchanged)
for epoch in range(EPOCHS):
    classifier.train()
    total_loss = 0
    correct = 0
    total = 0

    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = classifier(input_ids)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.4f}")

# Step 7: Evaluate with Tolerance
print("Evaluating with tolerance...")
train_accuracy = evaluate_with_tolerance(classifier, train_loader, tolerance=3)
print(f"Train Accuracy (Tolerance 3): {train_accuracy * 100:.2f}%")

# Step 7: Evaluate with Tolerance
print("Evaluating with tolerance...")
test_accuracy = evaluate_with_tolerance(classifier, test_loader, tolerance=3)
print(f"Test Accuracy (Tolerance 3): {test_accuracy * 100:.2f}%")

# Step 8: Text Generation (unchanged)
target_label = 1  # Example target label
generated_text = generate_text_for_label(target_label, tokenizer, classifier)
print(f"Generated Text for Class {target_label}: {generated_text}")

