import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 2e-5  # Learning rate for AdamW optimizer

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

# Step 3: Evaluation Function with Tolerance
def evaluate_with_tolerance_and_save(model, dataloader, device, output_file, tolerance=3):
    model.eval()
    correct = 0
    total = 0

    original_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (torch.abs(predictions - labels) <= tolerance).sum().item()
            total += labels.size(0)

            original_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    accuracy = correct / total

    # Save results to CSV
    results_df = pd.DataFrame({
        "original": original_labels,
        "predicted": predicted_labels
    })
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return accuracy


# Step 4: Gradient-Based Text Generation
def generate_text(target_class, vocab, model, max_len=MAX_LEN):
    model.eval()
    # Initialize input indices randomly as NumPy array
    input_indices = np.random.randint(1, len(vocab), (1, max_len))
    input_tensor = torch.tensor(input_indices, dtype=torch.float32, requires_grad=True)

    optimizer_gen = optim.Adam([input_tensor], lr=0.1)

    for step in range(300):
        optimizer_gen.zero_grad()
        logits = model(torch.round(input_tensor).long())  # Ensure valid indices
        loss = -logits[0, target_class]  # Maximize the logit for the target class
        loss.backward()
        optimizer_gen.step()

        with torch.no_grad():
            input_tensor.clamp_(1, len(vocab))  # Keep indices within the valid vocab range

    generated_tokens = [list(vocab.keys())[int(idx) - 1] for idx in torch.round(input_tensor[0]).detach().numpy() if idx > 0]
    return " ".join(generated_tokens)


# Step 4: Load Data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_file_path = "./ir_data/train.csv"  # Replace with your train CSV file path
test_file_path = "./ir_data/test.csv"    # Replace with your test CSV file path

train_texts, train_labels, test_texts, test_labels, num_labels = load_csv_data(train_file_path, test_file_path)

# Tokenize texts
train_input_ids, train_attention_masks = tokenize_texts(train_texts, tokenizer, MAX_LEN)
test_input_ids, test_attention_masks = tokenize_texts(test_texts, tokenizer, MAX_LEN)

# Convert labels to tensors
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Step 5: Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 6: Initialize BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Step 7: Training Loop
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
        # predictions = torch.argmax(logits, dim=1)
        # correct_train += (torch.abs(predictions - labels) <= 3).sum().item()
        correct_train += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Train Accuracy (Tolerance 3): {train_accuracy * 100:.2f}%")

# Save the Model
model_save_path = "ir_classifier_bert.pth"  # Path to save the model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

train_results_file = "train_results.csv"
train_accuracy = evaluate_with_tolerance_and_save(model, test_loader, device, train_results_file, tolerance=3)
print(f"Train Accuracy (Tolerance 3): {train_accuracy * 100:.2f}%")

test_results_file = "test_results.csv"
test_accuracy = evaluate_with_tolerance_and_save(model, test_loader, device, test_results_file, tolerance=3)
print(f"Test Accuracy (Tolerance 3): {test_accuracy * 100:.2f}%")

