import torch
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

# Step 1: Load AG News Dataset (1000 samples per class)
def load_ag_news_subset(samples_per_class=1000, seed=42):
    dataset = load_dataset("ag_news", split="train")
    texts, labels = [], []
    class_samples = {i: [] for i in range(4)}

    for example in dataset:
        class_samples[example['label']].append(example['text'])

    random.seed(seed)
    for label, samples in class_samples.items():
        selected_samples = random.sample(samples, samples_per_class)
        texts.extend(selected_samples)
        labels.extend([label] * samples_per_class)

    return texts, labels

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

# Step 3: Load Data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Use EOS token as padding token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id  # Set pad_token_id for the model
model = model.to(device)

train_texts, train_labels = load_ag_news_subset(samples_per_class=1000)
test_texts, test_labels = load_ag_news_subset(samples_per_class=1000, seed=999)

train_input_ids = tokenize_texts(train_texts, tokenizer, MAX_LEN)
test_input_ids = tokenize_texts(test_texts, tokenizer, MAX_LEN)

train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Step 4: Create DataLoaders
train_dataset = TensorDataset(train_input_ids, train_labels)
test_dataset = TensorDataset(test_input_ids, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Step 5: Define GPT-based Classifier
class GPTClassifier(nn.Module):
    def __init__(self, model):
        super(GPTClassifier, self).__init__()
        self.gpt = model
        self.fc = nn.Linear(self.gpt.config.n_embd, 4)  # Output 4 classes

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

# Step 6: Training Loop
print("Training...")
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

# Step 7: Evaluation
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

test_accuracy = evaluate(classifier, test_loader)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

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

# Example: Generate text for a specific class label
target_label = 1  # Sports
generated_text = generate_text_for_label(target_label, tokenizer, classifier)
print(f"Generated Text for Class {target_label} (Sports): {generated_text}")
