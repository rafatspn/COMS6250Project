import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import random

MAX_LEN = 10
EMBED_DIM = 16
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 10


# Step 1: Load AG News Dataset
def load_ag_news_subset(split="train", samples_per_class=1000):
    dataset = load_dataset("ag_news", split=split)
    texts, labels = [], []

    # Organize samples per class
    class_samples = {i: [] for i in range(4)}
    for example in dataset:
        class_samples[example['label']].append(example['text'])

    # Sample examples per class
    for label, samples in class_samples.items():
        selected_samples = random.sample(samples, samples_per_class)
        texts.extend(selected_samples)
        labels.extend([label] * samples_per_class)
    
    return texts, labels

# Tokenization and Vocabulary Building
def build_vocab(texts):
    tokens = list(chain.from_iterable(word_tokenize(text.lower()) for text in texts))
    counter = Counter(tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.items())}  # idx 0 for padding
    vocab["<unk>"] = len(vocab) + 1
    return vocab

def text_to_indices(text, vocab, max_len):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    indices = indices[:max_len] + [0] * (max_len - len(indices))  # Pad to max_len
    return np.array(indices, dtype=np.int32)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [text_to_indices(text, vocab, max_len) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        embeds = self.embedding(x)  # Shape: [Batch, Time, Embed]
        x = embeds.mean(dim=1)      # Average embeddings
        x = self.fc(x)
        return x


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



# Accuracy Function
def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


# Main Execution
train_texts, train_labels = load_ag_news_subset(split="train", samples_per_class=10000)
test_texts, test_labels = load_ag_news_subset(split="test", samples_per_class=1000)

vocab = build_vocab(train_texts)

# DataLoaders
train_dataset = CustomDataset(train_texts, train_labels, vocab, MAX_LEN)
test_dataset = CustomDataset(test_texts, test_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
model = TextClassifier(len(vocab) + 1, EMBED_DIM, NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate Accuracy
train_accuracy = compute_accuracy(model, train_loader)
test_accuracy = compute_accuracy(model, test_loader)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate text for the positive class (1)
generated_text = generate_text(target_class=1, vocab=vocab, model=model)
print("Generated Text for Class 1 (Positive):", generated_text)
