import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain

# Step 1: Data Preparation
train_texts = ["This is a great product", "Terrible experience", "I loved it", "Hated it", "It was fantastic"]
train_labels = [1, 0, 1, 0, 1]  # 1: Positive, 0: Negative
MAX_LEN = 10
EMBED_DIM = 16
NUM_CLASSES = 2

# Tokenization and Vocabulary Building
def build_vocab(texts):
    tokens = list(chain.from_iterable(word_tokenize(text.lower()) for text in texts))
    counter = Counter(tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.items())}  # idx 0 for padding
    vocab["<unk>"] = len(vocab) + 1
    return vocab

vocab = build_vocab(train_texts)

def text_to_indices(text, vocab, max_len):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    indices = indices[:max_len] + [0] * (max_len - len(indices))  # Pad to max_len
    return np.array(indices, dtype=np.int32)

# Convert Data to NumPy Arrays
train_data = np.array([text_to_indices(text, vocab, MAX_LEN) for text in train_texts])
train_labels = np.array(train_labels, dtype=np.int32)

# Step 2: PyTorch Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Convert numpy input to torch tensor
        x = torch.as_tensor(x, dtype=torch.long).clone().detach()
        embeds = self.embedding(x)  # [Batch, Time, Embedding]
        x = embeds.mean(dim=1)  # Mean embeddings across sequence
        x = self.fc(x)
        return x

model = TextClassifier(len(vocab) + 1, EMBED_DIM, NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Step 3: Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(len(train_data)):
        inputs = train_data[i].reshape(1, -1)  # NumPy input
        labels = torch.tensor([train_labels[i]], dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")

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

# Generate text for the positive class (1)
generated_text = generate_text(target_class=1, vocab=vocab, model=model)
print("Generated Text for Class 1 (Positive):", generated_text)
