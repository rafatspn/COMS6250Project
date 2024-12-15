import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Dataset: Text and Class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Dummy Data (Replace with Real Dataset)
texts = ["The mining disaster caused serious damage.", "Safety measures in underground mines are critical.",
         "A rockburst incident occurred, leading to collapse."]
labels = [0, 1, 0]  # Binary classification for demonstration

# Load Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create Dataset and DataLoader
dataset = TextClassificationDataset(texts, labels, tokenizer)
data_loader = DataLoader(dataset, batch_size=1)

# Train the Model and Save Gradients
optimizer = optim.Adam(model.parameters(), lr=5e-5)
model.train()
real_gradients = None
for batch in data_loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    if real_gradients is None:
        real_gradients = [torch.zeros_like(param.grad) for param in model.parameters() if param.grad is not None]
    for idx, param in enumerate(model.parameters()):
        if param.grad is not None:
            real_gradients[idx] += param.grad.clone()
    optimizer.step()

# Finalize the averaged gradients for attack
real_gradients = [grad / len(data_loader) for grad in real_gradients]

# DLG Reconstruction: Matching Gradients
class DeepLeakageAttack:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer

    def reconstruct(self, real_gradients, lr=0.01, max_iter=300):
        """
        real_gradients: Gradients obtained from real data
        lr: Learning rate for optimization
        max_iter: Maximum iterations for reconstruction
        """
        dummy_data = torch.randint(0, self.model.config.vocab_size, (1, 128), dtype=torch.long, device=self.device, requires_grad=False)
        dummy_data = dummy_data.clone().detach().requires_grad_(True)
        dummy_labels = torch.randint(0, 2, (1,), device=self.device, requires_grad=True)
        optimizer = optim.SGD([dummy_data, dummy_labels], lr=lr)

        for i in range(max_iter):
            optimizer.zero_grad()
            outputs = self.model(input_ids=dummy_data, labels=dummy_labels)
            dummy_gradients = torch.autograd.grad(outputs.loss, self.model.parameters(), create_graph=True)

            # Compute Gradient Distance
            grad_loss = 0
            for g_real, g_dummy in zip(real_gradients, dummy_gradients):
                grad_loss += ((g_real - g_dummy).pow(2)).sum()
            grad_loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Iteration {i}: Gradient Loss = {grad_loss.item():.4f}")

        return dummy_data, dummy_labels

# Reconstruct Text
attack = DeepLeakageAttack(model, tokenizer)
dummy_data, dummy_labels = attack.reconstruct(real_gradients)

# Decode the Reconstructed Text
reconstructed_text = tokenizer.decode(dummy_data[0].detach().cpu().numpy(), skip_special_tokens=True)
print("\nReconstructed Text:", reconstructed_text)
print("Reconstructed Label:", dummy_labels.item())
