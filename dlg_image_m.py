import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to compute the gradient of the model
def get_gradients(model, criterion, x, y):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.clone())
    return grads

# DLG algorithm to reconstruct the image
def deep_leakage_from_gradients(model, criterion, gradients, dummy_x, dummy_y, lr=0.1, iterations=1000):
    optimizer = optim.LBFGS([dummy_x, dummy_y], lr=lr)

    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            dummy_output = model(dummy_x)
            dummy_loss = criterion(dummy_output, dummy_y)
            dummy_loss.backward()
            dummy_grads = get_gradients(model, criterion, dummy_x, dummy_y)
            grad_diff = 0
            for g1, g2 in zip(dummy_grads, gradients):
                grad_diff += torch.sum((g1 - g2) ** 2)
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        if i % 100 == 0:
            print(f'Iteration {i}/{iterations}, Loss: {closure().item()}')

    return dummy_x.detach(), dummy_y.detach()

# Function to save images
def save_image(image, filename):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize the model, criterion, and optimizer
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()

    # Create a dummy input and label
    dummy_x = torch.randn(1, 1, 28, 28, requires_grad=True)
    dummy_y = torch.randn(1, 10, requires_grad=True)  # One-hot encoding for the label

    # Assume we have the gradients from the original input and label
    original_x = torch.randn(1, 1, 28, 28)
    original_y = torch.randint(0, 10, (1,))
    original_y_one_hot = torch.zeros(1, 10)
    original_y_one_hot[0, original_y] = 1
    original_gradients = get_gradients(model, criterion, original_x, original_y_one_hot)

    # Save the original image
    save_image(original_x, 'original_image_m.png')

    # Reconstruct the image using DLG
    reconstructed_x, reconstructed_y = deep_leakage_from_gradients(model, criterion, original_gradients, dummy_x, dummy_y)

    # Save the reconstructed image
    save_image(reconstructed_x, 'reconstructed_image_m.png')

    # Print the reconstructed image and label
    print("Reconstructed Label:", torch.argmax(reconstructed_y).item())
    print("Reconstructed Image:", reconstructed_x.squeeze().detach().numpy())
