import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

opt_directory = "./resources"

os.makedirs(opt_directory, exist_ok=True)

# Define the neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# Gradient matching loss
def gradient_matching_loss(dummy_data, dummy_label, model, loss_fn, true_gradients):
    # Compute gradients for dummy data
    dummy_output = model(dummy_data)
    dummy_loss = loss_fn(dummy_output, dummy_label)
    dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
    
    # Compute the L2 norm between gradients (detach true_grad to avoid graph issues)
    loss = 0
    for dummy_grad, true_grad in zip(dummy_gradients, true_gradients):
        loss += ((dummy_grad - true_grad.detach()).norm())**2
    return loss

def binarize_image(data, threshold=0.5):
    """
    Convert pixel values of a tensor to binary (0 or 1) based on a threshold.

    Args:
        data (torch.Tensor): Input tensor with pixel values.
        threshold (float): Threshold value to binarize the data.

    Returns:
        torch.Tensor: Binarized tensor where values are either 0 or 1.
    """
    with torch.no_grad():  # Disable gradient calculations for binarization
        binary_data = torch.where(data >= threshold, torch.tensor(1.0, device=data.device), torch.tensor(0.0, device=data.device))
    return binary_data


def save_image(data, iteration, save_dir):
    plt.imshow(data.detach().cpu().squeeze(), cmap='gray')
    plt.title(f"Reconstructed Image at Iteration {iteration}")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"dlg_image_iter_{iteration}.png"))
    print(f"Saved image at iteration {iteration}")

# Main Function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize model and loss function
    model = SimpleNet().to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Select a data point
    data_iter = iter(train_loader)
    real_data, real_label = next(data_iter)
    real_data, real_label = real_data.to(device), real_label.to(device)

    # Compute true gradients for the real data
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    output = model(real_data)
    loss = loss_fn(output, real_label)
    true_gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Initialize dummy data and label
    dummy_data = torch.randn_like(real_data, requires_grad=True, device=device)
    dummy_label = torch.randint(0, 10, (1,), device=device)

    # Define LBFGS optimizer
    optimizer_dummy = optim.LBFGS([dummy_data], lr=0.1)

    # Reconstruct image using optimization
    num_iterations = 1000
    print("Starting reconstruction...")
    for i in range(num_iterations):
        def closure():
            optimizer_dummy.zero_grad()
            loss = gradient_matching_loss(dummy_data, dummy_label, model, loss_fn, true_gradients)
            loss.backward(retain_graph=True)  # retain_graph to handle repeated calls
            return loss

        optimizer_dummy.step(closure)

        if i % 100 == 0:
            save_image(dummy_data, i, opt_directory)
            print(f"Iteration {i}/{num_iterations}")


    output_dummy = model(dummy_data)

    dummy_data = binarize_image(dummy_data, 0.3)

    print(output_dummy)

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(real_data.detach().cpu().squeeze(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(dummy_data.detach().cpu().squeeze(), cmap='gray')
    plt.show()
    plt.savefig(os.path.join(opt_directory, "dlg_image_g.png"))

if __name__ == "__main__":
    main()
