import torch
import torch.nn as nn

# --- Neural Network Definition ---
class SimpleANN(nn.Module):
    """A simple feedforward neural network for FashionMNIST."""
    def __init__(self, input_size, num_classes):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, 128) # Input layer to Hidden layer 1
        self.relu1 = nn.ReLU()                # Activation function
        self.fc2 = nn.Linear(128, 64)         # Hidden layer 1 to Hidden layer 2
        self.relu2 = nn.ReLU()                # Activation function
        self.fc3 = nn.Linear(64, num_classes) # Hidden layer 2 to Output layer

    def forward(self, x):
        x = self.flatten(x) # Flatten image
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x) # Output raw logits
        return x

