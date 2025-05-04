import torch

# --- Files and Directories ---
DATA_DIR = "."  # Base directory where FashionMNIST folder is located
MODEL_SAVE_DIR = "." # Directory to save the trained model
MODEL_NAME = "fashion_mnist_model.pth" # Path to save/load the trained model
LOG_FILE = "log.txt" # Log file name

# --- Dataset Parameters ---
INPUT_SIZE = 28 * 28 # 784 pixels
NUM_CLASSES = 10
# Define the FashionMNIST classes (index matches label)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
# Pre-calculated or commonly used mean/std for FashionMNIST
# Calculating dynamically can be slow, so hardcoding is common.
# Ensure these match the values used in data_utils.py if calculated there.
FASHION_MNIST_MEAN = torch.tensor([0.2860])
FASHION_MNIST_STD = torch.tensor([0.3530])


# --- Training Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 10 # Adjust as needed for good performance
LEARNING_RATE = 0.001

# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Other ---
NUM_WORKERS = 2 # Number of workers for DataLoader

