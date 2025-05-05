import torch

DATA_DIR = "."  
MODEL_SAVE_DIR = "." 
MODEL_NAME = "fashion_mnist_model.pth" 
LOG_FILE = "log.txt" 


INPUT_SIZE = 28 * 28 
NUM_CLASSES = 10

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

FASHION_MNIST_MEAN = torch.tensor([0.2860])
FASHION_MNIST_STD = torch.tensor([0.3530])



BATCH_SIZE = 64
EPOCHS = 10 
LEARNING_RATE = 0.001

# --- Hardware ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Other ---NUM_WORKERS = 2 # Number of workers for DataLoader

