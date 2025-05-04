import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split # Import random_split
from PIL import Image
import os

# Import configuration for consistency
import config

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image or numpy.ndarray [0, 255] to FloatTensor [0.0, 1.0]
    transforms.Normalize(config.FASHION_MNIST_MEAN, config.FASHION_MNIST_STD) # Normalize
])

test_transform = transforms.Compose([ # Test and Validation use the same transform
    transforms.ToTensor(),
    transforms.Normalize(config.FASHION_MNIST_MEAN, config.FASHION_MNIST_STD)
])

predict_transform = transforms.Compose([
    transforms.Resize((28, 28)), # Ensure image is 28x28
    transforms.Grayscale(num_output_channels=1), # Ensure grayscale
    transforms.ToTensor(), # Converts to tensor and scales [0, 1]
    transforms.Normalize(config.FASHION_MNIST_MEAN, config.FASHION_MNIST_STD) # Same normalization
])

# --- Data Loading Function ---
def get_dataloaders(batch_size, data_dir=config.DATA_DIR, num_workers=config.NUM_WORKERS, val_split=0.2):
    """
    Loads the FashionMNIST dataset and creates training, validation, and testing DataLoaders.
    The validation set is split from the original training set.

    Args:
        batch_size (int): The number of samples per batch.
        data_dir (str): The directory where the FashionMNIST dataset is stored.
        num_workers (int): How many subprocesses to use for data loading.
        val_split (float): The fraction of the training data to use for validation (e.g., 0.2 for 20%).

    Returns:
        tuple: (train_loader, val_loader, test_loader) or (None, None, None) if dataset not found.
    """
    fashion_mnist_path = os.path.join(data_dir, "FashionMNIST")
    raw_folder_path = os.path.join(fashion_mnist_path, "raw")

    if not os.path.isdir(raw_folder_path):
        print(f"Error: 'FashionMNIST/raw' directory not found in '{os.path.abspath(data_dir)}'.")
        return None, None, None

    try:
        # Load the full original training dataset
        full_train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=False, # Explicitly disable download
            transform=train_transform # Apply training transforms initially
        )

        # Load the test dataset
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=False,
            transform=test_transform # Apply test transforms
        )

        # Calculate split sizes
        num_train = len(full_train_dataset)
        num_val = int(val_split * num_train)
        num_train_new = num_train - num_val

        if num_train_new <= 0 or num_val <= 0:
             raise ValueError(f"Invalid split sizes. Train: {num_train_new}, Validation: {num_val}")

        print(f"Splitting original training data ({num_train}) into New Train ({num_train_new}) and Validation ({num_val})...")

        # Split the original training dataset into new training and validation sets
        train_dataset, val_dataset = random_split(full_train_dataset, [num_train_new, num_val])

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, # Shuffle training data
            num_workers=num_workers,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=num_workers,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=num_workers,
            pin_memory=True if config.DEVICE == 'cuda' else False
        )

        print("Datasets loaded and split successfully.")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        return train_loader, val_loader, test_loader

    except RuntimeError as e:
         print(f"Error loading dataset via torchvision: {e}")
         return None, None, None
    except ValueError as e:
         print(f"Error during dataset split: {e}")
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading/splitting: {e}")
        return None, None, None


# --- Optional: Mean/Std Calculation ---
def calculate_fashion_mnist_mean_std(data_dir=config.DATA_DIR):
    """
    Calculates mean and std dev for the FashionMNIST training dataset.
    This is useful if you don't want to use pre-calculated values.

    Args:
        data_dir (str): The directory where the FashionMNIST dataset is stored.

    Returns:
        tuple: (mean, std) tensors or (None, None) if calculation fails.
    """
    # Check dataset path
    fashion_mnist_path = os.path.join(data_dir, "FashionMNIST")
    raw_folder_path = os.path.join(fashion_mnist_path, "raw")
    if not os.path.isdir(raw_folder_path):
        print(f"Error calculating mean/std: 'FashionMNIST/raw' directory not found in '{os.path.abspath(data_dir)}'.")
        return None, None

    # Use a basic ToTensor transform for calculation (scales to [0, 1])
    temp_transform = transforms.Compose([transforms.ToTensor()])
    try:
        # Load only the training set for calculation
        temp_train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=False,
            transform=temp_transform
            )
    except Exception as e:
        print(f"Could not load dataset for mean/std calculation: {e}")
        return None, None

    # Use a DataLoader to iterate efficiently over the entire dataset
    # Use a large batch size to speed up calculation
    # num_workers=0 might be more stable for this type of one-off calculation
    temp_loader = DataLoader(
        temp_train_dataset,
        batch_size=1024, # Process in large chunks
        shuffle=False,   # No need to shuffle
        num_workers=0    # Use main process
        )

    mean = 0.
    std = 0.
    total_images_count = 0
    print("\nCalculating dataset mean and std (this might take a moment)...")

    try:
        for images, _ in temp_loader:
            # images shape is [batch_size, 1, 28, 28] for FashionMNIST
            # Number of images in this batch (might be smaller for the last batch)
            batch_samples = images.size(0)
            # Reshape to [batch_size, num_channels, height*width]
            images = images.view(batch_samples, images.size(1), -1)
            # Calculate mean and std per image, then sum over batch dimension
            mean += images.mean([0, 2]).sum(0) # Sum means over the batch dimension
            std += images.std([0, 2]).sum(0) # Sum stds over the batch dimension
            total_images_count += batch_samples

        # Check if dataset was empty or loading failed
        if total_images_count == 0:
            print("Error: No images processed during mean/std calculation.")
            return None, None

        # Average the sums over the total number of images
        mean /= total_images_count
        std /= total_images_count

        print(f"Calculated Mean: {mean.item():.4f}") # .item() assuming single channel (grayscale)
        print(f"Calculated Std: {std.item():.4f}")
        print("You can update FASHION_MNIST_MEAN and FASHION_MNIST_STD in config.py with these values.")
        return mean, std
    except Exception as e:
        print(f"Error during mean/std calculation loop: {e}")
        return None, None

