import torch
import torch.nn.functional as F
from PIL import Image
import os

def predict_image(model, image_path, transform, class_names, device):
    """
    Loads, preprocesses, and classifies a single JPEG image.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        image_path (str): Path to the JPEG image file.
        transform (callable): The torchvision transform to apply to the image.
        class_names (list): A list of class names corresponding to model output indices.
        device (torch.device): The device (CPU or CUDA) to run inference on.

    Returns:
        tuple: (predicted_class_name, confidence) or (None, None) if an error occurs.
    """
    try:
        # Check if file exists before opening
        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'")
            return None, None

        # Load image using PIL for robustness with various JPEG formats
        img = Image.open(image_path)

        img_tensor = transform(img).unsqueeze(0) # Add batch dimension (B x C x H x W)
        img_tensor = img_tensor.to(device)

        model.eval() # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model(img_tensor)
            # Apply Softmax to get probabilities
            probabilities = F.softmax(outputs, dim=1)
            # Get the class with the highest probability
            confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = class_names[predicted_idx.item()]
            confidence_percent = confidence.item() * 100

            # print(f"Raw outputs: {outputs}") # Debug
            # print(f"Probabilities: {probabilities}") # Debug
            return predicted_class, confidence_percent

    except FileNotFoundError: # Double check, though os.path.exists should catch it
        print(f"Error: File not found at '{image_path}'")
        return None, None
    except Exception as e:
        # Catch potential errors during image opening, transformation, or prediction
        print(f"Error processing image '{image_path}': {e}")
        return None, None

