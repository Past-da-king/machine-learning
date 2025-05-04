
import torch
import os
import sys

import config
from model import SimpleANN
from data_utils import predict_transform 
from predict import predict_image      

TEST_IMAGES = {
    "fashion-jpegs/bag.jpg": "Bag",
    "fashion-jpegs/dress1.jpg": "Dress",
    "fashion-jpegs/sneaker1.jpg": "Sneaker",
    "fashion-jpegs/sneaker2.jpg": "Sneaker",
    "fashion-jpegs/dress2.jpg": "Dress",
    "fashion-jpegs/trouser.jpg": "Trouser"
}


def load_trained_model(model_path, device):
    """Loads the trained SimpleANN model."""
    if not os.path.exists(model_path):
        print(f"Error: Trained model file not found at '{model_path}'")
        print("Please train the model first by running main.py.")
        return None

    try:

        model = SimpleANN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES)

        model.load_state_dict(torch.load(model_path, map_location=device))

        model.eval()
        model.to(device)
        print(f"Trained model loaded successfully from '{model_path}'.")
        return model
    except Exception as e:
        print(f"Error loading model state from '{model_path}': {e}")
        return None


if __name__ == "__main__":
    print("--- Running Predictor Test Script ---")

    # Basic checks
    if not os.path.isdir("fashion-jpegs"):
        print("Error: Directory 'fashion-jpegs' not found.")
        print("Please unzip 'fashion-jpegs.zip' in the current directory.")
        sys.exit(1)


    device = config.DEVICE
    print(f"Using device: {device}")


    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)

   
    model = load_trained_model(model_path, device)
    if model is None:
        sys.exit(1) # Exit if model loading failed


    correct_predictions = 0
    total_images = len(TEST_IMAGES)

    print("\n--- Testing Predictions ---")
    for img_path, expected_label in TEST_IMAGES.items():
        print(f"Testing image: '{img_path}' (Expected: {expected_label})")

        if not os.path.exists(img_path):
             print(f"  -> Error: Image file not found at '{img_path}'. Skipping.")
             total_images -= 1 # Adjust total if file is missing
             continue


        predicted_class, confidence = predict_image(
            model=model,
            image_path=img_path,
            transform=predict_transform, # Use the correct transform
            class_names=config.CLASS_NAMES,
            device=device
        )

        if predicted_class is not None:
            is_correct = (predicted_class == expected_label)
            if is_correct:
                correct_predictions += 1
                print(f"  -> Prediction: {predicted_class} (Confidence: {confidence:.2f}%) - CORRECT")
            else:
                print(f"  -> Prediction: {predicted_class} (Confidence: {confidence:.2f}%) - INCORRECT (Expected: {expected_label})")
        else:
            print(f"  -> Prediction failed for this image.")


    print("\n--- Test Summary ---")
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"{correct_predictions} out of {total_images} images correctly classified.")
        print(f"Accuracy on this test set: {accuracy:.2f}%")
    else:
        print("No test images were successfully processed.")

    print("\n--- Test Script Finished ---")