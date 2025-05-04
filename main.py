import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Import modules from the project
import config
from model import SimpleANN
from data_utils import get_dataloaders, predict_transform # predict_transform still needed
from engine import train_model, evaluate_model # evaluate_model now handles both validation and test
from predict import predict_image

# --- Main Execution ---
if __name__ == "__main__":

    # --- Setup ---
    if os.path.exists(config.LOG_FILE):
        try:
            os.remove(config.LOG_FILE)
            print(f"Cleared previous log file: {config.LOG_FILE}")
        except OSError as e:
            print(f"Warning: Could not remove previous log file '{config.LOG_FILE}': {e}")

    device = config.DEVICE
    print(f"Using device: {device}")

    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME)
    if config.MODEL_SAVE_DIR and not os.path.exists(config.MODEL_SAVE_DIR):
        os.makedirs(config.MODEL_SAVE_DIR)
        print(f"Created model save directory: {config.MODEL_SAVE_DIR}")

    # --- Data Loading (Now gets 3 loaders) ---
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        data_dir=config.DATA_DIR,
        num_workers=config.NUM_WORKERS,
        val_split=0.2 # Example: 20% validation split
    )

    # Exit if data loading failed
    if train_loader is None or val_loader is None or test_loader is None:
        print("Failed to load or split datasets. Exiting.")
        sys.exit(1)

    # --- Model Initialization ---
    model = SimpleANN(input_size=config.INPUT_SIZE, num_classes=config.NUM_CLASSES).to(device)
    print(f"Model initialized: {type(model).__name__}")

    # --- Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- Training ---

    print(f"\nStarting training for {config.EPOCHS} epochs...")
    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader, # Pass validation loader
        device=device,
        epochs=config.EPOCHS,
        log_file=config.LOG_FILE
    )

    # --- Final Evaluation on Test Set ---
    # Load the *best* model saved during training (based on validation accuracy)
    print(f"\nLoading best model saved at {model_path} for final test evaluation...")
    try:
        # Ensure we load the state dict saved based on validation performance
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Best model loaded successfully.")
            # Perform final evaluation on the *test* set
            evaluate_model(
                model=model,
                criterion=criterion,
                data_loader=test_loader, # Use TEST loader here
                device=device,
                log_file=config.LOG_FILE,
                phase="Test" # Explicitly label as Test phase
            )
        else:
             print(f"Warning: Model file '{model_path}' not found after training. Could not perform final test evaluation.")

    except Exception as e:
        print(f"Error loading best model or during final test evaluation: {e}")


    print("\n--- Setup Complete ---")

    # --- Interactive Prediction Loop (Remains the same) ---
    print("\nEnter image filepath for classification or 'exit' to quit.")
    while True:
        try:
            filepath = input("Please enter a filepath: ")
            if filepath.lower().strip() == 'exit':
                print("Exiting...")
                break

            # Ensure the model is loaded (it should be from the step above)
            if not os.path.exists(model_path) and 'model' not in locals():
                 print("Error: Model not available for prediction.")
                 continue # Or exit

            # Predict the class of the image
            predicted_class, confidence = predict_image(
                model=model,
                image_path=filepath,
                transform=predict_transform,
                class_names=config.CLASS_NAMES,
                device=device
            )

            if predicted_class is not None:
                 print(f"Classifier: {predicted_class}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the prediction loop: {e}")


