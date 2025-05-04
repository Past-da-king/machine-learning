
import torch
import os

import config

# --- Evaluation/Validation Function (Generalized) ---
def evaluate_model(model, criterion, data_loader, device, log_file, phase="Test"):
    """Evaluates the model on a given dataset (validation or test)."""
    print(f"\n--- Evaluating Model ({phase}) ---")
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():  # No need to track gradients
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f'{phase} Loss: {avg_loss:.4f} | {phase} Accuracy: {accuracy:.2f}%')

        # Append results to the log file
        with open(log_file, 'a') as f:
            f.write(f"\n--- {phase} Results ---\n")
            f.write(f"{phase} Accuracy: {accuracy:.2f}%\n")
            f.write(f"{phase} Loss: {avg_loss:.4f}\n")
        print(f"--- {phase} Evaluation Complete ---")
        return accuracy, avg_loss # Return performance metrics

    except IOError as e:
        print(f"Error writing {phase} results to log file '{log_file}': {e}")
        return 0.0, float('inf') # Indicate failure
    except Exception as e:
        print(f"An unexpected error occurred during {phase} evaluation: {e}")
        return 0.0, float('inf') # Indicate failure


# --- Training Function (Modified to include validation) ---
def train_model(model, criterion, optimizer, train_loader, val_loader, device, epochs, log_file):
    """Trains the neural network model, performing validation after each epoch."""
    print("\n--- Starting Training (with Validation) ---")
    # Ensure log file directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    best_val_accuracy = 0.0 # Keep track of best validation accuracy

    try:
        with open(log_file, 'a') as f: # Append to log file
            f.write("\n--- Training Log ---\n")
            f.write(f"Epochs: {epochs}, Batch Size: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}, Val Split: {0.2}\n") # Log hyperparams

            for epoch in range(epochs):
                model.train() # Set model to training mode for the epoch
                running_loss = 0.0
                # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} Training") # Optional TQDM
                # for i, (inputs, labels) in progress_bar:
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # if isinstance(progress_bar, tqdm): progress_bar.set_postfix(loss=loss.item()) # Optional TQDM

                epoch_loss = running_loss / len(train_loader)
                print(f"\nEpoch [{epoch + 1}/{epochs}] Training Loss: {epoch_loss:.4f}")
                f.write(f"\nEpoch [{epoch + 1}/{epochs}] Training Loss: {epoch_loss:.4f}\n")

                # --- Validation Step ---
                val_accuracy, val_loss = evaluate_model(model, criterion, val_loader, device, log_file, phase="Validation")
                # Log validation results per epoch
                f.write(f"Epoch [{epoch + 1}/{epochs}] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%\n")

                # --- Optional: Save best model based on validation accuracy ---
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    # Construct model path inside the loop if saving best model
                    model_path = os.path.join(config.MODEL_SAVE_DIR, config.MODEL_NAME) # Use configured path
                    try:
                         torch.save(model.state_dict(), model_path)
                         print(f"Epoch [{epoch + 1}/{epochs}]: New best model saved to {model_path} with Validation Accuracy: {best_val_accuracy:.2f}%")
                         f.write(f"Epoch [{epoch + 1}/{epochs}]: Saved new best model.\n")
                    except Exception as e_save:
                         print(f"Error saving best model during epoch {epoch + 1}: {e_save}")
                         f.write(f"Epoch [{epoch + 1}/{epochs}]: Error saving best model: {e_save}\n")
                # -------------------------------------------------------------

        print("\n--- Finished Training ---")

    except IOError as e:
        print(f"Error writing to log file '{log_file}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")


