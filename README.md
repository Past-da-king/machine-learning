
# CSC3022F Assignment 1 - ANN for FashionMNIST

A PyTorch feedforward neural network to classify FashionMNIST images. Includes training, evaluation, and interactive JPEG prediction.

## Key Files

*   `main.py`: Main script to run the application.
*   `config.py`: Hyperparameters and settings.
*   `model.py`: Neural Network class (`SimpleANN`).
*   `data_utils.py`: Data loading and transforms.
*   `engine.py`: Training/evaluation functions.
*   `predict.py`: Single image prediction function.
*   `requirements.txt`: Python dependencies.
*   `Makefile`: Helper commands (`run`, `install`, `clean`).
*   `log.txt`: (Generated) Training/evaluation log.
*   `fashion_mnist_model.pth`: (Generated) Saved trained model.

## Setup

1.  **Prerequisites:** Python 3.x, Pip, Git.
2.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    # OR
    make install
    ```
3.  **Get Dataset (MANUAL):**
    *   Download FashionMNIST `.gz` files (from Amathuba/assignment source).
    *   Create folders: `./FashionMNIST/raw/`
    *   Place downloaded `.gz` files **inside** `./FashionMNIST/raw/`.
    *   *The code will NOT download the data.*
4.  **Git:** Initialize and use Git for version control (`git init`, `git add`, `git commit`).

## Running

Navigate to the project directory in your terminal:

```bash
python main.py
# OR
make run
```

## Expected Output

1.  Trains model (if no `.pth` file found) or loads existing model.
2.  Prints epoch loss during training.
3.  Evaluates on test set, prints accuracy (results also in `log.txt`).
4.  Enters interactive mode:
    ```
    Please enter a filepath: ./path/to/your_image.jpg
    Classifier: [Predicted Class]
    ```
5.  Type `exit` to quit.

## Cleaning Up

```bash
make clean
```
(Removes Python cache files).
