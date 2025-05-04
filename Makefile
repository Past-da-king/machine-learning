# Makefile for CSC3022F Assignment 1 - ANN FashionMNIST Classifier

# Variables
PYTHON = python3 # Use python3 or python depending on your system setup
MAIN_SCRIPT = main.py
REQUIREMENTS_FILE = requirements.txt
PIP = pip

# Phony targets are not files
.PHONY: all run install clean help

# Default target executed when you just run 'make'
all: run

# Target to run the main application script
run:
	@echo "--- Running the FashionMNIST Classifier ---"
	$(PYTHON) $(MAIN_SCRIPT)
	@echo "--- Script execution finished ---"

# Target to install dependencies from requirements.txt
# Ensure you have a requirements.txt file for this to work
install: $(REQUIREMENTS_FILE)
	@echo "--- Installing dependencies from $(REQUIREMENTS_FILE) ---"
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "--- Dependencies installed ---"

# Target to clean up generated files (Python cache, etc.)
# Add other files to clean if necessary (e.g., *.log, *.pth if desired)
clean:
	@echo "--- Cleaning up generated files ---"
	@# Remove Python cache directories and files
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@find . -type f -name "*.py[co]" -delete
	@# Optionally remove the log file (uncomment if needed)
	# @rm -f log.txt
	@# Optionally remove the saved model file (uncomment if needed)
	# @rm -f fashion_mnist_model.pth
	@echo "--- Cleanup complete ---"

# Target to display help information about the Makefile commands
help:
	@echo "Available Makefile commands:"
	@echo "  make all       : Run the main script (default)."
	@echo "  make run       : Run the main script ($(MAIN_SCRIPT))."
	@echo "  make install   : Install dependencies from $(REQUIREMENTS_FILE)."
	@echo "  make clean     : Remove generated Python cache files."
	@echo "  make help      : Show this help message."

# Indicate that requirements.txt is a prerequisite for install,
# but don't provide a rule to build it automatically here.
# The user should create requirements.txt.
$(REQUIREMENTS_FILE):
	@# This target is just a prerequisite placeholder.
	@# Ensure requirements.txt exists before running 'make install'.