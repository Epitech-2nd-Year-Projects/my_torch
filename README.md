# my_torch

## Project overview

**my_torch** is a custom implementation of a Neural Network library and a specific Chess Analyzer tool.

The project is divided into two main components:
1.  **NN Module (`my_torch`)**: A generic library providing tools to generate, train, save, load, and analyze neural networks. It is built from scratch using NumPy, without relying on high-level frameworks like PyTorch or TensorFlow.
2.  **Chess Analyzer (`my_torch_analyzer`)**: An executable tool that uses the NN module to evaluate chessboard states (FEN strings) and determine if the state is "Check", "Checkmate", or "Nothing".

## Goal and architecture

The primary goal is to demonstrate a machine-learning-based solution trained with supervised learning.

-   **High-level architecture**:
    -   **`my_torch/`**: Contains the core definition of Layers, Neurons, Activation Functions, Loss Functions, and the Network engine itself.
    -   **`my_torch_analyzer/`**: The application layer that parses command-line arguments, loads datasets, and orchestrates the training or prediction process using the `my_torch` library.

## Installation

### Prerequisites
-   **Python**: Version 3.10 or higher.
-   **NumPy**: Version 1.24 or higher.

### Steps
1.  Clone the repository:
    ```bash
    git clone https://github.com/Epitech-2nd-Year-Projects/my_torch.git
    cd my_torch
    ```

2.  Install the project in **editable mode** (for development):
    ```bash
    pip install -e .
    ```
    
    To install with development dependencies (pytest, mypy, ruff, black):
    ```bash
    pip install -e ".[dev]"
    ```

## Pre-trained model

A pre-trained neural network is provided at the root of the repository:
-   **`my_torch_network.nn`**: Our best-performing CNN model, ready for evaluation.

## Usage: `my_torch_analyzer`

The `my_torch_analyzer` command is the main entry point for the project. It supports two modes: `--train` and `--predict`.

### Command syntax
```bash
my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

-   `LOADFILE`: Path to the neural network file (e.g., `my_torch_network.nn`).
-   `CHESSFILE`: Path to the dataset file containing FEN strings (and labels for training).
-   `--save SAVEFILE`: (Optional, Train mode only) Path to save the trained network.

### 1. Training mode (`--train`)
Example: Train an existing network `my_torch_network.nn` using `dataset_train.txt` and save the improved model to `new_network.nn`.

```bash
my_torch_analyzer --train --save new_network.nn my_torch_network.nn dataset_train.txt
```

*Note: In training mode, the `CHESSFILE` must contain FEN strings followed by the expected output (e.g., "Checkmate").*

### 2. Prediction mode (`--predict`)
Example: Predict the state of chessboards in `chessboards.txt` using the trained network `my_torch_network.nn`.

```bash
my_torch_analyzer --predict my_torch_network.nn chessboards.txt
```

**Output format**:
The program prints one prediction per line corresponding to the input chessboards:
```
Check
Nothing
Checkmate
Check
```

## Documentation and benchmarks

-   **Benchmarks**: detailed benchmark comparison between MLP and CNN architectures, justifying our design choices, can be found in [`docs/benchmarks.md`](docs/benchmarks.md).
-   **NN Format**: Description of the neural network file format is available in [`docs/nn_format.md`](docs/nn_format.md).
-   **Contributing**: See [`docs/contributing.md`](docs/contributing.md).