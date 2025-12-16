# Neural Network module architecture

## Overview
The `my_torch` module provides a bespoke implementation of a feedforward neural network library, built on top of NumPy. It is designed to be modular, supporting dynamic network creation, training, and serialization.

## Main classes

### 1. `DenseLayer` (`my_torch.layers`)
The fundamental building block of the network. It represents a fully connected (dense) layer.
*   **Parameters**: Holds `weights` (shape `[out, in]`) and `bias` (shape `[out]`).
*   **Forward**: Computes $y = \sigma(xW^T + b)$, where $\sigma$ is the activation function. Cache input and pre-activation values for the backward pass.
*   **Backward**: Computes gradients with respect to weights, biases, and inputs using the chain rule.
*   **Updates**: Delegated to an optimizer via `apply_updates()`.

### 2. `NeuralNetwork` (`my_torch.neural_network`)
A container for a sequence of layers.
*   **Structure**: Stores a list of `TrainableLayer` objects (currently primarily `DenseLayer`).
*   **Forward/Backward**: Sequentially propagates inputs forward and gradients backward through the list of layers.
*   **Parameter Management**: Provides flattened views of all parameters and gradients for the optimizer.
*   **Construction**: Can be initialized directly with layer instances or from a configuration list (dictionaries).

### 3. `SGD` (`my_torch.optimizers`)
Implements Stochastic Gradient Descent.
*   **Logic**: Updates parameters using the rule $\theta = \theta - \eta \cdot \nabla_\theta J$, where $\eta$ is the learning rate.
*   **Features**: Supports L2 weight decay (regularization).
*   **API**:
    *   `update()`: Functional update of a single parameter array.
    *   `step()`: In-place update of a list of parameters given a list of gradients.

---

## Data flow

### Training flow (`my_torch.training.train`)
1.  **Batching**: Training data is split into mini-batches.
2.  **Forward pass**: 
    `Batch Input` $\rightarrow$ `network.forward()` $\rightarrow$ `Logits`.
3.  **Loss calculation**: 
    `Logits` + `Labels` $\rightarrow$ `Loss Function` $\rightarrow$ `Scalar Loss` & `Gradient of Loss w.r.t Logits`.
4.  **Backward pass**:
    `Gradient of Loss` $\rightarrow$ `network.backward()` $\rightarrow$ Gradients for all weights/biases accumulated in layers.
5.  **Optimization**:
    `network.parameters()` + `network.gradients()` $\rightarrow$ `optimizer.step()` $\rightarrow$ Update weights & biases.
6.  **Loop**: Use `network.zero_grad()` to reset gradients before the next batch.

### Prediction flow
1.  **Input**: Raw input features (e.g., FEN encoded vectors).
2.  **Forward pass**: `network.forward(input)` produces logits.
3.  **Interpretation**: Logits are converted to class labels (e.g., via `argmax`).

---

## Serialization (`my_torch.nn_io`)

The library uses a custom `.nn` format, which is a NumPy `.npz` archive.

### Format structure
1.  **`metadata_json`**: A JSON string containing:
    *   **Architecture**: A list of layer configuration dictionaries (e.g., input/output sizes, activation function names).
    *   **Training metadata**: Epochs, validation history, etc.
    *   **Extras**: User-defined metadata.
2.  **Parameter arrays**:
    *   `layer_{i}_weights`: Weight array for the $i$-th layer.
    *   `layer_{i}_bias`: Bias array for the $i$-th layer.

### Saving & loading
*   **`save_nn`**:
    *   Extracts configuration from layers.
    *   Maps activation functions to their string names (e.g., `relu` -> `"relu"`).
    *   Saves architecture JSON and parameter arrays into the `.npz` file.
*   **`load_nn`**:
    *   Reads `metadata_json` to reconstruct the layer objects.
    *   Maps string names back to functions using a registry (default + custom).
    *   Loads weight/bias arrays and assigns them to the reconstructed layers.

## Configuration

Networks can be defined using a list of configuration dictionaries, which is useful for procedural generation or external config files.

**Example config:**
```json
[
  {
    "type": "dense",
    "in_features": 128,
    "out_features": 64,
    "activation": "relu",
    "weight_initializer": "xavier"
  },
  {
    "type": "dense",
    "in_features": 64,
    "out_features": 3,
    "activation": "identity"
  }
]
```

This configuration is processed by `NeuralNetwork._build_layer_from_config` to instantiate the corresponding `DenseLayer` objects.
