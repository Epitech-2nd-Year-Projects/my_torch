from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from my_torch.layers import (
    Conv2DLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    GlobalAvgPool2D,
)
from my_torch.nn_io import load_network
from my_torch_analyzer.dataset import load_dataset


def _should_flatten_inputs(network: object) -> bool:
    layers = getattr(network, "layers", [])
    for layer in layers:
        if isinstance(layer, DropoutLayer):
            continue
        if isinstance(layer, DenseLayer):
            return True
        if isinstance(layer, (Conv2DLayer, FlattenLayer, GlobalAvgPool2D)):
            return False
        return True
    return True


def _prepare_inputs(network: object, inputs: np.ndarray) -> np.ndarray:
    if _should_flatten_inputs(network):
        if inputs.ndim != 2:
            return inputs.reshape(inputs.shape[0], -1)
        return inputs
    if inputs.ndim == 4:
        return inputs
    if inputs.ndim == 2 and inputs.shape[1] == 18 * 8 * 8:
        return inputs.reshape(inputs.shape[0], 18, 8, 8)
    raise ValueError("Model expects 4D inputs but dataset shape is incompatible")


def compute_accuracy(model_path: str, dataset_path: str) -> float:
    network = load_network(model_path)
    inputs, labels = load_dataset(dataset_path)
    inputs = _prepare_inputs(network, inputs)
    logits = network.forward(inputs)
    predictions = np.argmax(logits, axis=1)
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute accuracy of a trained .nn model on a labeled dataset."
    )
    parser.add_argument("model", help="Path to the .nn file to evaluate.")
    parser.add_argument(
        "dataset",
        help="Path to the dataset file (FEN + label per line) used for evaluation.",
    )
    args = parser.parse_args()

    try:
        accuracy = compute_accuracy(args.model, args.dataset)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 84
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error: {exc}", file=sys.stderr)
        return 84

    print(f"Accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
