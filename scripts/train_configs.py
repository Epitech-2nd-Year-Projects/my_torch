import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parents[1]))

from my_torch.activations import relu, relu_derivative
from my_torch.losses import cross_entropy_grad, cross_entropy_loss
from my_torch.neural_network import NeuralNetwork
from my_torch.nn_io import save_network
from my_torch.optimizers import SGD
from my_torch.training import train, train_validation_split
from my_torch_analyzer.dataset import load_dataset


def get_configs() -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns a dictionary of network configurations.
    Input size is 18*8*8 = 1152.
    Output size is 3 (Nothing, Check, Checkmate).
    """
    input_size = 1152
    output_size = 3

    return {
        "small": [
            {
                "type": "dense",
                "in_features": input_size,
                "out_features": 64,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {"type": "dense", "in_features": 64, "out_features": output_size},
        ],
        "medium": [
            {
                "type": "dense",
                "in_features": input_size,
                "out_features": 128,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {
                "type": "dense",
                "in_features": 128,
                "out_features": 64,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {"type": "dense", "in_features": 64, "out_features": output_size},
        ],
        "large": [
            {
                "type": "dense",
                "in_features": input_size,
                "out_features": 256,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {
                "type": "dense",
                "in_features": 256,
                "out_features": 128,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {
                "type": "dense",
                "in_features": 128,
                "out_features": 64,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {"type": "dense", "in_features": 64, "out_features": output_size},
        ],
    }


def run_training(dataset_path: str) -> None:
    print(f"Loading dataset from {dataset_path}...")
    try:
        inputs, labels = load_dataset(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.", file=sys.stderr)
        return
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return

    inputs = inputs.reshape(inputs.shape[0], -1)

    train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
        inputs, labels, val_ratio=0.2, seed=42
    )

    configs = get_configs()

    for name, layer_configs in configs.items():
        print(f"\n--- Training Configuration: {name.upper()} ---")

        network = NeuralNetwork(layer_configs=layer_configs)

        optimizer = SGD(lr=0.01, weight_decay=0.001)

        history = train(
            network=network,
            optimizer=optimizer,
            train_inputs=train_inputs,
            train_labels=train_labels,
            val_inputs=val_inputs,
            val_labels=val_labels,
            loss_fn=cross_entropy_loss,
            loss_grad_fn=cross_entropy_grad,
            epochs=20,
            batch_size=32,
            early_stopping_patience=5,
            weight_decay=0.001,
        )

        final_train = history.train[-1] if history.train else None
        final_val = history.validation[-1] if history.validation else None

        if final_train:
            print(
                f"  Final Train Loss: {final_train.loss:.4f}, "
                f"Accuracy: {final_train.accuracy:.4f}"
            )
        if final_val:
            print(
                f"  Final Val Loss:   {final_val.loss:.4f}, "
                f"Accuracy: {final_val.accuracy:.4f}"
            )

        if history.best_parameters:
            for param, best_param in zip(network.parameters(), history.best_parameters):
                param[...] = best_param

        save_filename = f"my_torch_network_{name}.nn"
        save_network(save_filename, network)
        print(f"  Saved model to {save_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiple network configurations on chess dataset."
    )
    parser.add_argument("dataset", help="Path to the chess dataset file (FEN + label)")
    args = parser.parse_args()

    run_training(args.dataset)


if __name__ == "__main__":
    main()
