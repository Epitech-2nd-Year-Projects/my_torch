from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))

from my_torch.activations import relu, relu_derivative
from my_torch.layers import (
    Conv2DLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    GlobalAvgPool2D,
)
from my_torch.losses import (
    cross_entropy_grad,
    cross_entropy_loss,
    softmax_cross_entropy_with_logits,
)
from my_torch.neural_network import NeuralNetwork
from my_torch.nn_io import save_network
from my_torch.optimizers import SGD, AdamW, SGDMomentum
from my_torch.training import (
    LossFn,
    LossGradFn,
    compute_class_weights,
    train,
    train_validation_split,
)
from my_torch_analyzer_pkg.dataset import load_dataset
from my_torch_analyzer_pkg.fen import mirror_tensor_lr
from my_torch_analyzer_pkg.labels import get_num_classes


class _CrossEntropyAdapter:
    def __init__(self, class_weights: np.ndarray | None) -> None:
        self._class_weights = class_weights
        self._cache_logits: np.ndarray | None = None
        self._cache_labels: np.ndarray | None = None
        self._cache_value: tuple[float, np.ndarray] | None = None

    def loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        loss, _ = self._compute(logits, labels)
        return loss

    def grad(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        _, grad = self._compute(logits, labels)
        return grad

    def _compute(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> tuple[float, np.ndarray]:
        if (
            self._cache_logits is logits
            and self._cache_labels is labels
            and self._cache_value is not None
        ):
            return self._cache_value
        loss, grad = softmax_cross_entropy_with_logits(
            logits,
            labels,
            class_weights=self._class_weights,
        )
        self._cache_logits = logits
        self._cache_labels = labels
        self._cache_value = (loss, grad)
        return loss, grad


def _apply_mirror_augmentation(
    inputs: np.ndarray,
    *,
    rng: np.random.Generator,
    mirror_prob: float,
) -> np.ndarray:
    if mirror_prob < 0.0 or mirror_prob > 1.0:
        raise ValueError("mirror_prob must be in the range [0, 1]")
    if inputs.ndim != 4:
        raise ValueError("mirror augmentation requires 4D inputs")
    if mirror_prob == 0.0:
        return inputs
    mask = rng.random(inputs.shape[0]) < mirror_prob
    if not np.any(mask):
        return inputs
    augmented = np.array(inputs, copy=True)
    for idx in np.flatnonzero(mask):
        augmented[idx] = mirror_tensor_lr(augmented[idx])
    return augmented


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


def _build_optimizer(args: argparse.Namespace) -> SGD | SGDMomentum | AdamW:
    if args.optimizer == "sgd":
        return SGD(lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd_momentum":
        return SGDMomentum(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return AdamW(lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def _format_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def _build_save_name(config_name: str, args: argparse.Namespace) -> str:
    parts = [
        config_name,
        args.optimizer,
        f"lr{_format_float(args.lr)}",
        f"e{args.epochs}",
        f"bs{args.batch_size}",
    ]
    if args.seed is not None:
        parts.append(f"seed{args.seed}")
    if args.augment_mirror:
        parts.append("aug")
    if args.class_weights:
        parts.append("cw")
    if args.stratify:
        parts.append("strat")
    return f"my_torch_network_{'_'.join(parts)}.nn"


def get_configs() -> dict[str, list[dict[str, Any]]]:
    input_size = 18 * 8 * 8
    output_size = get_num_classes()

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
        "cnn": [
            {
                "type": "conv2d",
                "in_channels": 18,
                "out_channels": 32,
                "kernel_size": 3,
                "padding": 1,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {
                "type": "conv2d",
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 3,
                "padding": 1,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {"type": "global_avg_pool2d"},
            {
                "type": "dense",
                "in_features": 64,
                "out_features": 64,
                "activation": relu,
                "activation_derivative": relu_derivative,
            },
            {"type": "dense", "in_features": 64, "out_features": output_size},
        ],
    }


def run_training(args: argparse.Namespace) -> None:
    print(f"Loading dataset from {args.dataset}...")
    try:
        inputs, labels = load_dataset(args.dataset)
    except FileNotFoundError:
        print(f"Error: Dataset file '{args.dataset}' not found.", file=sys.stderr)
        return
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return

    rng = np.random.default_rng(args.seed)
    num_classes = get_num_classes()
    train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
        inputs,
        labels,
        val_ratio=args.val_ratio,
        stratify=args.stratify,
        num_classes=num_classes,
        rng=rng,
    )

    if args.augment_mirror:
        train_inputs = _apply_mirror_augmentation(
            train_inputs, rng=rng, mirror_prob=args.mirror_prob
        )

    configs = get_configs()

    for name, layer_configs in configs.items():
        print(f"\n--- Training Configuration: {name.upper()} ---")

        network = NeuralNetwork(layer_configs=layer_configs)
        optimizer = _build_optimizer(args)

        if _should_flatten_inputs(network):
            train_features = train_inputs.reshape(train_inputs.shape[0], -1)
            val_features = val_inputs.reshape(val_inputs.shape[0], -1)
        else:
            train_features = train_inputs
            val_features = val_inputs

        class_weights = (
            compute_class_weights(train_labels, num_classes)
            if args.class_weights
            else None
        )
        loss_fn: LossFn
        loss_grad_fn: LossGradFn
        if class_weights is not None:
            loss_adapter = _CrossEntropyAdapter(class_weights=class_weights)
            loss_fn = cast(LossFn, loss_adapter.loss)
            loss_grad_fn = cast(LossGradFn, loss_adapter.grad)
        else:
            loss_fn = cross_entropy_loss
            loss_grad_fn = cross_entropy_grad

        history = train(
            network=network,
            optimizer=optimizer,
            train_inputs=train_features,
            train_labels=train_labels,
            val_inputs=val_features,
            val_labels=val_labels,
            loss_fn=loss_fn,
            loss_grad_fn=loss_grad_fn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=5,
            weight_decay=args.weight_decay,
            rng=rng,
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

        save_filename = _build_save_name(name, args)
        save_network(save_filename, network)
        print(f"  Saved model to {save_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiple network configurations on chess dataset."
    )
    parser.add_argument("dataset", help="Path to the chess dataset file (FEN + label)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "sgd_momentum", "adamw"],
        default="sgd",
    )
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--augment-mirror", action="store_true")
    parser.add_argument("--mirror-prob", type=float, default=0.5)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
