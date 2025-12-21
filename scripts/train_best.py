from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from my_torch.activations import relu, relu_derivative
from my_torch.layers import Conv2DLayer, DenseLayer, DropoutLayer, GlobalAvgPool2D
from my_torch.losses import softmax_cross_entropy_with_logits
from my_torch.neural_network import NeuralNetwork
from my_torch.nn_io import load_network, save_network
from my_torch.optimizers import AdamW
from my_torch.training import compute_class_weights, train_validation_split
from my_torch_analyzer.dataset import load_dataset
from my_torch_analyzer.fen import mirror_tensor_lr
from my_torch_analyzer.labels import get_label_from_index, get_num_classes

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]


@dataclass(slots=True)
class Metrics:

    loss: float
    accuracy: float


def _apply_mirror_augmentation(
    inputs: ArrayFloat,
    *,
    rng: np.random.Generator,
    mirror_prob: float,
) -> ArrayFloat:
    if mirror_prob < 0.0 or mirror_prob > 1.0:
        raise ValueError("mirror_prob must be in the range [0, 1]")
    if inputs.ndim != 4 or inputs.shape[1:] != (18, 8, 8):
        raise ValueError("mirror augmentation requires inputs of shape (N, 18, 8, 8)")
    if mirror_prob == 0.0:
        return inputs
    mask = rng.random(inputs.shape[0]) < mirror_prob
    if not np.any(mask):
        return inputs
    augmented = np.array(inputs, copy=True)
    for idx in np.flatnonzero(mask):
        augmented[idx] = mirror_tensor_lr(augmented[idx])
    return augmented


def _batch_accuracy(logits: ArrayFloat, labels: ArrayInt) -> float:
    predictions = np.argmax(logits, axis=1)
    if labels.size == 0:
        return 0.0
    return float(np.mean(predictions == labels))


def _compute_l2_loss(network: NeuralNetwork, weight_decay: float) -> float:
    if weight_decay <= 0.0:
        return 0.0
    l2_sum = sum(
        float(np.sum(np.square(param)))
        for param in network.parameters()
        if param.ndim >= 2
    )
    return 0.5 * weight_decay * l2_sum


def _iter_batches(
    inputs: ArrayFloat, labels: ArrayInt, batch_size: int
) -> Iterator[tuple[ArrayFloat, ArrayInt]]:
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        yield inputs[start:end], labels[start:end]


def _evaluate(
    network: NeuralNetwork,
    inputs: ArrayFloat,
    labels: ArrayInt,
    *,
    batch_size: int,
    class_weights: ArrayFloat | None,
    label_smoothing: float,
    weight_decay: float,
) -> Metrics:
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = inputs.shape[0]
    reg_loss = _compute_l2_loss(network, weight_decay)

    for batch_inputs, batch_labels in _iter_batches(inputs, labels, batch_size):
        logits = network.forward(batch_inputs, training=False)
        loss, _ = softmax_cross_entropy_with_logits(
            logits,
            batch_labels,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        batch_size_actual = batch_labels.shape[0]
        total_loss += (loss + reg_loss) * batch_size_actual
        total_accuracy += _batch_accuracy(logits, batch_labels) * batch_size_actual

    if total_samples == 0:
        return Metrics(loss=0.0, accuracy=0.0)
    return Metrics(
        loss=total_loss / total_samples, accuracy=total_accuracy / total_samples
    )


def _cosine_warmup_lr(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr
    step_clamped = min(max(step, 0), total_steps)
    warmup_steps = min(max(warmup_steps, 0), total_steps)
    if warmup_steps > 0 and step_clamped <= warmup_steps:
        return base_lr * step_clamped / warmup_steps
    if total_steps == warmup_steps:
        return base_lr
    progress = (step_clamped - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _build_network(
    *,
    init_rng: np.random.Generator | None,
    dropout_rng: np.random.Generator | None,
) -> NeuralNetwork:
    layers = [
        Conv2DLayer(
            in_channels=18,
            out_channels=64,
            kernel_size=3,
            padding=1,
            activation=relu,
            activation_derivative=relu_derivative,
            rng=init_rng,
        ),
        Conv2DLayer(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            activation=relu,
            activation_derivative=relu_derivative,
            rng=init_rng,
        ),
        DropoutLayer(p=0.1, rng=dropout_rng),
        GlobalAvgPool2D(),
        DenseLayer(
            in_features=64,
            out_features=64,
            activation=relu,
            activation_derivative=relu_derivative,
            rng=init_rng,
        ),
        DenseLayer(in_features=64, out_features=3, rng=init_rng),
    ]
    return NeuralNetwork(layers=layers)


def _per_class_recall(
    labels: ArrayInt, predictions: ArrayInt, num_classes: int
) -> list[float]:
    recalls: list[float] = []
    for class_idx in range(num_classes):
        true_mask = labels == class_idx
        total = int(np.sum(true_mask))
        if total == 0:
            recalls.append(0.0)
            continue
        recalls.append(float(np.sum(predictions[true_mask] == class_idx) / total))
    return recalls


def _train(
    *,
    network: NeuralNetwork,
    optimizer: AdamW,
    train_inputs: ArrayFloat,
    train_labels: ArrayInt,
    val_inputs: ArrayFloat,
    val_labels: ArrayInt,
    epochs: int,
    batch_size: int,
    class_weights: ArrayFloat | None,
    label_smoothing: float,
    weight_decay: float,
    early_stopping_patience: int,
    max_grad_norm: float,
    warmup_ratio: float,
    augment_mirror: bool,
    mirror_prob: float,
    shuffle_rng: np.random.Generator,
    augment_rng: np.random.Generator,
) -> tuple[Metrics, int, tuple[ArrayFloat, ...]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")

    num_train = train_inputs.shape[0]
    steps_per_epoch = int(math.ceil(num_train / batch_size)) if num_train else 0
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(math.floor(total_steps * warmup_ratio))
    if warmup_ratio > 0.0 and warmup_steps == 0 and total_steps > 0:
        warmup_steps = 1

    best_val_loss = float("inf")
    best_val_metrics = Metrics(loss=0.0, accuracy=0.0)
    best_epoch = 0
    best_parameters: tuple[ArrayFloat, ...] | None = None
    epochs_since_improvement = 0
    global_step = 0
    base_lr = optimizer.lr

    for epoch in range(1, epochs + 1):
        indices = (
            shuffle_rng.permutation(num_train)
            if num_train > 0
            else np.array([], dtype=int)
        )
        epoch_inputs = (
            _apply_mirror_augmentation(
                train_inputs, rng=augment_rng, mirror_prob=mirror_prob
            )
            if augment_mirror
            else train_inputs
        )
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        seen = 0
        reg_loss = _compute_l2_loss(network, weight_decay)

        for start in range(0, num_train, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = epoch_inputs[batch_indices]
            batch_labels = train_labels[batch_indices]

            logits = network.forward(batch_inputs, training=True)
            loss, grad = softmax_cross_entropy_with_logits(
                logits,
                batch_labels,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
            network.zero_grad()
            network.backward(grad)
            gradients = network.gradients()

            total_norm_sq = sum(float(np.sum(np.square(grad))) for grad in gradients)
            if total_norm_sq > 0.0:
                total_norm = math.sqrt(total_norm_sq)
                if total_norm > max_grad_norm:
                    scale = max_grad_norm / total_norm
                    for grad_array in gradients:
                        grad_array *= scale

            global_step += 1
            optimizer.lr = _cosine_warmup_lr(
                global_step,
                total_steps=total_steps,
                base_lr=base_lr,
                warmup_steps=warmup_steps,
            )
            optimizer.step(network.parameters(), gradients)

            batch_size_actual = batch_labels.shape[0]
            epoch_loss += (loss + reg_loss) * batch_size_actual
            epoch_accuracy += _batch_accuracy(logits, batch_labels) * batch_size_actual
            seen += batch_size_actual

        train_metrics = Metrics(
            loss=epoch_loss / seen if seen else 0.0,
            accuracy=epoch_accuracy / seen if seen else 0.0,
        )
        val_metrics = _evaluate(
            network,
            val_inputs,
            val_labels,
            batch_size=batch_size,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
        )

        print(
            "Epoch {}/{} - train loss {:.4f} acc {:.4f} - val loss {:.4f} acc {:.4f}".format(
                epoch,
                epochs,
                train_metrics.loss,
                train_metrics.accuracy,
                val_metrics.loss,
                val_metrics.accuracy,
            )
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_parameters = tuple(
                np.array(param, copy=True) for param in network.parameters()
            )
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            break

    if best_parameters is None:
        best_parameters = tuple(np.array(param, copy=True) for param in network.parameters())
        best_epoch = epochs
        best_val_metrics = val_metrics

    return best_val_metrics, best_epoch, best_parameters


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a CNN model and save my_torch_network_best.nn"
    )
    parser.add_argument("dataset", type=Path, help="Path to the dataset file (.txt or .npz)")
    parser.add_argument(
        "--output",
        "--save",
        dest="output",
        type=Path,
        default=Path("my_torch_network_best.nn"),
        help="Output path for the trained .nn model",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to an existing .nn model to resume training from",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--augment-mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mirror-prob", type=float, default=0.5)
    parser.add_argument(
        "--class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument(
        "--cache-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}", file=sys.stderr)
        return 84
    if args.lr <= 0:
        print("Error: --lr must be positive", file=sys.stderr)
        return 84
    if args.weight_decay < 0:
        print("Error: --weight-decay must be non-negative", file=sys.stderr)
        return 84
    if args.batch_size <= 0:
        print("Error: --batch-size must be positive", file=sys.stderr)
        return 84
    if args.epochs <= 0:
        print("Error: --epochs must be positive", file=sys.stderr)
        return 84
    if args.patience <= 0:
        print("Error: --patience must be positive", file=sys.stderr)
        return 84
    if args.max_grad_norm <= 0:
        print("Error: --max-grad-norm must be positive", file=sys.stderr)
        return 84
    if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
        print("Error: --val-ratio must be between 0 and 1", file=sys.stderr)
        return 84
    if args.resume is not None and not args.resume.exists():
        print(f"Error: resume model not found: {args.resume}", file=sys.stderr)
        return 84

    base_rng = np.random.default_rng(args.seed)
    init_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    dropout_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    split_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    shuffle_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    augment_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))

    try:
        inputs, labels = load_dataset(
            str(args.dataset), cache_npz=args.cache_npz
        )
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 84

    if inputs.ndim != 4 or inputs.shape[1:] != (18, 8, 8):
        print("Error: dataset inputs must have shape (N, 18, 8, 8)", file=sys.stderr)
        return 84

    num_classes = get_num_classes()
    try:
        train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
            inputs,
            labels,
            val_ratio=args.val_ratio,
            stratify=True,
            num_classes=num_classes,
            rng=split_rng,
        )
    except Exception as exc:
        print(f"Error splitting dataset: {exc}", file=sys.stderr)
        return 84

    class_weights = None
    if args.class_weights:
        try:
            class_weights = compute_class_weights(train_labels, num_classes)
        except Exception as exc:
            print(f"Error computing class weights: {exc}", file=sys.stderr)
            return 84

    if args.resume is None:
        network = _build_network(init_rng=init_rng, dropout_rng=dropout_rng)
    else:
        try:
            network = load_network(args.resume)
        except Exception as exc:
            print(f"Error loading resume model: {exc}", file=sys.stderr)
            return 84
        for layer in getattr(network, "layers", []):
            if isinstance(layer, DropoutLayer):
                layer.rng = dropout_rng
        print(f"Resuming training from {args.resume}")
    optimizer = AdamW(lr=args.lr, weight_decay=args.weight_decay)

    best_val_metrics, best_epoch, best_parameters = _train(
        network=network,
        optimizer=optimizer,
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        augment_mirror=args.augment_mirror,
        mirror_prob=args.mirror_prob,
        shuffle_rng=shuffle_rng,
        augment_rng=augment_rng,
    )

    for param, best_param in zip(network.parameters(), best_parameters):
        param[...] = best_param

    logits = network.forward(val_inputs, training=False)
    predictions = np.argmax(logits, axis=1)
    recalls = _per_class_recall(val_labels, predictions, num_classes)

    print("\nPer-class Recall (validation):")
    for idx, recall in enumerate(recalls):
        label = get_label_from_index(idx)
        print(f"{label}: {recall:.4f}")

    config = {
        "model": {
            "input_shape": [18, 8, 8],
            "conv_channels": [64, 64],
            "kernel_size": 3,
            "padding": 1,
            "dropout": 0.1,
            "global_avg_pool": True,
            "dense": [64, 3],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "adamw",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "class_weights": args.class_weights,
            "val_ratio": args.val_ratio,
            "augment_mirror": args.augment_mirror,
            "mirror_prob": args.mirror_prob,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "patience": args.patience,
        },
    }
    metadata = {
        "config": config,
        "seed": args.seed,
        "val_accuracy": best_val_metrics.accuracy,
        "epoch": best_epoch,
    }

    save_network(args.output, network, metadata=metadata)
    print(
        "Saved best model to {} (val acc {:.4f} at epoch {})".format(
            args.output,
            best_val_metrics.accuracy,
            best_epoch,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
