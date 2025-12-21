import argparse
import sys
from typing import NoReturn

import numpy as np

from my_torch import activations
from my_torch.layers import (
    Conv2DLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    GlobalAvgPool2D,
)
from my_torch.losses import softmax_cross_entropy_with_logits
from my_torch.nn_io import load_network, save_network
from my_torch.optimizers import SGD, AdamW, SGDMomentum
from my_torch.training import compute_class_weights, train, train_validation_split
from my_torch_analyzer.dataset import load_dataset, load_prediction_dataset
from my_torch_analyzer.labels import get_label_from_index, get_num_classes

EXIT_SUCCESS = 0
EXIT_ERROR = 84

HELP_TEXT = """USAGE
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] [OPTIONS] LOADFILE CHESSFILE
DESCRIPTION
--train
Launch the neural network in training mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation and the expected output
separated by space. If specified, the newly trained neural network will be saved in
SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
--predict
Launch the neural network in prediction mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation, and optionally an expected
output.
--save
Save neural network into SAVEFILE. Only works in train mode.
--lr
Learning rate for training (default: 0.01)
--epochs
Number of training epochs (default: 100)
--batch-size
Training batch size (default: 256)
--weight-decay
L2 weight decay strength (default: 0.001)
--seed
Random seed for data splitting and shuffling
--optimizer {sgd,sgd_momentum,adamw}
Optimizer selection (default: sgd)
--val-ratio
Validation split ratio between 0 and 1 (default: 0.2)
--stratify / --no-stratify
Enable or disable stratified splitting (default: enabled)
--class-weights
Enable class-weighted cross entropy loss
--label-smoothing
Label smoothing factor between 0 and 1 (default: 0.0)
--augment-mirror
Enable random mirror augmentation for training inputs
--mirror-prob
Mirror probability per training sample (default: 0.5)
--max-grad-norm
Max global gradient norm for clipping
LOADFILE
File containing an artificial neural network
CHESSFILE
File containing chessboards"""


class SubjectArgumentParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser that exits with code 84 on error, as required by the subject.
    """

    def error(self, message: str) -> NoReturn:
        print(HELP_TEXT)
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(EXIT_ERROR)


class _CrossEntropyLossAdapter:
    def __init__(
        self,
        *,
        class_weights: np.ndarray | None,
        label_smoothing: float,
    ) -> None:
        self._class_weights = class_weights
        self._label_smoothing = label_smoothing
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
            label_smoothing=self._label_smoothing,
        )
        self._cache_logits = logits
        self._cache_labels = labels
        self._cache_value = (loss, grad)
        return loss, grad


def _apply_mirror_augmentation(
    inputs: np.ndarray,
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
    augmented[mask] = np.flip(augmented[mask], axis=3)
    return augmented


def _build_optimizer(args: argparse.Namespace) -> SGD | SGDMomentum | AdamW:
    if args.optimizer == "sgd":
        return SGD(lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd_momentum":
        return SGDMomentum(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return AdamW(lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def run_analyzer(args: argparse.Namespace) -> None:
    """
    Core logic for handling the analyzer commands.
    """
    if args.predict and args.train:
        raise ValueError("--predict and --train are mutually exclusive.")

    if not args.predict and not args.train:
        raise ValueError("You must specify either --predict or --train.")

    if args.predict and args.save:
        raise ValueError("--save is only valid in --train mode.")

    if not args.loadfile or not args.chessfile:
        raise ValueError("Missing LOADFILE or CHESSFILE.")

    if args.predict:
        network = load_network(args.loadfile)
        inputs = load_prediction_dataset(args.chessfile)
        if _should_flatten_inputs(network):
            inputs = inputs.reshape(inputs.shape[0], -1)

        outputs = network.forward(inputs, training=False)

        predicted_indices = np.argmax(outputs, axis=1)

        for idx in predicted_indices:
            print(get_label_from_index(idx))

    elif args.train:
        network = load_network(args.loadfile)
        last_layer = network.layers[-1] if network.layers else None
        if (
            last_layer is not None
            and getattr(last_layer, "activation", None) is activations.softmax
        ):
            raise ValueError(
                "Last layer must be identity (logits). Loss already applies softmax."
            )
        inputs, labels = load_dataset(args.chessfile)

        num_classes = get_num_classes()
        rng = np.random.default_rng(args.seed)
        (
            train_inputs,
            val_inputs,
            train_labels,
            val_labels,
        ) = train_validation_split(
            inputs,
            labels,
            stratify=args.stratify,
            num_classes=num_classes,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        if args.augment_mirror:
            train_inputs = _apply_mirror_augmentation(
                train_inputs, rng, args.mirror_prob
            )
        if _should_flatten_inputs(network):
            train_inputs = train_inputs.reshape(train_inputs.shape[0], -1)
            val_inputs = val_inputs.reshape(val_inputs.shape[0], -1)

        train_counts = np.bincount(train_labels, minlength=num_classes)
        val_counts = np.bincount(val_labels, minlength=num_classes)
        print("\nClass Distribution:")
        for idx in range(num_classes):
            label = get_label_from_index(idx)
            print(f"{label}: train={train_counts[idx]}, val={val_counts[idx]}")

        class_weights = (
            compute_class_weights(train_labels, num_classes)
            if args.class_weights
            else None
        )
        loss_adapter = _CrossEntropyLossAdapter(
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
        optimizer = _build_optimizer(args)

        history = train(
            network=network,
            optimizer=optimizer,
            train_inputs=train_inputs,
            train_labels=train_labels,
            val_inputs=val_inputs,
            val_labels=val_labels,
            loss_fn=loss_adapter.loss,
            loss_grad_fn=loss_adapter.grad,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=10,
            weight_decay=args.weight_decay,
            rng=rng,
            max_grad_norm=args.max_grad_norm,
        )

        final_train = history.train[-1] if history.train else None
        final_val = history.validation[-1] if history.validation else None
        num_epochs = len(history.train)

        print("\nTraining Summary:")
        if final_train:
            print(f"Final Training Loss: {final_train.loss:.4f}")
            print(f"Final Training Accuracy: {final_train.accuracy:.4f}")
        if final_val:
            print(f"Final Validation Loss: {final_val.loss:.4f}")
            print(f"Final Validation Accuracy: {final_val.accuracy:.4f}")
        print(f"Epochs Performed: {num_epochs}")

        if history.best_parameters is not None:
            for param, best_param in zip(network.parameters(), history.best_parameters):
                param[...] = best_param

        save_path = args.save if args.save else args.loadfile
        save_network(save_path, network)


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


def main() -> int:
    """
    Main entry point for the my_torch_analyzer CLI.
    """
    if "-h" in sys.argv or "--help" in sys.argv:
        print(HELP_TEXT)
        return EXIT_SUCCESS

    parser = SubjectArgumentParser(add_help=False)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save", type=str)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "sgd_momentum", "adamw"],
        default="sgd",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augment-mirror", action="store_true")
    parser.add_argument("--mirror-prob", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("loadfile", nargs="?")
    parser.add_argument("chessfile", nargs="?")

    try:
        args = parser.parse_args()
        run_analyzer(args)
        return EXIT_SUCCESS

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
