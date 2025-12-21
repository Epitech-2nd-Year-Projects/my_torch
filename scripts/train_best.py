from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

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
from my_torch_analyzer.labels import get_label_from_index, get_num_classes

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]


@dataclass(slots=True)
class Metrics:
    loss: float
    accuracy: float


def _batch_accuracy(logits: ArrayFloat, labels: ArrayInt) -> float:
    if labels.size == 0:
        return 0.0
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels))


def _compute_l2_loss(network: NeuralNetwork, weight_decay: float) -> float:
    # Reporting only (optimizer does decoupled weight decay)
    if weight_decay <= 0.0:
        return 0.0
    l2_sum = 0.0
    for p in network.parameters():
        if getattr(p, "ndim", 0) >= 2:
            l2_sum += float(np.sum(p * p))
    return 0.5 * weight_decay * l2_sum


def _iter_batches(
    inputs: ArrayFloat, labels: ArrayInt, batch_size: int
) -> Iterator[tuple[ArrayFloat, ArrayInt]]:
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        yield inputs[start:end], labels[start:end]


def _mirror_lr_subset_inplace(x: ArrayFloat) -> None:
    """
    x: (B, 18, 8, 8)
    Mirror columns (a<->h) + swap castling channels:
      - 13 <-> 14 (white K/Q)
      - 15 <-> 16 (black K/Q)
    """
    x[...] = x[..., ::-1].copy()

    tmp = x[:, 13].copy()
    x[:, 13] = x[:, 14]
    x[:, 14] = tmp

    tmp = x[:, 15].copy()
    x[:, 15] = x[:, 16]
    x[:, 16] = tmp


def _maybe_mirror_augment_batch(
    batch_inputs: ArrayFloat,
    *,
    rng: np.random.Generator,
    mirror_prob: float,
) -> ArrayFloat:
    if mirror_prob <= 0.0:
        return batch_inputs
    if mirror_prob > 1.0:
        raise ValueError("mirror_prob must be in [0, 1]")
    if batch_inputs.ndim != 4 or batch_inputs.shape[1:] != (18, 8, 8):
        raise ValueError("expected inputs of shape (B, 18, 8, 8) for CNN")

    mask = rng.random(batch_inputs.shape[0]) < mirror_prob
    if not np.any(mask):
        return batch_inputs

    augmented = np.array(batch_inputs, copy=True)
    idxs = np.flatnonzero(mask)
    subset = augmented[idxs]
    _mirror_lr_subset_inplace(subset)
    augmented[idxs] = subset
    return augmented


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
    total_acc = 0.0
    n = int(inputs.shape[0])
    if n == 0:
        return Metrics(loss=0.0, accuracy=0.0)

    reg_loss = _compute_l2_loss(network, weight_decay)

    for xb, yb in _iter_batches(inputs, labels, batch_size):
        logits = network.forward(xb, training=False)
        loss, _ = softmax_cross_entropy_with_logits(
            logits,
            yb,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        bs = int(yb.shape[0])
        total_loss += (loss + reg_loss) * bs
        total_acc += _batch_accuracy(logits, yb) * bs

    return Metrics(loss=total_loss / n, accuracy=total_acc / n)


def _cosine_warmup_lr(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr

    step = min(max(step, 0), total_steps)
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    if total_steps == warmup_steps:
        return base_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _build_network(
    *,
    init_rng: np.random.Generator | None,
    dropout_rng: np.random.Generator | None,
    conv_channels: Sequence[int],
    mlp_hidden: Sequence[int],
    conv_dropout: float,
    dense_dropout: float,
) -> NeuralNetwork:
    if not conv_channels:
        raise ValueError("conv_channels must contain at least one value")
    if any(c <= 0 for c in conv_channels):
        raise ValueError("all conv_channels must be positive")
    if any(h <= 0 for h in mlp_hidden):
        raise ValueError("all mlp_hidden values must be positive")
    if not (0.0 <= conv_dropout < 1.0):
        raise ValueError("conv_dropout must be in [0,1)")
    if not (0.0 <= dense_dropout < 1.0):
        raise ValueError("dense_dropout must be in [0,1)")

    layers = []
    in_ch = 18
    for out_ch in conv_channels:
        layers.append(
            Conv2DLayer(
                in_channels=in_ch,
                out_channels=int(out_ch),
                kernel_size=3,
                padding=1,
                activation=relu,
                activation_derivative=relu_derivative,
                rng=init_rng,
            )
        )
        in_ch = int(out_ch)

    if conv_dropout > 0.0:
        layers.append(DropoutLayer(p=float(conv_dropout), rng=dropout_rng))

    layers.append(GlobalAvgPool2D())

    prev = in_ch
    for hidden in mlp_hidden:
        layers.append(
            DenseLayer(
                in_features=int(prev),
                out_features=int(hidden),
                activation=relu,
                activation_derivative=relu_derivative,
                rng=init_rng,
            )
        )
        prev = int(hidden)
        if dense_dropout > 0.0:
            layers.append(DropoutLayer(p=float(dense_dropout), rng=dropout_rng))

    layers.append(DenseLayer(in_features=int(prev), out_features=3, rng=init_rng))
    return NeuralNetwork(layers=layers)


def _per_class_recall(labels: ArrayInt, preds: ArrayInt, num_classes: int) -> list[float]:
    out: list[float] = []
    for k in range(num_classes):
        m = labels == k
        total = int(np.sum(m))
        out.append(0.0 if total == 0 else float(np.sum(preds[m] == k) / total))
    return out


def _batched_predictions(network: NeuralNetwork, inputs: ArrayFloat, *, batch_size: int) -> ArrayInt:
    preds: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], batch_size):
        xb = inputs[start : start + batch_size]
        logits = network.forward(xb, training=False)
        preds.append(np.argmax(logits, axis=1))
    if not preds:
        return np.array([], dtype=int)
    return np.concatenate(preds).astype(int, copy=False)


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
    min_delta: float,
    max_grad_norm: float,
    warmup_ratio: float,
    augment_mirror: bool,
    mirror_prob: float,
    shuffle_rng: np.random.Generator,
    augment_rng: np.random.Generator,
    use_ema: bool,
    ema_decay: float,
) -> tuple[Metrics, int, tuple[ArrayFloat, ...]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be >= 0")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")
    if use_ema and not (0.0 < ema_decay < 1.0):
        raise ValueError("ema_decay must be in (0,1)")

    n = int(train_inputs.shape[0])
    steps_per_epoch = int(math.ceil(n / batch_size)) if n else 0
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(math.floor(total_steps * warmup_ratio))
    if warmup_ratio > 0.0 and warmup_steps == 0 and total_steps > 0:
        warmup_steps = 1

    # Cache param list for speed
    params = network.parameters()

    ema_params = None
    if use_ema:
        ema_params = [np.array(p, copy=True) for p in params]

    best_val = float("inf")
    best_metrics = Metrics(loss=0.0, accuracy=0.0)
    best_epoch = 0
    best_params = None
    best_ema_snapshot = None
    since = 0

    global_step = 0
    base_lr = float(optimizer.lr)

    for epoch in range(1, epochs + 1):
        idx = shuffle_rng.permutation(n) if n else np.array([], dtype=int)

        seen = 0
        sum_loss = 0.0
        sum_acc = 0.0
        reg_loss = _compute_l2_loss(network, weight_decay)

        for start in range(0, n, batch_size):
            bidx = idx[start : start + batch_size]
            xb = train_inputs[bidx]
            yb = train_labels[bidx]

            if augment_mirror and mirror_prob > 0.0:
                xb = _maybe_mirror_augment_batch(xb, rng=augment_rng, mirror_prob=mirror_prob)

            logits = network.forward(xb, training=True)
            loss, grad = softmax_cross_entropy_with_logits(
                logits, yb, class_weights=class_weights, label_smoothing=label_smoothing
            )

            network.zero_grad()
            network.backward(grad)
            grads = network.gradients()

            # Global norm clipping
            norm_sq = 0.0
            for g in grads:
                norm_sq += float(np.sum(g * g))
            if norm_sq > 0.0:
                norm = math.sqrt(norm_sq)
                if norm > max_grad_norm:
                    scale = max_grad_norm / norm
                    for g in grads:
                        g *= scale

            global_step += 1
            optimizer.lr = _cosine_warmup_lr(
                global_step, total_steps=total_steps, base_lr=base_lr, warmup_steps=warmup_steps
            )
            optimizer.step(params, grads)

            if ema_params is not None:
                for e, p in zip(ema_params, params):
                    e *= ema_decay
                    e += (1.0 - ema_decay) * p

            bs = int(yb.shape[0])
            sum_loss += (loss + reg_loss) * bs
            sum_acc += _batch_accuracy(logits, yb) * bs
            seen += bs

        train_m = Metrics(loss=sum_loss / seen if seen else 0.0, accuracy=sum_acc / seen if seen else 0.0)
        val_m = _evaluate(
            network,
            val_inputs,
            val_labels,
            batch_size=batch_size,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            weight_decay=weight_decay,
        )

        print(
            f"Epoch {epoch}/{epochs} - train loss {train_m.loss:.4f} acc {train_m.accuracy:.4f} "
            f"- val loss {val_m.loss:.4f} acc {val_m.accuracy:.4f}"
        )

        improved = val_m.loss < (best_val - min_delta)
        if improved or best_params is None:
            best_val = val_m.loss
            best_metrics = val_m
            best_epoch = epoch
            best_params = tuple(np.array(p, copy=True) for p in params)
            if ema_params is not None:
                best_ema_snapshot = tuple(np.array(e, copy=True) for e in ema_params)
            since = 0
        else:
            since += 1
            if since >= early_stopping_patience:
                break

    if best_params is None:
        best_params = tuple(np.array(p, copy=True) for p in params)
        best_epoch = epochs
        best_metrics = val_m

    if use_ema and best_ema_snapshot is not None:
        return best_metrics, best_epoch, best_ema_snapshot
    return best_metrics, best_epoch, best_params


def _parse_pos_int_list(values: Sequence[str]) -> list[int]:
    out: list[int] = []
    for raw in values:
        try:
            v = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid int: {raw}") from exc
        if v <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        out.append(v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CNN model and save my_torch_network_best.nn")

    parser.add_argument("datasets", nargs="+", type=Path, help="Path(s) to dataset file(s) (.txt or .npz)")
    parser.add_argument("--val-dataset", type=Path, help="Optional validation dataset path (no random split).")

    parser.add_argument(
        "--output",
        "--save",
        dest="output",
        type=Path,
        default=Path("my_torch_network_best.nn"),
        help="Output path for the trained .nn model",
    )
    parser.add_argument("--resume", type=Path, help="Path to an existing .nn model to resume training from")

    parser.add_argument(
        "--conv-channels",
        nargs="+",
        default=["64", "64", "64", "64"],
        help="Conv out_channels per layer (e.g. --conv-channels 64 64 64 64)",
    )
    parser.add_argument(
        "--mlp-hidden",
        nargs="+",
        default=["128"],
        help="Dense hidden sizes after GAP (e.g. --mlp-hidden 128 64)",
    )
    parser.add_argument("--conv-dropout", type=float, default=0.10)
    parser.add_argument("--dense-dropout", type=float, default=0.20)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--augment-mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror-prob", type=float, default=0.5)

    parser.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--cache-npz", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    # Basic validation
    for ds in args.datasets:
        if not ds.exists():
            print(f"Error: dataset not found: {ds}", file=sys.stderr)
            return 84
    if args.val_dataset is not None and not args.val_dataset.exists():
        print(f"Error: val dataset not found: {args.val_dataset}", file=sys.stderr)
        return 84
    if args.resume is not None and not args.resume.exists():
        print(f"Error: resume model not found: {args.resume}", file=sys.stderr)
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
    if args.val_dataset is None and (args.val_ratio <= 0.0 or args.val_ratio >= 1.0):
        print("Error: --val-ratio must be between 0 and 1", file=sys.stderr)
        return 84
    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        print("Error: --label-smoothing must be in [0, 1)", file=sys.stderr)
        return 84
    if args.mirror_prob < 0.0 or args.mirror_prob > 1.0:
        print("Error: --mirror-prob must be in [0, 1]", file=sys.stderr)
        return 84
    if args.conv_dropout < 0.0 or args.conv_dropout >= 1.0:
        print("Error: --conv-dropout must be in [0, 1)", file=sys.stderr)
        return 84
    if args.dense_dropout < 0.0 or args.dense_dropout >= 1.0:
        print("Error: --dense-dropout must be in [0, 1)", file=sys.stderr)
        return 84
    if args.min_delta < 0.0:
        print("Error: --min-delta must be >= 0", file=sys.stderr)
        return 84
    if args.ema and (args.ema_decay <= 0.0 or args.ema_decay >= 1.0):
        print("Error: --ema-decay must be in (0, 1)", file=sys.stderr)
        return 84

    try:
        conv_channels = _parse_pos_int_list(args.conv_channels)
        mlp_hidden = _parse_pos_int_list(args.mlp_hidden)
    except argparse.ArgumentTypeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 84

    # RNGs
    base_rng = np.random.default_rng(args.seed)
    init_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    dropout_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    split_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    shuffle_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    augment_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))

    # Load training datasets + concat
    xs, ys = [], []
    for ds in args.datasets:
        try:
            x, y = load_dataset(str(ds), cache_npz=args.cache_npz)
        except Exception as exc:
            print(f"Error loading dataset {ds}: {exc}", file=sys.stderr)
            return 84
        if x.ndim != 4 or x.shape[1:] != (18, 8, 8):
            print(f"Error: dataset inputs must have shape (N, 18, 8, 8), got {x.shape} from {ds}", file=sys.stderr)
            return 84
        xs.append(np.asarray(x))
        ys.append(np.asarray(y))

    inputs_all = xs[0] if len(xs) == 1 else np.concatenate(xs, axis=0)
    labels_all = ys[0] if len(ys) == 1 else np.concatenate(ys, axis=0)

    num_classes = get_num_classes()

    # Validation: external or split
    if args.val_dataset is not None:
        try:
            val_inputs, val_labels = load_dataset(str(args.val_dataset), cache_npz=args.cache_npz)
        except Exception as exc:
            print(f"Error loading val dataset: {exc}", file=sys.stderr)
            return 84
        if val_inputs.ndim != 4 or val_inputs.shape[1:] != (18, 8, 8):
            print("Error: val inputs must have shape (N, 18, 8, 8)", file=sys.stderr)
            return 84
        train_inputs, train_labels = inputs_all, labels_all
    else:
        try:
            train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
                inputs_all,
                labels_all,
                val_ratio=args.val_ratio,
                stratify=True,
                num_classes=num_classes,
                rng=split_rng,
            )
        except Exception as exc:
            print(f"Error splitting dataset: {exc}", file=sys.stderr)
            return 84

    def _counts(arr: ArrayInt) -> list[int]:
        return np.bincount(np.asarray(arr, dtype=int), minlength=num_classes).tolist()

    print("Train samples:", int(train_inputs.shape[0]), "Val samples:", int(val_inputs.shape[0]))
    print("Train class counts:", _counts(train_labels))
    print("Val class counts:", _counts(val_labels))

    class_weights = None
    if args.class_weights:
        try:
            class_weights = compute_class_weights(train_labels, num_classes)
        except Exception as exc:
            print(f"Error computing class weights: {exc}", file=sys.stderr)
            return 84

    # Build / resume
    if args.resume is None:
        network = _build_network(
            init_rng=init_rng,
            dropout_rng=dropout_rng,
            conv_channels=conv_channels,
            mlp_hidden=mlp_hidden,
            conv_dropout=float(args.conv_dropout),
            dense_dropout=float(args.dense_dropout),
        )
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

    optimizer = AdamW(lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val_metrics, best_epoch, best_params = _train(
        network=network,
        optimizer=optimizer,
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        class_weights=class_weights,
        label_smoothing=float(args.label_smoothing),
        weight_decay=float(args.weight_decay),
        early_stopping_patience=int(args.patience),
        min_delta=float(args.min_delta),
        max_grad_norm=float(args.max_grad_norm),
        warmup_ratio=float(args.warmup_ratio),
        augment_mirror=bool(args.augment_mirror),
        mirror_prob=float(args.mirror_prob),
        shuffle_rng=shuffle_rng,
        augment_rng=augment_rng,
        use_ema=bool(args.ema),
        ema_decay=float(args.ema_decay),
    )

    # Restore best
    for p, bp in zip(network.parameters(), best_params):
        p[...] = bp

    preds = _batched_predictions(network, val_inputs, batch_size=int(args.batch_size))
    recalls = _per_class_recall(val_labels, preds, num_classes)

    print("\nPer-class Recall (validation):")
    for i, r in enumerate(recalls):
        print(f"{get_label_from_index(i)}: {r:.4f}")

    config = {
        "model": {
            "input_shape": [18, 8, 8],
            "conv_channels": [int(c) for c in conv_channels],
            "kernel_size": 3,
            "padding": 1,
            "conv_dropout": float(args.conv_dropout),
            "global_avg_pool": True,
            "mlp_hidden": [int(h) for h in mlp_hidden],
            "dense_dropout": float(args.dense_dropout),
            "output_classes": 3,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "optimizer": "adamw",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "label_smoothing": float(args.label_smoothing),
            "class_weights": bool(args.class_weights),
            "val_ratio": float(args.val_ratio) if args.val_dataset is None else None,
            "val_dataset": str(args.val_dataset) if args.val_dataset is not None else None,
            "augment_mirror": bool(args.augment_mirror),
            "mirror_prob": float(args.mirror_prob),
            "warmup_ratio": float(args.warmup_ratio),
            "max_grad_norm": float(args.max_grad_norm),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "ema": bool(args.ema),
            "ema_decay": float(args.ema_decay) if args.ema else None,
        },
    }
    metadata = {
        "config": config,
        "seed": args.seed,
        "val_accuracy": float(best_val_metrics.accuracy),
        "epoch": int(best_epoch),
    }

    save_network(args.output, network, metadata=metadata)
    print(f"Saved best model to {args.output} (val acc {best_val_metrics.accuracy:.4f} at epoch {best_epoch})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
