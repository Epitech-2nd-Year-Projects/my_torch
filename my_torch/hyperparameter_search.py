from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from .neural_network import NeuralNetwork
from .optimizers import SGD
from .training import AccuracyFn, LossFn, LossGradFn, TrainingHistory, train

SearchMode = Literal["grid", "random"]


@dataclass(slots=True)
class HyperparameterConfig:
    """Hyperparameters explored during search."""

    learning_rate: float
    batch_size: int
    weight_decay: float


@dataclass(slots=True)
class HyperparameterTrial:
    """Outcome for a single hyperparameter configuration."""

    config: HyperparameterConfig
    history: TrainingHistory
    validation_accuracy: float


@dataclass(slots=True)
class HyperparameterSearchSummary:
    """Collection of all trials alongside the best performing one."""

    trials: list[HyperparameterTrial]
    best_trial: HyperparameterTrial


def _ensure_positive(values: Iterable[float], name: str) -> tuple[float, ...]:
    casted = tuple(float(value) for value in values)
    if not casted:
        raise ValueError(f"at least one {name} must be provided")
    if any(value <= 0 for value in casted):
        raise ValueError(f"all {name} values must be positive")
    return casted


def _ensure_non_negative(values: Iterable[float], name: str) -> tuple[float, ...]:
    casted = tuple(float(value) for value in values)
    if not casted:
        raise ValueError(f"at least one {name} must be provided")
    if any(value < 0 for value in casted):
        raise ValueError(f"all {name} values must be non-negative")
    return casted


def _resolve_rng(
    seed: int | None, rng: np.random.Generator | None
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("provide either rng or seed, not both")
    return rng if rng is not None else np.random.default_rng(seed)


def _build_configurations(
    learning_rates: Sequence[float],
    batch_sizes: Sequence[int],
    weight_decays: Sequence[float],
) -> list[HyperparameterConfig]:
    lr_values = _ensure_positive(learning_rates, "learning rate")
    batch_values = tuple(int(value) for value in batch_sizes)
    if not batch_values:
        raise ValueError("at least one batch size must be provided")
    if any(value <= 0 for value in batch_values):
        raise ValueError("batch sizes must be positive integers")
    wd_values = _ensure_non_negative(weight_decays, "weight decay")
    return [
        HyperparameterConfig(lr, batch, wd)
        for lr, batch, wd in product(lr_values, batch_values, wd_values)
    ]


def _select_random_configs(
    configs: Sequence[HyperparameterConfig],
    sample_count: int,
    rng: np.random.Generator,
) -> list[HyperparameterConfig]:
    if sample_count <= 0:
        raise ValueError("num_samples must be positive for random search")
    total = len(configs)
    if total == 0:
        return []
    sample_count = min(sample_count, total)
    indices = rng.choice(total, size=sample_count, replace=False)
    return [configs[int(idx)] for idx in np.atleast_1d(indices)]


def search_hyperparameters(
    network_builder: Callable[[], NeuralNetwork],
    train_inputs,
    train_labels,
    val_inputs,
    val_labels,
    *,
    loss_fn: LossFn,
    loss_grad_fn: LossGradFn,
    learning_rates: Sequence[float],
    batch_sizes: Sequence[int],
    weight_decays: Sequence[float],
    epochs: int,
    mode: SearchMode = "grid",
    num_samples: int | None = None,
    optimizer_factory: Callable[[HyperparameterConfig], SGD] | None = None,
    shuffle: bool = True,
    accuracy_fn: AccuracyFn | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    early_stopping_patience: int | None = None,
) -> HyperparameterSearchSummary:
    """
    Explore hyperparameter combinations via grid or random search.

    Args:
        network_builder: Callable returning a new, untrained ``NeuralNetwork``.
        train_inputs: Training features.
        train_labels: Training labels.
        val_inputs: Validation features.
        val_labels: Validation labels.
        loss_fn: Loss function used during training.
        loss_grad_fn: Gradient of ``loss_fn``.
        learning_rates: Candidate learning rate values.
        batch_sizes: Candidate batch sizes.
        weight_decays: Candidate L2 regularization strengths.
        epochs: Number of epochs per trial.
        mode: ``"grid"`` considers all configs, ``"random"`` samples a subset.
        num_samples: Number of random configurations to evaluate.
        optimizer_factory: Optional factory returning an optimizer per config.
        shuffle: Whether to shuffle batches inside ``train``.
        accuracy_fn: Optional accuracy metric for ``train``.
        seed: Seed for the random sampler when ``mode == "random"``.
        rng: Optional NumPy generator superseding ``seed`` when provided.
        early_stopping_patience: Optional patience forwarded to ``train``.
    Returns:
        ``HyperparameterSearchSummary`` describing the evaluated trials.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if mode not in {"grid", "random"}:
        raise ValueError("mode must be either 'grid' or 'random'")

    configs = _build_configurations(learning_rates, batch_sizes, weight_decays)
    if not configs:
        raise ValueError("no hyperparameter configurations to evaluate")

    if optimizer_factory is None:
        optimizer_factory = lambda config: SGD(  # type: ignore[assignment]
            lr=config.learning_rate, weight_decay=config.weight_decay
        )

    if mode == "grid":
        if seed is not None or rng is not None:
            raise ValueError("seed/rng are only supported for random search")
        selected = configs
    else:
        if num_samples is None:
            raise ValueError("num_samples must be provided for random search")
        rng_instance = _resolve_rng(seed, rng)
        selected = _select_random_configs(configs, num_samples, rng_instance)

    trials: list[HyperparameterTrial] = []
    for config in selected:
        network = network_builder()
        optimizer = optimizer_factory(config)
        history = train(
            network,
            optimizer,
            train_inputs,
            train_labels,
            val_inputs=val_inputs,
            val_labels=val_labels,
            loss_fn=loss_fn,
            loss_grad_fn=loss_grad_fn,
            epochs=epochs,
            batch_size=config.batch_size,
            shuffle=shuffle,
            accuracy_fn=accuracy_fn,
            weight_decay=config.weight_decay,
            early_stopping_patience=early_stopping_patience,
        )
        best_val_accuracy = max(
            (epoch.accuracy for epoch in history.validation), default=0.0
        )
        trials.append(
            HyperparameterTrial(
                config=config,
                history=history,
                validation_accuracy=best_val_accuracy,
            )
        )

    best_trial = max(trials, key=lambda trial: trial.validation_accuracy)
    return HyperparameterSearchSummary(trials=trials, best_trial=best_trial)


def format_search_summary(
    summary: HyperparameterSearchSummary, *, top_k: int = 3
) -> str:
    """Return a summary highlighting the best configuration."""

    total_trials = len(summary.trials)
    lines = [
        f"Evaluated {total_trials} hyperparameter configuration(s).",
        "Best configuration:",
        (
            f"  learning_rate={summary.best_trial.config.learning_rate:.6g}, "
            f"batch_size={summary.best_trial.config.batch_size}, "
            f"weight_decay={summary.best_trial.config.weight_decay:.3g}"
        ),
        (f"validation_accuracy={summary.best_trial.validation_accuracy:.4f}"),
    ]

    if total_trials > 1 and top_k > 0:
        lines.append("")
        lines.append(f"Top {min(top_k, total_trials)} results:")
        sorted_trials = sorted(
            summary.trials,
            key=lambda trial: trial.validation_accuracy,
            reverse=True,
        )
        for trial in sorted_trials[:top_k]:
            lines.append(
                (
                    f"  lr={trial.config.learning_rate:.6g}, "
                    f"batch={trial.config.batch_size}, "
                    f"wd={trial.config.weight_decay:.3g} -> "
                    f"val_acc={trial.validation_accuracy:.4f}"
                )
            )
    return "\n".join(lines)


__all__ = [
    "HyperparameterConfig",
    "HyperparameterTrial",
    "HyperparameterSearchSummary",
    "search_hyperparameters",
    "format_search_summary",
]
