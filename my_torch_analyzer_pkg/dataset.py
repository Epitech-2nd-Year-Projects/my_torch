from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from my_torch_analyzer_pkg.fen import fen_to_tensor, mirror_tensor_lr
from my_torch_analyzer_pkg.labels import get_label_index, simplify_label


def _load_npz_dataset(
    file_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    with np.load(file_path, allow_pickle=False) as data:
        inputs = data["inputs"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
    return inputs, labels


def _save_npz_dataset(
    file_path: Path,
    inputs: NDArray[np.float32],
    labels: NDArray[np.int64],
) -> None:
    np.savez(file_path, inputs=inputs, labels=labels)


def _parse_text_dataset(
    file_path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    inputs = []
    targets = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(
                    f"Line {line_num}: Malformed line. Expected at least 7 fields "
                    "(6 for FEN, 1+ for label)."
                )

            fen_str = " ".join(parts[:6])
            label_str = " ".join(parts[6:])

            try:
                tensor = fen_to_tensor(fen_str)
                simplified_label = simplify_label(label_str)
                label_idx = get_label_index(simplified_label)

                inputs.append(tensor)
                targets.append(label_idx)
            except Exception as exc:
                raise ValueError(f"Line {line_num}: Failed to parse data.") from exc

    return (
        np.array(inputs, dtype=np.float32),
        np.array(targets, dtype=np.int64),
    )


def _apply_mirror_augmentation(
    inputs: NDArray[np.float32],
    *,
    mirror_prob: float,
    seed: int | None,
) -> NDArray[np.float32]:
    if mirror_prob < 0.0 or mirror_prob > 1.0:
        raise ValueError("mirror_prob must be in the range [0, 1]")
    if inputs.ndim != 4 or inputs.shape[1:] != (18, 8, 8):
        raise ValueError("mirror augmentation requires inputs of shape (N, 18, 8, 8)")
    if mirror_prob == 0.0:
        return inputs

    rng = np.random.default_rng(seed)
    mask = rng.random(inputs.shape[0]) < mirror_prob
    if not np.any(mask):
        return inputs

    augmented = np.array(inputs, copy=True)
    for idx in np.flatnonzero(mask):
        augmented[idx] = mirror_tensor_lr(augmented[idx])
    return augmented


def load_dataset(
    file_path: str,
    *,
    augment_mirror: bool = False,
    mirror_prob: float = 0.5,
    seed: int | None = None,
    cache_npz: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Loads a chessboard training dataset from a file.

    Each line in the file is expected to contain a FEN string (6 fields)
    followed by a label, separated by spaces.

    Args:
        file_path: Path to the dataset file.
        augment_mirror: Enable left-right mirror augmentation for training data
        mirror_prob: Probability of mirroring each sample
        seed: Random seed for mirror augmentation
        cache_npz: Enable caching to a sibling .npz for .txt inputs

    Returns:
        A tuple (inputs, labels):
            - inputs: A NumPy array of shape (N, 18, 8, 8) containing the encoded
              boards.
            - labels: A NumPy array of shape (N,) containing the integer class labels.

    Raises:
        ValueError: If a line is malformed or parsing fails.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        inputs, labels = _load_npz_dataset(path)
    else:
        cache_path = (
            path.with_suffix(".npz") if cache_npz and suffix == ".txt" else None
        )
        if cache_path is not None and cache_path.exists():
            inputs, labels = _load_npz_dataset(cache_path)
        else:
            inputs, labels = _parse_text_dataset(path)
            if cache_path is not None:
                _save_npz_dataset(cache_path, inputs, labels)

    if augment_mirror:
        inputs = _apply_mirror_augmentation(
            inputs,
            mirror_prob=mirror_prob,
            seed=seed,
        )

    return inputs, labels


def load_prediction_dataset(file_path: str) -> NDArray[np.float32]:
    """
    Loads a chessboard prediction dataset from a file.

    Each line in the file is expected to contain a FEN string (6 fields).
    Any additional fields (e.g. labels) are ignored.

    Args:
        file_path: Path to the dataset file.

    Returns:
        A NumPy array of shape (N, 18, 8, 8) containing the encoded boards.

    Raises:
        ValueError: If a line is malformed or parsing fails.
    """
    inputs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Line {line_num}: Malformed line. Expected at least 6 fields "
                    "for FEN."
                )

            fen_str = " ".join(parts[:6])

            try:
                tensor = fen_to_tensor(fen_str)
                inputs.append(tensor)
            except Exception as exc:
                raise ValueError(f"Line {line_num}: Failed to parse data.") from exc

    return np.array(inputs, dtype=np.float32)
