from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from my_torch_analyzer.fen import fen_to_tensor
from my_torch_analyzer.labels import get_label_index, simplify_label


def load_dataset(file_path: str) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Loads a chessboard training dataset from a file.

    Each line in the file is expected to contain a FEN string (6 fields)
    followed by a label, separated by spaces.

    Args:
        file_path: Path to the dataset file.

    Returns:
        A tuple (inputs, labels):
            - inputs: A NumPy array of shape (N, 18, 8, 8) containing the encoded
              boards.
            - labels: A NumPy array of shape (N,) containing the integer class labels.

    Raises:
        ValueError: If a line is malformed or parsing fails.
    """
    inputs = []
    targets = []

    with open(file_path, "r", encoding="utf-8") as f:
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
