from pathlib import Path

import numpy as np
import pytest

from my_torch_analyzer.dataset import load_prediction_dataset


def test_load_prediction_dataset(tmp_path: Path) -> None:
    file_path = tmp_path / "prediction_chessboards.txt"
    content = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1 Check\n"
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 Nothing\n"
    )
    file_path.write_text(content, encoding="utf-8")

    inputs = load_prediction_dataset(str(file_path))

    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (3, 18, 8, 8)
    assert inputs.dtype == np.float32

    assert np.all(inputs[0, 17, :, :] == 1.0)

    assert np.all(inputs[1, 17, :, :] == 0.0)

    assert np.all(inputs[2, 17, :, :] == 1.0)


def test_load_prediction_dataset_malformed(tmp_path: Path) -> None:
    file_path = tmp_path / "malformed.txt"
    file_path.write_text("not a fen\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected at least 6 fields"):
        load_prediction_dataset(str(file_path))
