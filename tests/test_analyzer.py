from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from my_torch_analyzer import main


def test_main_help(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["my_torch_analyzer", "--help"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "USAGE" in captured.out
    assert "DESCRIPTION" in captured.out


def test_main_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["my_torch_analyzer"]):
        assert main() == 1
    captured = capsys.readouterr()
    assert "USAGE" in captured.out


@patch("my_torch_analyzer.cli.load_prediction_dataset")
@patch("my_torch_analyzer.cli.load_network")
def test_main_predict_valid(
    mock_load_net: Any,
    mock_load_data: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_net_instance = MagicMock()
    mock_net_instance.forward.return_value = [[0.1, 0.8, 0.1]]
    mock_load_net.return_value = mock_net_instance

    mock_load_data.return_value = MagicMock(
        shape=(1, 18, 8, 8), reshape=MagicMock(return_value=MagicMock())
    )

    args = ["my_torch_analyzer", "--predict", "model.nn", "games.fen"]
    with patch.object(sys, "argv", args):
        assert main() == 0


def test_main_train_valid(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["my_torch_analyzer", "--train", "model.nn", "games.fen"]
    with patch.object(sys, "argv", args):
        assert main() == 0


def test_main_train_save_valid(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "my_torch_analyzer",
        "--train",
        "--save",
        "new.nn",
        "model.nn",
        "games.fen",
    ]
    with patch.object(sys, "argv", args):
        assert main() == 0


def test_main_invalid_mutex(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "my_torch_analyzer",
        "--predict",
        "--train",
        "model.nn",
        "games.fen",
    ]
    with patch.object(sys, "argv", args):
        assert main() == 1
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.out


def test_main_invalid_save(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "my_torch_analyzer",
        "--predict",
        "--save",
        "new.nn",
        "model.nn",
        "games.fen",
    ]
    with patch.object(sys, "argv", args):
        assert main() == 1
    captured = capsys.readouterr()
    assert "--save is only valid in --train mode" in captured.out
