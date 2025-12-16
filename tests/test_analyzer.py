from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from my_torch_analyzer import main


@patch("builtins.print")
def test_main_no_args(
    mock_print: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(sys, "argv", ["my_torch_analyzer"]):
        assert main() == 84


def test_main_help_short(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["my_torch_analyzer", "-h"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "USAGE" in captured.out


def test_main_help_long(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(sys, "argv", ["my_torch_analyzer", "--help"]):
        assert main() == 0
    captured = capsys.readouterr()
    assert "USAGE" in captured.out


@patch("my_torch_analyzer.cli.load_network")
@patch("my_torch_analyzer.cli.load_prediction_dataset")
def test_main_predict(
    mock_load_dataset: MagicMock,
    mock_load_network: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_network = MagicMock()
    mock_network.forward.return_value = np.array(
        [[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]], dtype=np.float32
    )
    mock_load_network.return_value = mock_network

    mock_load_dataset.return_value = np.zeros((2, 18, 8, 8), dtype=np.float32)

    args = ["my_torch_analyzer", "--predict", "model.nn", "games.fen"]
    with patch.object(sys, "argv", args):
        assert main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() != ""
    mock_network.forward.assert_called_once()


@patch("my_torch_analyzer.cli.load_network")
@patch("my_torch_analyzer.cli.load_dataset")
@patch("my_torch_analyzer.cli.train_validation_split")
@patch("my_torch_analyzer.cli.train")
@patch("my_torch_analyzer.cli.save_network")
def test_main_train(
    mock_save: MagicMock,
    mock_train: MagicMock,
    mock_split: MagicMock,
    mock_load_dataset: MagicMock,
    mock_load_network: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_network = MagicMock()
    mock_load_network.return_value = mock_network

    mock_load_dataset.return_value = (MagicMock(), MagicMock())
    mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())

    mock_history = MagicMock()
    mock_history.best_parameters = [np.array([1.0])]
    mock_metric = MagicMock()
    mock_metric.loss = 0.5
    mock_metric.accuracy = 0.8
    mock_history.train = [mock_metric]
    mock_history.validation = [mock_metric]
    mock_train.return_value = mock_history

    mock_param = np.array([0.0])
    mock_network.parameters.return_value = [mock_param]

    args = ["my_torch_analyzer", "--train", "model.nn", "games.fen"]
    with patch.object(sys, "argv", args):
        assert main() == 0

    mock_train.assert_called_once()
    mock_save.assert_called_with("model.nn", mock_network)
    assert mock_param[0] == 1.0


@patch("my_torch_analyzer.cli.load_network")
@patch("my_torch_analyzer.cli.load_dataset")
@patch("my_torch_analyzer.cli.train_validation_split")
@patch("my_torch_analyzer.cli.train")
@patch("my_torch_analyzer.cli.save_network")
def test_main_train_with_save(
    mock_save: MagicMock,
    mock_train: MagicMock,
    mock_split: MagicMock,
    mock_load_dataset: MagicMock,
    mock_load_network: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_network = MagicMock()
    mock_load_network.return_value = mock_network
    mock_load_dataset.return_value = (MagicMock(), MagicMock())
    mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_history = MagicMock()
    mock_history.best_parameters = None
    mock_metric = MagicMock()
    mock_metric.loss = 0.5
    mock_metric.accuracy = 0.8
    mock_history.train = [mock_metric]
    mock_history.validation = [mock_metric]
    mock_train.return_value = mock_history

    args = [
        "my_torch_analyzer",
        "--train",
        "--save",
        "new_model.nn",
        "model.nn",
        "games.fen",
    ]
    with patch.object(sys, "argv", args):
        assert main() == 0

    mock_save.assert_called_with("new_model.nn", mock_network)


def test_main_invalid_mutex(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "my_torch_analyzer",
        "--predict",
        "--train",
        "model.nn",
        "games.fen",
    ]
    with patch.object(sys, "argv", args):
        assert main() == 84


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
        assert main() == 84
    captured = capsys.readouterr()
    assert "--save is only valid in --train mode" in captured.err
