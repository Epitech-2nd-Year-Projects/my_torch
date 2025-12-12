from __future__ import annotations

import sys
from unittest.mock import patch

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


def test_main_predict_valid(capsys: pytest.CaptureFixture[str]) -> None:
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
