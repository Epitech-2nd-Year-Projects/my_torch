import subprocess
import sys
from pathlib import Path

import numpy as np

from my_torch.activations import relu
from my_torch.layers import DenseLayer
from my_torch.neural_network import NeuralNetwork
from my_torch.nn_io import save_nn

PYTHON = sys.executable
CLI_MODULE = "my_torch_analyzer"


def run_cli(
    args: list[str], capture_output: bool = True
) -> subprocess.CompletedProcess[str]:
    """Result of running the CLI with given arguments."""
    return subprocess.run(
        [PYTHON, "-m", CLI_MODULE] + args,
        capture_output=capture_output,
        text=True,
    )


def create_dummy_network(path: Path) -> None:
    """Creates a dummy neural network and saves it to the given path."""
    rng = np.random.default_rng(42)
    layer1 = DenseLayer(in_features=1152, out_features=10, activation=relu, rng=rng)
    layer2 = DenseLayer(
        in_features=10,
        out_features=3,
        rng=rng,
    )
    network = NeuralNetwork(layers=[layer1, layer2])
    save_nn(network, path)


def create_dummy_dataset(path: Path, include_labels: bool = True) -> None:
    """Creates a dummy chess dataset."""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "8/8/8/8/8/8/8/k1K5 w - - 0 1",
    ] * 2
    labels = ["Nothing", "Check", "Nothing"] * 2

    lines = []
    for i, fen in enumerate(fens):
        line = fen
        if include_labels:
            line += f" {labels[i]}"
        lines.append(line)

    path.write_text("\n".join(lines), encoding="utf-8")


def test_help_output() -> None:
    """Test that --help returns 0 and prints usage."""
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "USAGE" in result.stdout
    assert "DESCRIPTION" in result.stdout


def test_integration_train_and_predict(tmp_path: Path) -> None:
    """
    Test a full flow:
    1. Create a network.
    2. Train on a small dataset.
    3. Predict on the same dataset.
    """
    nn_path = tmp_path / "initial.nn"
    create_dummy_network(nn_path)

    train_data_path = tmp_path / "train_chess.txt"
    create_dummy_dataset(train_data_path, include_labels=True)

    saved_model_path = tmp_path / "trained.nn"

    train_args = [
        "--train",
        "--save",
        str(saved_model_path),
        str(nn_path),
        str(train_data_path),
    ]
    train_result = run_cli(train_args)

    assert train_result.returncode == 0, f"Training failed: {train_result.stderr}"
    assert saved_model_path.exists(), "Saved model file was not created"
    assert "Training Summary:" in train_result.stdout
    assert "Final Training Loss:" in train_result.stdout
    assert "Epochs Performed:" in train_result.stdout

    predict_args = ["--predict", str(saved_model_path), str(train_data_path)]
    predict_result = run_cli(predict_args)

    assert predict_result.returncode == 0, f"Prediction failed: {predict_result.stderr}"

    output_lines = predict_result.stdout.strip().split("\n")
    assert len(output_lines) == 6
    for line in output_lines:
        assert line.strip() in ["Nothing", "Check", "Checkmate"]


def test_predict_without_valid_file(tmp_path: Path) -> None:
    """Test prediction fails gracefully (exit code 84) with missing file."""
    result = run_cli(
        ["--predict", str(tmp_path / "missing.nn"), str(tmp_path / "chess.txt")]
    )
    assert result.returncode == 84
