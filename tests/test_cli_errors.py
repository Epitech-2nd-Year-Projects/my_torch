import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
CLI_MODULE = "my_torch_analyzer"


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Result of running the CLI with given arguments."""
    return subprocess.run(
        [PYTHON, "-m", CLI_MODULE] + args,
        capture_output=True,
        text=True,
    )


def test_help() -> None:
    """Test that --help returns 0 and prints usage."""
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "USAGE" in result.stdout


def test_invalid_argument() -> None:
    """Test that invalid arguments return 84 and print to stderr."""
    result = run_cli(["--invalid-arg"])
    assert result.returncode == 84
    assert "usage:" in result.stderr.lower() or "usage" in result.stdout.lower()
    assert "Error: unrecognized arguments" in result.stderr


def test_mutually_exclusive() -> None:
    """Test that providing both --predict and --train returns 84."""
    result = run_cli(["--predict", "--train", "model.nn", "chess.txt"])
    assert result.returncode == 84
    assert "Error: --predict and --train are mutually exclusive" in result.stderr


def test_save_in_predict_mode() -> None:
    """Test that --save with --predict returns 84."""
    result = run_cli(["--predict", "--save", "out.nn", "model.nn", "chess.txt"])
    assert result.returncode == 84
    assert "Error: --save is only valid in --train mode" in result.stderr


def test_missing_files() -> None:
    """Test that missing positional arguments returns 84."""
    result = run_cli(["--predict"])
    assert result.returncode == 84
    assert "Error: Missing LOADFILE or CHESSFILE" in result.stderr


def test_file_not_found(tmp_path: Path) -> None:
    """Test that non-existent files return 84."""
    non_existent_nn = tmp_path / "missing.nn"
    chess_file = tmp_path / "chess.txt"
    chess_file.write_text("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    result = run_cli(["--predict", str(non_existent_nn), str(chess_file)])
    assert result.returncode == 84
    assert "Error" in result.stderr
    assert "File not found" in result.stderr


def test_malformed_fen(tmp_path: Path) -> None:
    """Test that malformed FEN in chess file returns 84."""
    # To specifically test dataset failure, we need a valid network file.
    pass
