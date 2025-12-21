import json
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_SCRIPT = Path(__file__).parents[1] / "scripts" / "my_torch_generator"


def test_generator_script_basic(tmp_path: Any) -> None:
    config = {"layers": [{"in_features": 4, "out_features": 2, "activation": "relu"}]}
    config_path = tmp_path / "net.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), str(config_path), "2"],
        check=True,
        cwd=tmp_path,
    )

    assert (tmp_path / "net_1.nn").exists()
    assert (tmp_path / "net_2.nn").exists()


def test_generator_script_cnn(tmp_path: Any) -> None:
    subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "--cnn", "1"],
        check=True,
        cwd=tmp_path,
    )

    assert (tmp_path / "chess_cnn_1.nn").exists()


def test_generator_invalid_args(tmp_path: Any) -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT)], capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    )
    
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), "file.json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 84
    assert "Arguments must be pairs" in result.stderr


def test_generator_invalid_count(tmp_path: Any) -> None:
    config_path = tmp_path / "net.json"
    config_path.touch()

    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT), str(config_path), "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 84
    assert "must be positive" in result.stderr
