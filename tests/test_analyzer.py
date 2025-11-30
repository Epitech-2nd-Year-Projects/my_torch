from __future__ import annotations

import pytest

from my_torch_analyzer import main


def test_main_prints_placeholder(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "my_torch_analyzer cli is not implemented yet" in captured.out.lower()
