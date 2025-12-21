import pytest

from my_torch_analyzer_pkg.labels import (
    LABEL_CHECK,
    LABEL_CHECK_BLACK,
    LABEL_CHECK_WHITE,
    LABEL_CHECKMATE,
    LABEL_CHECKMATE_BLACK,
    LABEL_CHECKMATE_WHITE,
    LABEL_NOTHING,
    get_label_from_index,
    get_label_index,
    get_num_classes,
    simplify_label,
)


def test_label_to_index_mapping() -> None:
    assert get_label_index(LABEL_NOTHING) == 0
    assert get_label_index(LABEL_CHECK) == 1
    assert get_label_index(LABEL_CHECKMATE) == 2


def test_index_to_label_mapping() -> None:
    assert get_label_from_index(0) == LABEL_NOTHING
    assert get_label_from_index(1) == LABEL_CHECK
    assert get_label_from_index(2) == LABEL_CHECKMATE


def test_invalid_label() -> None:
    with pytest.raises(ValueError):
        get_label_index("InvalidLabel")


def test_invalid_index() -> None:
    with pytest.raises(ValueError):
        get_label_from_index(999)
    with pytest.raises(ValueError):
        get_label_from_index(-1)


def test_whitespace_handling() -> None:
    assert get_label_index("Nothing ") == 0
    assert get_label_index(" Check") == 1


def test_num_classes() -> None:
    assert get_num_classes() == 3


def test_simplify_label() -> None:
    assert simplify_label(LABEL_CHECK_WHITE) == LABEL_CHECK
    assert simplify_label(LABEL_CHECK_BLACK) == LABEL_CHECK
    assert simplify_label(LABEL_CHECKMATE_WHITE) == LABEL_CHECKMATE
    assert simplify_label(LABEL_CHECKMATE_BLACK) == LABEL_CHECKMATE
    assert simplify_label(LABEL_NOTHING) == LABEL_NOTHING
    assert simplify_label("Unknown") == "Unknown"
