from typing import Dict, List

LABEL_NOTHING = "Nothing"
LABEL_CHECK = "Check"
LABEL_CHECKMATE = "Checkmate"

LABEL_CHECK_WHITE = "Check White"
LABEL_CHECK_BLACK = "Check Black"
LABEL_CHECKMATE_WHITE = "Checkmate White"
LABEL_CHECKMATE_BLACK = "Checkmate Black"

LABELS: List[str] = [
    LABEL_NOTHING,
    LABEL_CHECK,
    LABEL_CHECKMATE,
]

LABEL_TO_INDEX: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}

INDEX_TO_LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}


def simplify_label(label: str) -> str:
    """
    Simplifies a detailed label to one of the basic 3 labels.
    E.g. "Checkmate White" -> "Checkmate".
    """
    label = label.strip()
    if label in (LABEL_CHECK_WHITE, LABEL_CHECK_BLACK):
        return LABEL_CHECK
    if label in (LABEL_CHECKMATE_WHITE, LABEL_CHECKMATE_BLACK):
        return LABEL_CHECKMATE
    return label


def get_label_index(label: str) -> int:
    """
    Returns the class index for a given textual label.

    Args:
        label: The label string (e.g., "Checkmate").

    Returns:
        The integer class index.

    Raises:
        ValueError: If the label is not found in the supported labels.
    """
    cleaned_label = label.strip()
    if cleaned_label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label: '{label}'")
    return LABEL_TO_INDEX[cleaned_label]


def get_label_from_index(index: int) -> str:
    """
    Returns the textual label for a given class index.

    Args:
        index: The class index.

    Returns:
        The label string.

    Raises:
        ValueError: If the index is not valid.
    """
    if index not in INDEX_TO_LABEL:
        raise ValueError(f"Unknown class index: {index}")
    return INDEX_TO_LABEL[index]


def get_num_classes() -> int:
    """Returns the total number of supported classes."""
    return len(LABELS)
