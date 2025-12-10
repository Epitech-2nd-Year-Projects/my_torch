"""Analyzer entry point package for the my_torch project."""

from .fen import CastlingRights, FENError, FENPosition, parse_fen

__all__ = [
    "CastlingRights",
    "FENError",
    "FENPosition",
    "parse_fen",
    "main",
]


def main() -> None:
    """Placeholder console script entry point."""
    print("my_torch_analyzer CLI is not implemented yet.")


if __name__ == "__main__":
    main()
