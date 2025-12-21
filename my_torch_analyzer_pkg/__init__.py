"""Analyzer entry point package for the my_torch project."""

from .cli import main
from .fen import CastlingRights, FENError, FENPosition, parse_fen

__all__ = [
    "CastlingRights",
    "FENError",
    "FENPosition",
    "parse_fen",
    "main",
]
