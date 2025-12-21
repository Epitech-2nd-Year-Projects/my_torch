from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

BoardSquare = str | None
BoardRow = Tuple[BoardSquare, ...]
Board = Tuple[BoardRow, ...]


class FENError(ValueError):
    """Raised when a provided FEN line is malformed."""


@dataclass(frozen=True)
class CastlingRights:
    """Represents all four castling rights."""

    white_kingside: bool
    white_queenside: bool
    black_kingside: bool
    black_queenside: bool


@dataclass(frozen=True)
class FENPosition:
    """Structured representation of a chess position encoded with FEN."""

    board: Board
    active_color: str
    castling_rights: CastlingRights
    en_passant_target: str | None
    halfmove_clock: int
    fullmove_number: int


def parse_fen(fen: str) -> FENPosition:
    """Parse a Forsyth–Edwards Notation (FEN) string into a structured object.

    Args:
        fen: The FEN line to parse.

    Returns:
        A ``FENPosition`` describing the encoded chess position.

    Raises:
        FENError: If the FEN string is malformed.
    """

    trimmed = fen.strip()
    if not trimmed:
        raise FENError("FEN string cannot be empty.")

    parts = trimmed.split()
    if len(parts) != 6:
        raise FENError(
            "FEN string must contain exactly six fields: "
            "piece placement, active color, castling, en passant, "
            "halfmove clock, and fullmove number."
        )

    (
        piece_placement,
        active_color,
        castling,
        en_passant,
        halfmove_clock,
        fullmove_number,
    ) = parts

    board = _parse_piece_placement(piece_placement)
    _validate_kings(board)
    active = _parse_active_color(active_color)
    castling_rights = _parse_castling(castling)
    en_passant_square = _parse_en_passant(en_passant)
    halfmove = _parse_non_negative_int(halfmove_clock, "halfmove clock")
    fullmove = _parse_positive_int(fullmove_number, "fullmove number")

    return FENPosition(
        board=board,
        active_color=active,
        castling_rights=castling_rights,
        en_passant_target=en_passant_square,
        halfmove_clock=halfmove,
        fullmove_number=fullmove,
    )


def _parse_piece_placement(field: str) -> Board:
    ranks = field.split("/")
    if len(ranks) != 8:
        raise FENError("Piece placement must contain eight ranks separated by '/'.")

    rows: list[BoardRow] = []
    valid_pieces = set("prnbqkPRNBQK")

    for rank_index, rank in enumerate(ranks, start=8):
        squares: list[BoardSquare] = []
        file_index = 0

        for char in rank:
            if char.isdigit():
                value = int(char)
                if value < 1 or value > 8:
                    raise FENError(
                        f"Invalid empty square count '{char}' in rank {rank_index}."
                    )
                file_index += value
                squares.extend([None] * value)
                continue

            if char not in valid_pieces:
                raise FENError(
                    f"Invalid piece character '{char}' in rank {rank_index}."
                )

            squares.append(char)
            file_index += 1

        if file_index != 8:
            raise FENError(
                f"Rank {rank_index} must have exactly eight squares, got {file_index}."
            )

        rows.append(tuple(squares))

    return tuple(rows)


def _validate_kings(board: Board) -> None:
    white_kings = sum(1 for row in board for square in row if square == "K")
    black_kings = sum(1 for row in board for square in row if square == "k")
    if white_kings != 1 or black_kings != 1:
        raise FENError(
            "FEN positions must contain exactly one white king 'K' "
            "and one black king 'k'."
        )


def _parse_active_color(field: str) -> str:
    if field not in {"w", "b"}:
        raise FENError("Active color must be 'w' or 'b'.")
    return field


def _parse_castling(field: str) -> CastlingRights:
    if field == "-":
        return CastlingRights(False, False, False, False)

    valid_flags = "KQkq"
    seen: set[str] = set()
    for char in field:
        if char not in valid_flags:
            raise FENError("Castling rights may only contain characters KQkq or '-'.")
        if char in seen:
            raise FENError(f"Duplicate castling right '{char}' detected.")
        seen.add(char)

    return CastlingRights(
        white_kingside="K" in seen,
        white_queenside="Q" in seen,
        black_kingside="k" in seen,
        black_queenside="q" in seen,
    )


def _parse_en_passant(field: str) -> str | None:
    if field == "-":
        return None
    if len(field) != 2:
        raise FENError("En passant target square must be '-' or a valid board square.")

    file_char = field[0]
    rank_char = field[1]
    if file_char not in "abcdefgh":
        raise FENError("En passant file must be a letter from a to h.")
    if rank_char not in "36":
        raise FENError("En passant rank must be either '3' or '6'.")

    return field


def _parse_non_negative_int(value: str, field_name: str) -> int:
    integer = _parse_int(value, field_name)
    if integer < 0:
        raise FENError(f"{field_name} must be a non-negative integer.")
    return integer


def _parse_positive_int(value: str, field_name: str) -> int:
    integer = _parse_int(value, field_name)
    if integer <= 0:
        raise FENError(f"{field_name} must be a positive integer.")
    return integer


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise FENError(f"{field_name} must be an integer.") from exc


def mirror_tensor_lr(tensor: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Mirror a board tensor left to right

    Args:
        tensor: Board tensor shaped 18x8x8

    Returns:
        Mirrored tensor with castling and en passant aligned
    """
    if tensor.ndim != 3 or tensor.shape != (18, 8, 8):
        raise ValueError("tensor must have shape (18, 8, 8)")

    mirrored = np.flip(tensor, axis=2).copy()
    mirrored[[13, 14]] = mirrored[[14, 13]]
    mirrored[[15, 16]] = mirrored[[16, 15]]
    return mirrored


def fen_to_tensor(fen: str | FENPosition) -> np.ndarray[Any, Any]:
    """Encodes a FEN string or FENPosition into a NumPy tensor.

    The output is a (18, 8, 8) float32 NumPy array. The channels are defined as follows:

    - Channels 0-5: White pieces (P, N, B, R, Q, K)
    - Channels 6-11: Black pieces (p, n, b, r, q, k)
    - Channel 12: En Passant target (1 at the target square, 0 elsewhere)
    - Channel 13: White Kingside Castling (all 1s if allowed, 0 otherwise)
    - Channel 14: White Queenside Castling (all 1s if allowed, 0 otherwise)
    - Channel 15: Black Kingside Castling (all 1s if allowed, 0 otherwise)
    - Channel 16: Black Queenside Castling (all 1s if allowed, 0 otherwise)
    - Channel 17: Active Color (all 1s if White, all 0s if Black)

    Args:
        fen: The FEN string or FENPosition object to encode.

    Returns:
        A NumPy array of shape (18, 8, 8) representing the board state.
    """
    if isinstance(fen, str):
        position = parse_fen(fen)
    else:
        position = fen

    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = {
        "P": 0,
        "N": 1,
        "B": 2,
        "R": 3,
        "Q": 4,
        "K": 5,
        "p": 6,
        "n": 7,
        "b": 8,
        "r": 9,
        "q": 10,
        "k": 11,
    }

    for row_idx, row in enumerate(position.board):
        for col_idx, square in enumerate(row):
            if square and square in piece_map:
                channel = piece_map[square]
                tensor[channel, row_idx, col_idx] = 1.0

    if position.en_passant_target:
        file_char = position.en_passant_target[0]
        rank_char = position.en_passant_target[1]

        col_idx = ord(file_char) - ord("a")
        row_idx = 8 - int(rank_char)

        if 0 <= row_idx < 8 and 0 <= col_idx < 8:
            tensor[12, row_idx, col_idx] = 1.0

    if position.castling_rights.white_kingside:
        tensor[13, :, :] = 1.0
    if position.castling_rights.white_queenside:
        tensor[14, :, :] = 1.0
    if position.castling_rights.black_kingside:
        tensor[15, :, :] = 1.0
    if position.castling_rights.black_queenside:
        tensor[16, :, :] = 1.0

    if position.active_color == "w":
        tensor[17, :, :] = 1.0

    return tensor
