from __future__ import annotations

import pytest

from my_torch_analyzer.fen import (
    CastlingRights,
    FENError,
    FENPosition,
    parse_fen,
)


def test_parse_fen_valid_position() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"

    position = parse_fen(fen)

    assert isinstance(position, FENPosition)
    assert position.board[0][0] == "r"
    assert position.board[7][4] == "K"
    assert position.active_color == "w"
    assert position.castling_rights == CastlingRights(True, True, True, True)
    assert position.en_passant_target is None
    assert position.halfmove_clock == 1
    assert position.fullmove_number == 3


def test_parse_fen_minimal_board() -> None:
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    position = parse_fen(fen)
    assert position.board[0][4] == "k"
    assert position.board[7][4] == "K"
    assert position.castling_rights == CastlingRights(False, False, False, False)


def test_parse_fen_complex_state() -> None:
    fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR b Kkq c6 0 2"
    position = parse_fen(fen)
    assert position.active_color == "b"
    assert position.en_passant_target == "c6"
    assert position.castling_rights.white_kingside is True
    assert position.castling_rights.white_queenside is False
    assert position.castling_rights.black_kingside is True
    assert position.castling_rights.black_queenside is True
    assert position.fullmove_number == 2


def test_parse_fen_raises_for_invalid_structure() -> None:
    with pytest.raises(FENError, match="six fields"):
        parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0")


@pytest.mark.parametrize(
    "fen, expected_message",
    [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",
            "eight ranks",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "exactly eight squares",
        ),
        (
            "8/8/8/8/8/8/8/8 w KQkq - 0 1",
            "exactly one white king",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
            "Active color",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq a4 0 1",
            "En passant rank",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkqq - 0 1",
            "Duplicate castling",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1",
            "non-negative integer",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0",
            "positive integer",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - not_int 1",
            "halfmove clock must be an integer",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 not_int",
            "fullmove number must be an integer",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq h9 0 1",
            "En passant rank",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w G - 0 1",
            "Castling rights may only contain",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1 extra",
            "six fields",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - 0 1",
            "six fields",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1X w KQkq - 0 1",
            "Invalid piece character 'X'",
        ),
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB19 w KQkq - 0 1",
            "Invalid empty square count '9'",
        ),
    ],
)
def test_parse_fen_reports_clear_errors(fen: str, expected_message: str) -> None:
    with pytest.raises(FENError, match=expected_message):
        parse_fen(fen)
