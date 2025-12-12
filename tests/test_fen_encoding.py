import numpy as np

from my_torch_analyzer.fen import fen_to_tensor, parse_fen


def test_fen_to_tensor_shape():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen)
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32


def test_fen_to_tensor_pieces_start_pos():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen)

    assert tensor[3, 7, 0] == 1.0
    assert tensor[3, 7, 7] == 1.0

    assert tensor[11, 0, 4] == 1.0

    assert tensor[0, 3, 3] == 0.0


def test_fen_to_tensor_active_color():
    fen_w = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    tensor_w = fen_to_tensor(fen_w)
    assert np.all(tensor_w[17] == 1.0)

    fen_b = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    tensor_b = fen_to_tensor(fen_b)
    assert np.all(tensor_b[17] == 0.0)


def test_fen_to_tensor_castling():
    fen_all = "4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1"
    tensor_all = fen_to_tensor(fen_all)
    assert np.all(tensor_all[13] == 1.0)
    assert np.all(tensor_all[14] == 1.0)
    assert np.all(tensor_all[15] == 1.0)
    assert np.all(tensor_all[16] == 1.0)

    fen_partial = "4k3/8/8/8/8/8/8/4K3 w Kq - 0 1"
    tensor_partial = fen_to_tensor(fen_partial)
    assert np.all(tensor_partial[13] == 1.0)
    assert np.all(tensor_partial[14] == 0.0)
    assert np.all(tensor_partial[15] == 0.0)
    assert np.all(tensor_partial[16] == 1.0)

    fen_none = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    tensor_none = fen_to_tensor(fen_none)
    assert np.all(tensor_none[13:17] == 0.0)


def test_fen_to_tensor_en_passant():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1"
    tensor = fen_to_tensor(fen)

    assert tensor[12, 5, 4] == 1.0
    assert np.sum(tensor[12]) == 1.0


def test_fen_to_tensor_from_object():
    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    position = parse_fen(fen_str)
    tensor = fen_to_tensor(position)
    assert tensor.shape == (18, 8, 8)
    assert tensor[5, 7, 4] == 1.0


def test_fen_to_tensor_consistency() -> None:
    fen = "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    tensor = fen_to_tensor(fen)

    assert tensor[1, 5, 5] == 1.0
    assert tensor[6, 3, 2] == 1.0
    assert tensor[5, 7, 4] == 1.0
    assert tensor[11, 0, 4] == 1.0

    assert np.sum(tensor[0:12, 6, 4]) == 0.0

    assert np.all(tensor[17] == 1.0)

    assert np.all(tensor[13:17] == 1.0)


def test_fen_to_tensor_sparse() -> None:
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    tensor = fen_to_tensor(fen)

    assert np.sum(tensor[0:12]) == 2.0

    assert np.sum(tensor[12]) == 0.0

    assert np.sum(tensor[13:17]) == 0.0
