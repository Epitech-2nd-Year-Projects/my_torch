from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random
from typing import Iterable

try:
    import chess
except ImportError as exc:  # pragma: no cover - runtime guard
    print(
        "Error: python-chess is required for dataset generation "
        "Install with 'python -m pip install python-chess'",
        file=sys.stderr,
    )
    raise SystemExit(84) from exc


Label = str


def _classify(fen: str) -> Label:
    board = chess.Board(fen=fen)
    if board.is_checkmate():
        return "Checkmate"
    if board.is_check():
        return "Check"
    return "Nothing"


def _sample_game_positions(
    *, rng: Random, min_plies: int, max_plies: int
) -> Iterable[str]:
    board = chess.Board()
    plies = rng.randint(min_plies, max_plies)
    for _ in range(plies):
        if board.is_game_over():
            yield board.fen()
            return
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        yield board.fen()


def _generate_balanced(
    per_class: int,
    *,
    rng: Random,
    min_plies: int,
    max_plies: int,
    max_games: int,
) -> list[tuple[str, Label]]:
    buckets: dict[Label, set[str]] = {
        "Check": set(),
        "Checkmate": set(),
        "Nothing": set(),
    }
    games_played = 0
    while any(len(bucket) < per_class for bucket in buckets.values()):
        if games_played >= max_games:
            raise RuntimeError(
                f"Could not reach {per_class} samples per class within "
                f"{max_games} games"
            )
        games_played += 1
        for fen in _sample_game_positions(
            rng=rng, min_plies=min_plies, max_plies=max_plies
        ):
            label = _classify(fen)
            bucket = buckets[label]
            if len(bucket) < per_class:
                bucket.add(fen)
            if all(len(b) >= per_class for b in buckets.values()):
                break
    entries = [(fen, label) for label, fens in buckets.items() for fen in fens]
    rng.shuffle(entries)
    return entries


def write_dataset(entries: list[tuple[str, Label]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for fen, label in entries:
            handle.write(f"{fen} {label}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic chess dataset labeled as "
            "Nothing/Check/Checkmate"
        )
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=500,
        help="Number of samples to generate per class (default: 500)",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=4,
        help="Minimum number of plies to play in a random game (default: 4)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=100,
        help="Maximum number of plies to play in a random game (default: 100)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=20000,
        help="Safety cap on random games simulated (default: 20000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the generated dataset file",
    )
    args = parser.parse_args()

    if args.per_class <= 0:
        print("Error: --per-class must be positive", file=sys.stderr)
        return 84
    if args.min_plies <= 0 or args.max_plies < args.min_plies:
        print("Error: plies must satisfy 0 < min_plies <= max_plies", file=sys.stderr)
        return 84
    if args.max_games <= 0:
        print("Error: --max-games must be positive", file=sys.stderr)
        return 84

    rng = Random(args.seed)
    try:
        entries = _generate_balanced(
            args.per_class,
            rng=rng,
            min_plies=args.min_plies,
            max_plies=args.max_plies,
            max_games=args.max_games,
        )
        write_dataset(entries, args.output)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Error: {exc}", file=sys.stderr)
        return 84

    print(
        f"Generated {len(entries)} positions "
        f"({args.per_class} per class) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
