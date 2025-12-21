from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parents[1]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def _parse_accuracy(output: str) -> Optional[float]:
    match = re.search(r"Accuracy:\s*([0-9.]+)", output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate/train a model until it reaches a target accuracy on train "
            "and (optionally) test. You can supply a fixed train dataset or let the "
            "script generate one once."
        )
    )
    parser.add_argument("model", type=Path, help="Path to the starting .nn model.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Fixed training dataset (FEN + label). If omitted, a dataset is generated once and reused.",
    )
    parser.add_argument(
        "--work-model",
        type=Path,
        default=Path("work_model.nn"),
        help="Path to the working model (will be created/overwritten).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Target accuracy to stop (applied to train and test if provided).",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Optional held-out test file; loop stops only if both train and test accuracy >= threshold.",
    )
    parser.add_argument(
        "--train-runs",
        type=int,
        default=2,
        help="Number of training passes per iteration (default: 2).",
    )
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("best_model.nn"),
        help="Path to save the best model (highest test accuracy).",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Samples per class if generating a dataset (default: 300).",
    )
    parser.add_argument(
        "--min-plies",
        type=int,
        default=6,
        help="Minimum plies per random game when generating (default: 6).",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=200,
        help="Maximum plies per random game when generating (default: 200).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100000,
        help="Maximum random games per dataset generation (default: 100000).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/auto"),
        help="Directory to store generated datasets.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum eval/train cycles (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for reproducibility (used if generation is needed).",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        return 84
    if args.threshold <= 0 or args.threshold > 1:
        print("Error: --threshold must be in (0, 1].", file=sys.stderr)
        return 84
    if args.per_class <= 0:
        print("Error: --per-class must be positive.", file=sys.stderr)
        return 84
    if args.train_runs <= 0:
        print("Error: --train-runs must be positive.", file=sys.stderr)
        return 84

    rng = random.Random(args.seed)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.model.resolve() != args.work_model.resolve():
        shutil.copyfile(args.model, args.work_model)

    # Resolve training dataset (provided or generated once)
    if args.train_path is not None:
        dataset_path = args.train_path
        if not dataset_path.exists():
            print(f"Error: train dataset not found: {dataset_path}", file=sys.stderr)
            return 84
    else:
        seed = rng.randint(0, 2_000_000_000)
        dataset_path = args.dataset_dir / f"auto_{seed}.txt"
        gen_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "generate_dataset.py"),
            "--per-class",
            str(args.per_class),
            "--min-plies",
            str(args.min_plies),
            "--max-plies",
            str(args.max_plies),
            "--max-games",
            str(args.max_games),
            "--seed",
            str(seed),
            str(dataset_path),
        ]
        gen_proc = _run(gen_cmd)
        if gen_proc.returncode != 0:
            print(gen_proc.stderr, file=sys.stderr)
            return 84
        print(gen_proc.stdout.strip())

    best_accuracy = -1.0
    best_test_accuracy = -1.0
    for iteration in range(1, args.max_iterations + 1):
        acc_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "compute_accuracy.py"),
            str(args.work_model),
            str(dataset_path),
        ]
        acc_proc = _run(acc_cmd)
        if acc_proc.returncode != 0:
            print(acc_proc.stderr, file=sys.stderr)
            return 84
        accuracy = _parse_accuracy(acc_proc.stdout)
        if accuracy is None:
            print("Error: could not parse accuracy output.", file=sys.stderr)
            return 84
        best_accuracy = max(best_accuracy, accuracy)
        print(f"[{iteration}] Accuracy on generated set: {accuracy:.4f}")

        test_accuracy = None
        if args.test_path is not None:
            test_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "compute_accuracy.py"),
                str(args.work_model),
                str(args.test_path),
            ]
            test_proc = _run(test_cmd)
            if test_proc.returncode != 0:
                print(test_proc.stderr, file=sys.stderr)
                return 84
            test_accuracy = _parse_accuracy(test_proc.stdout)
            if test_accuracy is None:
                print("Error: could not parse test accuracy output.", file=sys.stderr)
                return 84
            best_test_accuracy = max(best_test_accuracy, test_accuracy)
            print(f"[{iteration}] Accuracy on test set: {test_accuracy:.4f}")

        meets_generated = accuracy >= args.threshold
        meets_test = (
            test_accuracy is None or test_accuracy >= args.threshold
        )
        if meets_generated and meets_test:
            print(
                f"Reached threshold {args.threshold:.2f} on generated "
                f"({accuracy:.4f})"
                + (
                    f" and test ({test_accuracy:.4f})"
                    if test_accuracy is not None
                    else ""
                )
                + f". Model: {args.work_model}"
            )
            return 0

        # Track best model based on test accuracy when available, otherwise train accuracy.
        if test_accuracy is not None and test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            shutil.copyfile(args.work_model, args.best_model)
        elif test_accuracy is None and accuracy > best_accuracy:
            shutil.copyfile(args.work_model, args.best_model)

        print(
            f"Below threshold, training {args.train_runs} time(s) on dataset "
            f"{dataset_path}..."
        )
        for _ in range(args.train_runs):
            train_cmd = [
                sys.executable,
                "-m",
                "my_torch_analyzer",
                "--train",
                "--optimizer",
                "adamw",
                "--augment-mirror",
                "--stratify",
            ]
            if args.seed is not None:
                train_cmd.extend(["--seed", str(args.seed)])
            train_cmd.extend(
                [
                    "--save",
                    str(args.work_model),
                    str(args.work_model),
                    str(dataset_path),
                ]
            )
            train_proc = _run(train_cmd)
            if train_proc.returncode != 0:
                print(train_proc.stderr, file=sys.stderr)
                return 84
        print("Training step done, regenerating dataset...")

    print(
        f"Max iterations reached without hitting threshold {args.threshold:.2f}. "
        f"Best accuracy (train): {best_accuracy:.4f}"
        + (
            f", Best accuracy (test): {best_test_accuracy:.4f}"
            if args.test_path is not None
            else ""
        )
        + f". Best model saved (if any improvement) to: {args.best_model}"
    )
    return 84


if __name__ == "__main__":
    sys.exit(main())
