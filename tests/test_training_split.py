import numpy as np
import pytest

from my_torch.training import train_validation_split


def test_train_validation_split_is_deterministic_with_seed() -> None:
    inputs = np.arange(20, dtype=float).reshape(10, 2)
    labels = np.arange(10, dtype=int)

    expected_rng = np.random.default_rng(7)
    expected_indices = expected_rng.permutation(inputs.shape[0])
    expected_val = expected_indices[:2]
    expected_train = expected_indices[2:]

    train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
        inputs, labels, val_ratio=0.2, seed=7
    )

    assert train_inputs.shape == (8, 2)
    assert val_inputs.shape == (2, 2)
    assert np.array_equal(train_inputs, inputs[expected_train])
    assert np.array_equal(val_inputs, inputs[expected_val])
    assert np.array_equal(train_labels, labels[expected_train])
    assert np.array_equal(val_labels, labels[expected_val])


def test_train_validation_split_without_shuffle_preserves_order() -> None:
    inputs = np.arange(15, dtype=float).reshape(5, 3)
    labels = np.arange(5, dtype=int)

    train_inputs, val_inputs, train_labels, val_labels = train_validation_split(
        inputs, labels, val_ratio=0.4, shuffle=False
    )

    assert np.array_equal(val_inputs, inputs[:2])
    assert np.array_equal(val_labels, labels[:2])
    assert np.array_equal(train_inputs, inputs[2:])
    assert np.array_equal(train_labels, labels[2:])


def test_train_validation_split_validates_ratios() -> None:
    inputs = np.arange(4, dtype=float).reshape(4, 1)
    labels = np.arange(4, dtype=int)

    with pytest.raises(ValueError):
        train_validation_split(inputs, labels, val_ratio=0.0)

    with pytest.raises(ValueError):
        train_validation_split(inputs, labels, val_ratio=0.01)

    with pytest.raises(ValueError):
        train_validation_split(inputs, labels, val_ratio=1.0)


def test_train_validation_split_rejects_seed_and_rng_combo() -> None:
    inputs = np.arange(6, dtype=float).reshape(3, 2)
    labels = np.arange(3, dtype=int)

    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        train_validation_split(inputs, labels, val_ratio=0.5, seed=1, rng=rng)
