from pathlib import Path

import numpy as np
import numpy.testing as npt

from my_torch.activations import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
)
from my_torch.layers import DenseLayer
from my_torch.neural_network import NeuralNetwork
from my_torch.nn_io import load_nn, save_nn


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    layer1 = DenseLayer(
        in_features=3,
        out_features=4,
        activation=relu,
        activation_derivative=relu_derivative,
        rng=rng,
    )
    layer2 = DenseLayer(
        in_features=4,
        out_features=2,
        activation=sigmoid,
        activation_derivative=sigmoid_derivative,
        rng=rng,
    )
    network = NeuralNetwork(layers=[layer1, layer2])

    path = tmp_path / "roundtrip.nn"
    training_meta = {"epochs": 5, "best_validation_score": 0.91}
    extra_meta = {"label": "roundtrip"}
    save_nn(
        network,
        path,
        training_metadata=training_meta,
        extra_metadata=extra_meta,
    )

    loaded_network, metadata = load_nn(path)

    for original, restored in zip(network.parameters(), loaded_network.parameters()):
        npt.assert_allclose(restored, original)

    assert metadata.training == training_meta
    assert metadata.extras == extra_meta
    assert len(metadata.architecture["layers"]) == 2
    assert metadata.architecture["layers"][0]["in_features"] == 3
