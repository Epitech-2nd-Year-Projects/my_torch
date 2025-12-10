from pathlib import Path
from typing import cast

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


def test_save_and_load_network_wrappers(tmp_path: Path) -> None:
    from my_torch.nn_io import load_network, save_network

    rng = np.random.default_rng(42)
    layer = DenseLayer(
        in_features=5,
        out_features=2,
        activation=relu,
        rng=rng,
    )
    network = NeuralNetwork(layers=[layer])
    metadata = {"accuracy": 0.95, "epoch": 10}

    path = tmp_path / "wrapper_test.nn"
    save_network(path, network, metadata)

    loaded_network = load_network(path)

    assert isinstance(loaded_network, NeuralNetwork)
    assert len(loaded_network.layers) == 1

    loaded_layer: DenseLayer = cast(DenseLayer, loaded_network.layers[0])
    npt.assert_allclose(loaded_layer.weights, network.layers[0].weights)
    npt.assert_allclose(loaded_layer.bias, network.layers[0].bias)


def test_serialization_predictions_roundtrip(tmp_path: Path) -> None:
    """
    Ensure that a saved and loaded network produces identical predictions
    and has identical parameters to the original network.
    """
    rng = np.random.default_rng(123)

    layer1 = DenseLayer(
        in_features=5,
        out_features=8,
        activation=relu,
        weight_initializer="he",
        rng=rng,
    )
    layer2 = DenseLayer(
        in_features=8,
        out_features=3,
        activation=sigmoid,
        weight_initializer="xavier",
        rng=rng,
    )
    network = NeuralNetwork(layers=[layer1, layer2])

    for layer in network.layers:
        assert np.any(layer.weights != 0), (
            "Weights should be initialized to non-zero values"
        )

    input_data = rng.standard_normal((4, 5))
    prediction_before = network.forward(input_data)

    save_path = tmp_path / "prediction_roundtrip.nn"
    save_nn(network, save_path)

    loaded_network, _ = load_nn(save_path)

    prediction_after = loaded_network.forward(input_data)

    npt.assert_allclose(
        prediction_after,
        prediction_before,
        err_msg="Predictions should be identical after loading",
    )

    for i, (orig_layer, loaded_layer) in enumerate(
        zip(network.layers, loaded_network.layers)
    ):
        npt.assert_allclose(
            loaded_layer.weights,
            orig_layer.weights,
            err_msg=f"Layer {i} weights mismatch",
        )
        npt.assert_allclose(
            loaded_layer.bias, orig_layer.bias, err_msg=f"Layer {i} bias mismatch"
        )
