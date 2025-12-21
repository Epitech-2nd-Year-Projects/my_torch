import json
from typing import Any

import pytest

from my_torch.activations import identity, relu, sigmoid
from my_torch.config_loader import load_config


def test_load_valid_config(tmp_path: Any) -> None:
    config = {
        "layers": [
            {
                "type": "dense",
                "in_features": 10,
                "out_features": 5,
                "activation": "relu"
            },
            {
                "type": "dense",
                "in_features": 5,
                "out_features": 2,
                "activation": "sigmoid"
            }
        ]
    }
    
    p = tmp_path / "valid.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    
    layers = load_config(p)
    assert len(layers) == 2
    assert layers[0]["in_features"] == 10
    assert layers[0]["activation"] == relu
    assert layers[1]["activation"] == sigmoid


def test_load_config_conv_flatten_dense(tmp_path: Any) -> None:
    config = {
        "layers": [
            {
                "type": "conv2d",
                "in_channels": 1,
                "out_channels": 2,
                "kernel_size": [3, 3],
                "stride": 1,
                "padding": 1,
                "activation": "relu",
            },
            {"type": "flatten"},
            {
                "type": "dense",
                "in_features": 128,
                "out_features": 3,
                "activation": "identity",
            },
        ]
    }

    p = tmp_path / "conv_flatten.json"
    p.write_text(json.dumps(config), encoding="utf-8")

    layers = load_config(p)
    assert layers[0]["type"] == "conv2d"
    assert layers[0]["activation"] == relu
    assert layers[1]["type"] == "flatten"
    assert layers[2]["activation"] == identity

def test_load_config_defaults(tmp_path: Any) -> None:
    config = {
        "layers": [
            {
                "in_features": 10,
                "out_features": 5
            }
        ]
    }
    p = tmp_path / "defaults.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    
    layers = load_config(p)
    assert len(layers) == 1
    assert layers[0]["type"] == "dense"

def test_load_config_invalid_json(tmp_path: Any) -> None:
    p = tmp_path / "invalid.json"
    p.write_text("{invalid json", encoding="utf-8")
    
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_config(p)

def test_load_config_missing_layers(tmp_path: Any) -> None:
    p = tmp_path / "missing.json"
    p.write_text("{}", encoding="utf-8")
    
    with pytest.raises(ValueError, match="Configuration must contain a 'layers' list"):
        load_config(p)

def test_load_config_unknown_activation(tmp_path: Any) -> None:
    config = {
        "layers": [
            {
                "in_features": 10, 
                "out_features": 5, 
                "activation": "invalid_act"
            }
        ]
    }
    p = tmp_path / "bad_act.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    
    with pytest.raises(ValueError, match="Unknown activation"):
        load_config(p)
