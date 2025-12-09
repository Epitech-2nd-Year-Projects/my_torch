# `.nn` file format

The `my_torch` project serializes neural networks into `.nn` files. Each `.nn`
file is a NumPy `.npz` archive composed of:

1. `metadata_json` – UTF‑8 JSON payload describing the file.
2. `layer_{i}_weights` and `layer_{i}_bias` arrays for every dense layer in the
   order they appear in the network.

## Metadata payload

```json
{
  "format": "my_torch.nn",
  "format_version": 1,
  "architecture": {
    "layers": [
      {
        "type": "dense",
        "index": 0,
        "in_features": 64,
        "out_features": 32,
        "activation": "relu",
        "activation_derivative": "relu_derivative",
        "weight_initializer": "xavier",
        "bias_initializer": "zeros"
      }
    ]
  },
  "training": {
    "epochs": 50,
    "best_validation_score": 0.84
  },
  "extras": {
    "notes": "optional user metadata"
  }
}
```

- `architecture.layers` lists every layer with type, size, activation, and
  initializer details.
- `training` is optional and stores run history (e.g., epochs, validation
  metrics).
- `extras` is optional free-form metadata copied verbatim.

When loading, the architecture is re-instantiated using the metadata while the
weights/bias arrays set the trained parameters.
