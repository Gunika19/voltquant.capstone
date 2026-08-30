"""Unit tests for the classical LSTM benchmark model."""

from __future__ import annotations


def test_lstm_model_build():
    """Verify that the LSTM model builds and compiles with the correct layers."""
    # We import inside the test to prevent importing TensorFlow during pytest collection
    # if tensorflow is not installed in the current environment
    from vq_classical.model import build_lstm_model
    
    model = build_lstm_model(sequence_length=50, num_features=22, num_classes=3)
    
    assert model.name == "VoltQuant_LSTM_Benchmark"
    assert len(model.layers) == 3  # LSTM, Dropout, Dense (Input is not a layer in sequential)
    
    # Check input and output shapes
    assert model.input_shape == (None, 50, 22)
    assert model.output_shape == (None, 3)
