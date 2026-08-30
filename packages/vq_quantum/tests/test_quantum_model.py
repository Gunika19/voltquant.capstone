"""Unit tests for the VQC quantum model."""

from __future__ import annotations
import numpy as np
from vq_quantum.model import init_weights, forward, predict


def test_vqc_initialization():
    """Verify that model weights initialize with correct shapes."""
    params = init_weights(num_layers=3, seed=42)
    
    assert "weights_q" in params
    assert "W_class" in params
    assert "b_class" in params
    
    # 3 layers, 8 qubits, 3 angles per rotation
    assert params["weights_q"].shape == (3, 8, 3)
    # 8 qubits output mapped to 3 classes
    assert params["W_class"].shape == (8, 3)
    assert params["b_class"].shape == (3,)


def test_vqc_forward_and_predict():
    """Verify forward pass and predictions with dummy inputs."""
    params = init_weights(num_layers=2, seed=123)
    
    # Batch of 4 samples, 8 features each
    X_dummy = np.random.normal(size=(4, 8))
    
    # Check forward pass shapes
    logits = forward(X_dummy, params)
    assert logits.shape == (4, 3)
    
    # Check predictions
    preds = predict(X_dummy, params)
    assert preds.shape == (4,)
    preds_list = [int(p) for p in preds]
    assert all(p in {0, 1, 2} for p in preds_list)
