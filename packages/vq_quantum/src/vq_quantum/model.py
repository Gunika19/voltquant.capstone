"""Hybrid Quantum-Classical Variational Classifier (VQC) Model.

Implements an 8-qubit variational quantum circuit for 3-class classification
of battery health/runaway status.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

# Define device with 8 qubits (matching our 8 VQC_FEATURES)
num_qubits = 8
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="autograd")
def quantum_circuit(features, weights):
    """8-qubit Variational Quantum Circuit.
    
    Args:
        features: Shape (batch_size, 8) or (8,)
        weights: Parameters for StronglyEntanglingLayers, shape (layers, 8, 3)
    """
    # Angle Embedding: encodes 8 normalized features into 8 qubits
    qml.AngleEmbedding(features, wires=range(num_qubits), rotation="X")
    
    # Variational layers (parameterized entangling layers)
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    
    # Measure expectation value of PauliZ on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]


def init_weights(num_layers: int = 3, seed: int = 42) -> dict[str, np.ndarray]:
    """Initialize weights for the quantum and classical layers.
    
    Args:
        num_layers: Number of strongly entangling layers in the circuit.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    
    # Weights for StronglyEntanglingLayers: (layers, num_qubits, 3)
    # Uniformly distributed in [0, 2*pi]
    weights_q = np.tensor(
        rng.uniform(low=0.0, high=2 * np.pi, size=(num_layers, num_qubits, 3)), 
        requires_grad=True
    )
    
    # Classical weights mapping 8 expectation values to 3 classes (normal, warning, runaway)
    # He initialization style
    W_class = np.tensor(
        rng.normal(loc=0.0, scale=np.sqrt(2.0 / num_qubits), size=(num_qubits, 3)), 
        requires_grad=True
    )
    b_class = np.tensor(np.zeros(3), requires_grad=True)
    
    return {
        "weights_q": weights_q,
        "W_class": W_class,
        "b_class": b_class,
    }


def forward(X_batch: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass of the hybrid model.
    
    Args:
        X_batch: Input batch of shape (batch_size, 8)
        params: Dict containing 'weights_q', 'W_class', 'b_class'
    """
    # X_batch is expected to be a 2D array of shape (batch_size, 8)
    q_out = quantum_circuit(X_batch, params["weights_q"])
    
    # q_out is a list of 8 arrays of shape (batch_size,)
    # Stack them to shape (batch_size, 8)
    q_out = np.stack(q_out, axis=1)
    
    # Linear projection to class scores (logits)
    logits = np.dot(q_out, params["W_class"]) + params["b_class"]
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(
    params: dict[str, np.ndarray], 
    X: np.ndarray, 
    y: np.ndarray,
    class_weights: np.ndarray | None = None
) -> float:
    """Softmax cross-entropy loss function.
    
    Args:
        params: Dict of parameters.
        X: Input array of shape (batch_size, 8)
        y: Target class indices of shape (batch_size,), values in {0, 1, 2}
        class_weights: Optional weights for each class of shape (3,)
    """
    logits = forward(X, params)
    probs = softmax(logits)
    
    # Compute negative log likelihood
    batch_size = len(y)
    nll = -np.log(probs[np.arange(batch_size), y] + 1e-15)
    
    if class_weights is not None:
        weighted_nll = nll * class_weights[y]
        loss = np.sum(weighted_nll) / np.sum(class_weights[y])
    else:
        loss = np.sum(nll) / batch_size
    return loss


def predict(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Predict class labels for the inputs.
    
    Args:
        X: Input array of shape (N, 8)
        params: Dict of parameters.
    """
    logits = forward(X, params)
    return np.argmax(logits, axis=1)
