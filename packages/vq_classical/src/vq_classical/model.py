"""Classical LSTM Benchmark Model.

Implements an LSTM-based recurrent neural network using TensorFlow Keras
to classify sequences of battery telemetry into 3 classes.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(
    sequence_length: int = 50, 
    num_features: int = 22, 
    num_classes: int = 3
) -> models.Sequential:
    """Build a recurrent LSTM classification network.
    
    Args:
        sequence_length: Length of sequence window (50 timesteps).
        num_features: Number of input features per timestep (22 features).
        num_classes: Number of classification targets (3 classes).
    """
    model = models.Sequential([
        # Input layer specifying sequence shape
        layers.Input(shape=(sequence_length, num_features)),
        
        # Single recurrent layer with 16 units to keep baseline capacity in check
        layers.LSTM(16, return_sequences=False, name="lstm_1"),
        layers.Dropout(0.3, name="dropout_1"),
        
        # Softmax classifier output layer
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ], name="VoltQuant_LSTM_Benchmark")
    
    return model


if __name__ == "__main__":
    # Print model summary to verify compilation
    model = build_lstm_model()
    model.summary()
