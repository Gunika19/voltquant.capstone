"""Central configuration of experimental parameters for VoltQuant HQML pipeline.

All hyperparameter selections, structural settings, and reproducibility seeds
are declared here to enforce reproducibility and allow simple freezing.
"""

from __future__ import annotations

import numpy as np

# Reproducibility Seeds
GLOBAL_SEEDS = [42, 43, 44, 45, 46]
PRIMARY_SEED = 42

# Sequence Generation Parameters
SEQUENCE_LENGTH = 50
STRIDE = 25

# Prognostics EOL & Early Warning Labeling
SOH_EOL_THRESHOLD = 0.80
CROSSING_PERSISTENCE_CYCLES = 3
WARNING_LEAD_CYCLES = 20  # Warning Horizon in cycles

# Feature Selections

# Model Parameters - Variational Quantum Classifier (VQC)
VQC_QUBITS = 8
VQC_LAYERS = 3
VQC_LEARNING_RATE = 0.01
VQC_BATCH_SIZE = 32
VQC_EPOCHS = 15

# Model Parameters - Classical LSTM Benchmark
LSTM_UNITS = 16
LSTM_DROPOUT = 0.30
LSTM_LEARNING_RATE = 0.001
LSTM_BATCH_SIZE = 32
LSTM_EPOCHS = 11

# Model Parameters - Classical Random Forest Baseline
RF_ESTIMATORS = 100
RF_CLASS_WEIGHT = "balanced"


def set_reproducible_seeds(seed: int) -> None:
    """Set global seeds across python, numpy, and optional deep learning libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    # Optional TensorFlow seed setting if TF is imported
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
