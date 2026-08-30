"""VoltQuant HQML Final Evaluation Runner.

Loads the scaled test dataset and the three trained models (RF, LSTM, VQC),
runs inference, aggregates predictions, and saves final metrics.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from vq_core.paths import PROJECT_ROOT
from vq_core.config import PRIMARY_SEED
from vq_core.features import FULL_FEATURES, VQC_FEATURES
from vq_core.evaluation import save_evaluation_results

DATA_DIR = PROJECT_ROOT / "data" / "04_model_input"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "exp_hqml_final"


def load_test_data() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    test_path = DATA_DIR / "scaled_test.npz"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Scaled test dataset not found: {test_path}. "
            "Please run the preprocessing pipeline first."
        )
    data = np.load(test_path, allow_pickle=True)
    return (
        data["X"],
        data["y"],
        list(data["feature_columns"]),
        data["battery_id"],
        data["discharge_cycle_index"]
    )


def evaluate_rf(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    battery_id: np.ndarray,
    discharge_cycle_index: np.ndarray
) -> pd.DataFrame:
    import joblib
    model_path = MODELS_DIR / "vq_classical_rf.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Random Forest model not found at: {model_path}")
        
    rf = joblib.load(model_path)
    
    # Preprocess (select 22 FULL_FEATURES)
    indices = [feature_columns.index(feat) for feat in FULL_FEATURES]
    X_test_clean = X_test[:, :, indices]
    X_test_flat = X_test_clean.reshape(X_test_clean.shape[0], -1)
    
    preds = rf.predict(X_test_flat)
    probs = rf.predict_proba(X_test_flat)
    
    df = pd.DataFrame({
        "battery_id": battery_id,
        "discharge_cycle_index": discharge_cycle_index,
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1],
        "probability_class_2": probs[:, 2],
    })
    return df


def evaluate_lstm(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    battery_id: np.ndarray,
    discharge_cycle_index: np.ndarray
) -> pd.DataFrame:
    import tensorflow as tf
    model_path = MODELS_DIR / "vq_classical_lstm.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM model not found at: {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess (select 22 FULL_FEATURES)
    indices = [feature_columns.index(feat) for feat in FULL_FEATURES]
    X_test_lstm = X_test[:, :, indices]
    
    probs = model.predict(X_test_lstm, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    df = pd.DataFrame({
        "battery_id": battery_id,
        "discharge_cycle_index": discharge_cycle_index,
        "true_label": y_test,
        "predicted_label": preds,
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1],
        "probability_class_2": probs[:, 2],
    })
    return df


def evaluate_vqc(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    battery_id: np.ndarray,
    discharge_cycle_index: np.ndarray
) -> pd.DataFrame:
    from vq_quantum.model import forward, predict
    model_path = MODELS_DIR / "vq_quantum_weights.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"VQC weights not found at: {model_path}")
        
    saved = np.load(model_path, allow_pickle=True)
    params = {
        "weights_q": saved["weights_q"],
        "W_class": saved["W_class"],
        "b_class": saved["b_class"],
    }
    
    # Preprocess (mean pool VQC features)
    indices = [feature_columns.index(feat) for feat in VQC_FEATURES]
    X_test_vqc = np.mean(X_test[:, :, indices], axis=1)
    
    preds = predict(X_test_vqc, params)
    logits = forward(X_test_vqc, params)
    # Compute softmax probabilities from logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    df = pd.DataFrame({
        "battery_id": battery_id,
        "discharge_cycle_index": discharge_cycle_index,
        "true_label": y_test,
        "predicted_label": preds.astype(int),
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1],
        "probability_class_2": probs[:, 2],
    })
    return df


def main():
    print("=== VoltQuant Final Evaluation Runner ===")
    try:
        X_test, y_test, feature_columns, battery_id, discharge_cycle_index = load_test_data()
    except FileNotFoundError as e:
        print(f"\n[!] Evaluation Blocker: {e}")
        return
        
    print(f"Loaded test dataset with shape: {X_test.shape}")
    
    # 1. Random Forest
    try:
        df_rf = evaluate_rf(X_test, y_test, feature_columns, battery_id, discharge_cycle_index)
        save_evaluation_results(df_rf, RESULTS_DIR, "RF", PRIMARY_SEED)
        print("RF test evaluation complete.")
    except Exception as e:  # noqa: BLE001
        print(f"RF evaluation failed: {e}")
        
    # 2. LSTM
    try:
        df_lstm = evaluate_lstm(X_test, y_test, feature_columns, battery_id, discharge_cycle_index)
        save_evaluation_results(df_lstm, RESULTS_DIR, "LSTM", PRIMARY_SEED)
        print("LSTM test evaluation complete.")
    except Exception as e:  # noqa: BLE001
        print(f"LSTM evaluation failed: {e}")
        
    # 3. VQC
    try:
        df_vqc = evaluate_vqc(X_test, y_test, feature_columns, battery_id, discharge_cycle_index)
        save_evaluation_results(df_vqc, RESULTS_DIR, "VQC", PRIMARY_SEED)
        print("VQC test evaluation complete.")
    except Exception as e:  # noqa: BLE001
        print(f"VQC evaluation failed: {e}")


if __name__ == "__main__":
    main()
