"""Training and Evaluation Script for a Classical Random Forest Benchmark.

Loads the scaled dataset, flattens sequence windows, trains a balanced
Random Forest classifier, and reports metrics.
"""

from __future__ import annotations
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from vq_core.features import FULL_FEATURES
from vq_core.paths import PROJECT_ROOT
from vq_classical.train import load_dataset, preprocess_data

# Setup directories
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_rf():
    """Train and evaluate the Random Forest benchmark classifier."""
    X_train, y_train, X_val, y_val, feature_columns = load_dataset()
    
    # Preprocess sequence windows (extract 22 FULL_FEATURES)
    X_train_clean = preprocess_data(X_train, feature_columns)
    X_val_clean = preprocess_data(X_val, feature_columns)
    
    # Flatten sequence dimensions from (num_samples, 50, 22) to (num_samples, 1100)
    X_train_flat = X_train_clean.reshape(X_train_clean.shape[0], -1)
    X_val_flat = X_val_clean.reshape(X_val_clean.shape[0], -1)
    
    print(f"Random Forest Input Dimensions:")
    print(f"  Train: {X_train_flat.shape}")
    print(f"  Val:   {X_val_flat.shape}")
    
    # Initialize Random Forest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining Random Forest model...")
    rf.fit(X_train_flat, y_train)
    
    # Predictions and metrics
    val_preds = rf.predict(X_val_flat)
    
    accuracy = accuracy_score(y_val, val_preds)
    macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    runaway_recall = recall_score(y_val, val_preds, labels=[2], average="macro", zero_division=0)
    
    print("\n=== Final Validation Metrics (Random Forest) ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Runaway Recall (Class 2): {runaway_recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=["Normal", "Warning", "Runaway"], zero_division=0))
    
    # Save the model using joblib (since joblib is installed in .venv-classical via sklearn dependencies)
    import joblib
    model_path = MODELS_DIR / "vq_classical_rf.joblib"
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_rf()
