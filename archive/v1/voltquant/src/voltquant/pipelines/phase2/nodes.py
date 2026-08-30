"""
VoltQuant — Phase 2 Pipeline Nodes
GAN Augmentation + PCA Feature Extraction
"""

import pandas as pd
import numpy as np
from ctgan import CTGAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# NODE 1: CTGAN Augmentation
# ─────────────────────────────────────────────────────────────
def ctgan_augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use CTGAN to generate synthetic Risk (Label=1) samples
    until dataset is perfectly balanced 50/50.
    """
    safe_count = len(df[df["Label"] == 0])
    risk_count = len(df[df["Label"] == 1])
    needed     = safe_count - risk_count

    print(f"[ctgan_augment] Safe (0): {safe_count} | Risk (1): {risk_count}")
    print(f"[ctgan_augment] Generating {needed} synthetic Risk samples...")

    # Train CTGAN only on Risk samples
    risk_df = df[df["Label"] == 1].copy()

    FEATURE_COLS = [
        "Age", "Cycle_Count", "Temperature",
        "FastChargeRatio", "DischargeRate", "Resistance", "SOH",
        "Health", "Health_Drop",
        "Temperature_Stress", "Cycle_Stress", "Stress",
        "Label"
    ]

    ctgan = CTGAN(epochs=100, verbose=True)
    ctgan.fit(risk_df[FEATURE_COLS], discrete_columns=["Label"])

    synthetic = ctgan.sample(needed)
    synthetic["Label"] = 1  # force label to Risk

    print(f"[ctgan_augment] Synthetic samples generated: {len(synthetic)}")
    return synthetic


# ─────────────────────────────────────────────────────────────
# NODE 2: Merge Original + Synthetic
# ─────────────────────────────────────────────────────────────
def merge_augmented(original: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """
    Combine original dataset with CTGAN-generated synthetic Risk rows.
    Shuffle to mix real and synthetic samples.
    """
    merged = pd.concat([original, synthetic], ignore_index=True)
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    counts = merged["Label"].value_counts()
    total  = len(merged)
    print(f"[merge_augmented] Final dataset shape: {merged.shape}")
    print(f"  Safe  (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Risk  (1): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")
    return merged


# ─────────────────────────────────────────────────────────────
# NODE 3: PCA Feature Extraction
# ─────────────────────────────────────────────────────────────
def apply_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize features and apply PCA.
    Retain components explaining 95% of variance.
    Label column is preserved separately.
    """
    FEATURE_COLS = [
        "Age", "Cycle_Count", "Temperature",
        "FastChargeRatio", "DischargeRate", "Resistance", "SOH",
        "Health", "Health_Drop",
        "Temperature_Stress", "Cycle_Stress", "Stress",
    ]

    X      = df[FEATURE_COLS].values
    labels = df["Label"].values

    # Standardize
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA — keep 95% variance
    pca        = PCA(n_components=0.95, random_state=42)
    X_pca      = pca.fit_transform(X_scaled)
    n_components = X_pca.shape[1]

    print(f"[apply_pca] Original features : {len(FEATURE_COLS)}")
    print(f"[apply_pca] PCA components    : {n_components} (95% variance retained)")
    print(f"[apply_pca] Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%")

    # Build output dataframe
    pca_cols   = [f"PC{i+1}" for i in range(n_components)]
    df_pca     = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca["Label"] = labels

    print(f"[apply_pca] Final shape: {df_pca.shape}")
    return df_pca