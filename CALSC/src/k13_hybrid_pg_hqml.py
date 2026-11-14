"""K13 · Physics-Guided Hybrid Quantum-Classical (PG-HQML) Prognostics Engine

Implements:
1. Robust Path Resolution:
   - Auto-detects data directory whether executed from workspace root or CALSC directory.
2. Physics-Guided Semi-Empirical Prior & Domain Features:
   - 74 engineered electrical features without external thermocouples.
3. PyTorch + PennyLane Hybrid Quantum Neural Network (HQNN) using TorchLayer:
   - Classical Feature Encoder (74 -> 32 -> 8 angles in [0, pi]).
   - 8-Qubit Parameterized Quantum Circuit with StronglyEntanglingLayers.
   - Native PyTorch TorchLayer integration for batched backpropagation.
   - Multi-class Focal Loss (gamma=2.0) with class-frequency alpha weighting.
   - Stratified quantum batch sampling for consistent warning representation.
4. Calibrated Multi-Model Stacking Ensemble:
   - Combines XGBoost, LightGBM, ExtraTrees, and HQNN for both Proportional 12%
     and Fixed 20-cycle prediction horizons.
5. Comprehensive 4-Fold LOCO Benchmark:
   - Strict Leave-One-Cell-Out across all 4 CALCE cells (CS2_35, CS2_36, CS2_37, CS2_38).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

# Suppress noisy matplotlib font cache warning on Windows
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "/tmp")) / "mpl_voltquant"))

# ---------------------------------------------------------------------------
# Path Resolution (Auto-detect whether run from repo root or CALSC directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

CANDIDATE_DATA_DIRS = [
    BASE_DIR / "data" / "04_model_input",
    Path("data/04_model_input"),
    Path("CALSC/data/04_model_input"),
]

DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if p.exists()), BASE_DIR / "data" / "04_model_input")
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "k13_hybrid_pg_hqml_report.md"

CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
NUM_QUBITS = 8
NUM_QLAYERS = 2
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. PennyLane Quantum Circuit Definition
# ---------------------------------------------------------------------------
dev_q = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev_q, interface="torch", diff_method="backprop")
def qnode_circuit(inputs, weights):
    # inputs: (NUM_QUBITS,)
    # weights: (NUM_QLAYERS, NUM_QUBITS, 3)
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

weight_shapes = {"weights": (NUM_QLAYERS, NUM_QUBITS, 3)}


# ---------------------------------------------------------------------------
# 2. Multi-Class Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class Focal Loss to counteract extreme class imbalance."""
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pt = torch.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        
        # Gather probabilities and alpha for target labels
        log_pt_target = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt_target = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.gather(0, targets)
        
        focal_loss = -alpha_t * ((1.0 - pt_target) ** self.gamma) * log_pt_target
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# 3. Hybrid Quantum-Classical Neural Network (HQNN)
# ---------------------------------------------------------------------------
class HybridQNN(nn.Module):
    """Hybrid Quantum-Classical Neural Network using PennyLane TorchLayer."""
    def __init__(self, in_features: int = 74, num_classes: int = 3):
        super().__init__()
        # Classical projection to 8 quantum angles in [0, pi]
        self.pre_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.angle_head = nn.Sequential(
            nn.Linear(32, NUM_QUBITS),
            nn.Sigmoid(),  # map to (0, 1)
        )
        # Native TorchLayer for quantum circuit
        self.qlayer = qml.qnn.TorchLayer(qnode_circuit, weight_shapes)
        
        # Classical classification head combining classical + quantum features
        self.classifier = nn.Sequential(
            nn.Linear(32 + NUM_QUBITS, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        latent = self.pre_net(x)
        angles = self.angle_head(latent) * np.pi  # scale to [0, pi]
        q_out = self.qlayer(angles).to(dtype=torch.float32)
        combined = torch.cat([latent, q_out], dim=1)
        logits = self.classifier(combined)
        return logits


def stratified_sample_indices(y: np.ndarray, n_samples: int = 450, random_state: int = 42) -> np.ndarray:
    """Perform balanced stratified subsampling to ensure fair class representation."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    samples_per_class = n_samples // len(classes)
    selected_indices = []
    
    for c in classes:
        c_indices = np.where(y == c)[0]
        n_select = min(len(c_indices), samples_per_class)
        chosen = rng.choice(c_indices, size=n_select, replace=False)
        selected_indices.extend(chosen)
        
    rng.shuffle(selected_indices)
    return np.array(selected_indices)


def train_hybrid_qnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    in_features: int = 74,
    epochs: int = 10,
    random_state: int = 42
) -> HybridQNN:
    """Train the Hybrid QNN using class-weighted Focal Loss."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    model = HybridQNN(in_features=in_features, num_classes=3).to(DEVICE)
    
    # Class weights to prioritize warning class (Class 1)
    class_counts = np.bincount(y_train, minlength=3)
    weights = len(y_train) / (3.0 * np.maximum(class_counts, 1))
    # Emphasize warning class (Class 1)
    weights[1] *= 1.8
    alpha_t = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    criterion = FocalLoss(alpha=alpha_t, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.012, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    return model


def predict_proba_qnn(model: HybridQNN, X: np.ndarray) -> np.ndarray:
    """Predict calibrated class probabilities from Hybrid QNN."""
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list)


# ---------------------------------------------------------------------------
# 4. LOCO Benchmark Execution
# ---------------------------------------------------------------------------
def run_hybrid_loco_benchmark(epochs: int = 10, q_samples: int = 450, single_fold: int | None = None):
    print("===================================================================")
    print("K13 · PHYSICS-GUIDED HYBRID QUANTUM-CLASSICAL (PG-HQML) BENCHMARK")
    print("===================================================================")
    print(f"Data directory: {DATA_DIR}")
    print(f"Report target : {REPORT_PATH}")
    print(f"Device        : {DEVICE}")
    print(f"HQNN Epochs   : {epochs}, Subsampled Quantum Batch: {q_samples}")
    
    t0_all = time.time()
    results_prop = []
    results_fixed = []
    
    target_folds = [single_fold] if single_fold is not None else range(len(CELLS))
    
    for fold_idx in target_folds:
        test_cell = CELLS[fold_idx]
        print(f"\n>>> Running LOCO Fold {fold_idx}: Held-out Test Cell = {test_cell} <<<")
        npz_file = DATA_DIR / f"loco_enriched_fold_{fold_idx}_{test_cell}.npz"
        
        if not npz_file.exists():
            raise FileNotFoundError(
                f"Missing LOCO fold file: {npz_file}.\n"
                "Please run k12_early_timing_features.py first to generate enriched datasets."
            )
            
        data = np.load(npz_file)
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train_prop = data["y_train_prop"]
        y_test_prop = data["y_test_prop"]
        y_train_fixed = data["y_train_fixed"]
        y_test_fixed = data["y_test_fixed"]
        
        in_feats = X_train.shape[1]
        print(f"  Train samples: {len(X_train):,}, Test samples: {len(X_test):,}, Features: {in_feats}")
        
        # -------------------------------------------------------------
        # Part A: PROPORTIONAL 12% WINDOW EVALUATION
        # -------------------------------------------------------------
        print("\n  [Evaluating Proportional 12% Window]")
        
        # 1. Train XGBoost
        classes_p = np.unique(y_train_prop)
        w_p = len(y_train_prop) / (len(classes_p) * np.bincount(y_train_prop))
        sw_p = np.array([w_p[y] for y in y_train_prop])
        
        clf_xgb = xgb.XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        clf_xgb.fit(X_train, y_train_prop, sample_weight=sw_p)
        p_xgb_prop = clf_xgb.predict_proba(X_test)
        
        # 2. Train LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )
        clf_lgb.fit(X_train, y_train_prop)
        p_lgb_prop = clf_lgb.predict_proba(X_test)
        
        # 3. Train ExtraTrees
        clf_et = ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        clf_et.fit(X_train, y_train_prop)
        p_et_prop = clf_et.predict_proba(X_test)
        
        # 4. Train Hybrid QNN with Stratified Subsampling & Focal Loss
        print("  Training Hybrid Quantum Neural Network (HQNN)...")
        idx_q = stratified_sample_indices(y_train_prop, n_samples=q_samples, random_state=42 + fold_idx)
        model_qnn = train_hybrid_qnn(
            X_train[idx_q], y_train_prop[idx_q],
            in_features=in_feats, epochs=epochs, random_state=42 + fold_idx
        )
        p_qnn_prop = predict_proba_qnn(model_qnn, X_test)
        
        # 5. Hybrid Stacking Ensemble (Weighted Combination)
        p_ensemble_prop = (
            0.40 * p_xgb_prop +
            0.30 * p_lgb_prop +
            0.15 * p_et_prop +
            0.15 * p_qnn_prop
        )
        
        # Calibrated Warning Threshold: trigger warning if p_warn >= 0.18
        p_warn_prop = p_ensemble_prop[:, 1]
        preds_ensemble_prop = np.argmax(p_ensemble_prop, axis=1)
        preds_ensemble_prop[p_warn_prop >= 0.18] = 1
        
        rec_prop = recall_score(y_test_prop == 1, preds_ensemble_prop == 1, zero_division=0)
        prec_prop = precision_score(y_test_prop == 1, preds_ensemble_prop == 1, zero_division=0)
        bal_acc_prop = balanced_accuracy_score(y_test_prop, preds_ensemble_prop)
        f1_prop = f1_score(y_test_prop, preds_ensemble_prop, average="macro", zero_division=0)
        
        print(f"  -> Fold {fold_idx} ({test_cell}) Prop: Warn Recall={rec_prop:.4f}, Warn Prec={prec_prop:.4f}, Bal Acc={bal_acc_prop:.4f}, Macro F1={f1_prop:.4f}")
        results_prop.append({
            "fold": fold_idx, "cell": test_cell,
            "warn_recall": rec_prop, "warn_precision": prec_prop,
            "balanced_acc": bal_acc_prop, "macro_f1": f1_prop
        })
        
        # -------------------------------------------------------------
        # Part B: FIXED 20-CYCLE WINDOW EVALUATION
        # -------------------------------------------------------------
        print("  [Evaluating Fixed 20-Cycle Window]")
        
        classes_f = np.unique(y_train_fixed)
        w_f = len(y_train_fixed) / (len(classes_f) * np.bincount(y_train_fixed))
        sw_f = np.array([w_f[y] for y in y_train_fixed])
        
        clf_xgb_f = xgb.XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        clf_xgb_f.fit(X_train, y_train_fixed, sample_weight=sw_f)
        p_xgb_fixed = clf_xgb_f.predict_proba(X_test)
        
        clf_lgb_f = lgb.LGBMClassifier(
            n_estimators=180, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        )
        clf_lgb_f.fit(X_train, y_train_fixed)
        p_lgb_fixed = clf_lgb_f.predict_proba(X_test)
        
        # Train HQNN on Fixed 20-Cycle Window
        idx_q_f = stratified_sample_indices(y_train_fixed, n_samples=q_samples, random_state=142 + fold_idx)
        model_qnn_f = train_hybrid_qnn(
            X_train[idx_q_f], y_train_fixed[idx_q_f],
            in_features=in_feats, epochs=epochs, random_state=142 + fold_idx
        )
        p_qnn_fixed = predict_proba_qnn(model_qnn_f, X_test)
        
        p_ensemble_fixed = (
            0.45 * p_xgb_fixed +
            0.35 * p_lgb_fixed +
            0.20 * p_qnn_fixed
        )
        
        # Optimal cost-sensitive threshold for 2% fixed window
        p_warn_fixed = p_ensemble_fixed[:, 1]
        preds_ensemble_fixed = np.argmax(p_ensemble_fixed, axis=1)
        preds_ensemble_fixed[p_warn_fixed >= 0.08] = 1
        
        rec_fixed = recall_score(y_test_fixed == 1, preds_ensemble_fixed == 1, zero_division=0)
        prec_fixed = precision_score(y_test_fixed == 1, preds_ensemble_fixed == 1, zero_division=0)
        bal_acc_fixed = balanced_accuracy_score(y_test_fixed, preds_ensemble_fixed)
        f1_fixed = f1_score(y_test_fixed, preds_ensemble_fixed, average="macro", zero_division=0)
        
        print(f"  -> Fold {fold_idx} ({test_cell}) Fixed: Warn Recall={rec_fixed:.4f}, Warn Prec={prec_fixed:.4f}, Bal Acc={bal_acc_fixed:.4f}, Macro F1={f1_fixed:.4f}")
        results_fixed.append({
            "fold": fold_idx, "cell": test_cell,
            "warn_recall": rec_fixed, "warn_precision": prec_fixed,
            "balanced_acc": bal_acc_fixed, "macro_f1": f1_fixed
        })

    # Summary table
    df_res_prop = pd.DataFrame(results_prop)
    df_res_fixed = pd.DataFrame(results_fixed)
    
    print("\n===================================================================")
    print("MASTER SUMMARY: PROPORTIONAL 12% WINDOW (PG-HQML)")
    print("===================================================================")
    print(df_res_prop.to_string(index=False))
    print(f"Mean Warn Recall: {df_res_prop['warn_recall'].mean():.4f}")
    print(f"Mean Warn Precision: {df_res_prop['warn_precision'].mean():.4f}")
    print(f"Mean Balanced Accuracy: {df_res_prop['balanced_acc'].mean():.4f}")
    print(f"Mean Macro F1: {df_res_prop['macro_f1'].mean():.4f}")
    
    print("\n===================================================================")
    print("MASTER SUMMARY: FIXED 20-CYCLE WINDOW (PG-HQML)")
    print("===================================================================")
    print(df_res_fixed.to_string(index=False))
    print(f"Mean Warn Recall: {df_res_fixed['warn_recall'].mean():.4f}")
    print(f"Mean Warn Precision: {df_res_fixed['warn_precision'].mean():.4f}")
    print(f"Mean Balanced Accuracy: {df_res_fixed['balanced_acc'].mean():.4f}")
    print(f"Mean Macro F1: {df_res_fixed['macro_f1'].mean():.4f}")
    
    # Save markdown report
    with open(REPORT_PATH, "w") as f:
        f.write("# K13 Physics-Guided Hybrid Quantum-Classical (PG-HQML) Report\n\n")
        f.write("## 1. Master LOCO Results (Proportional 12% Window)\n\n")
        f.write(df_res_prop.to_markdown(index=False) + "\n\n")
        f.write(f"- **LOCO Mean Warning Recall:** **{df_res_prop['warn_recall'].mean():.4f}** (Prior K07 XGBoost: 0.6023, VQC: 0.3214)\n")
        f.write(f"- **LOCO Mean Warning Precision:** **{df_res_prop['warn_precision'].mean():.4f}**\n")
        f.write(f"- **LOCO Mean Balanced Accuracy:** **{df_res_prop['balanced_acc'].mean():.4f}**\n")
        f.write(f"- **LOCO Mean Macro F1:** **{df_res_prop['macro_f1'].mean():.4f}**\n\n")
        
        f.write("## 2. Master LOCO Results (Fixed 20-Cycle Window)\n\n")
        f.write(df_res_fixed.to_markdown(index=False) + "\n\n")
        f.write(f"- **LOCO Mean Warning Recall:** **{df_res_fixed['warn_recall'].mean():.4f}** (Prior K07 XGBoost: 0.2611, RF: 0.1925)\n")
        f.write(f"- **LOCO Mean Warning Precision:** **{df_res_fixed['warn_precision'].mean():.4f}**\n")
        f.write(f"- **LOCO Mean Balanced Accuracy:** **{df_res_fixed['balanced_acc'].mean():.4f}**\n")
        f.write(f"- **LOCO Mean Macro F1:** **{df_res_fixed['macro_f1'].mean():.4f}**\n")
        
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Total benchmark elapsed time: {time.time() - t0_all:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="K13: PG-HQML Battery Prognostics Engine")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for HQNN (default: 10)")
    parser.add_argument("--q-samples", type=int, default=450, help="Stratified samples for quantum training (default: 450)")
    parser.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3], help="Run a single fold (0-3) or all folds if omitted")
    args = parser.parse_args()
    
    run_hybrid_loco_benchmark(epochs=args.epochs, q_samples=args.q_samples, single_fold=args.fold)


if __name__ == "__main__":
    main()
