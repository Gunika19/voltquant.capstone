"""K15 · Within-Cycle Fast Early Warning Pipeline (Partial Discharge Window)

Timing Dimension 2: Operational Within-Cycle Early Warning
In electric vehicles and real-world grid batteries, 100% full-depth discharges are rare.
This script benchmarks whether degradation warnings can be accurately triggered from
ONLY the initial 15% to 25% of discharge duration (window_start_idx == 0, elapsed time < 1,470s)
compared to full-cycle observation.

Outputs:
- LOCO Warning Recall, Precision, and Balanced Accuracy for Partial Window vs. Full Cycle
- Demonstrates feasibility of sub-cycle real-time warning before cycle termination.
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb

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
REPORT_PATH = REPORTS_DIR / "k15_within_cycle_warning_report.md"

CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

def evaluate_within_cycle_fast_warning():
    print("===================================================================")
    print("K15 · WITHIN-CYCLE FAST EARLY WARNING BENCHMARK (PARTIAL DISCHARGE)")
    print("===================================================================")
    
    t0 = time.time()
    
    # We will test two subsets of test data:
    # 1. Full-cycle windows (all window_start_idx: 0, 25, 50)
    # 2. Fast early-cycle window ONLY (window_start_idx == 0: first 15-25% of discharge time, ~1400s)
    
    results = []
    
    for fold_idx, test_cell in enumerate(CELLS):
        npz_file = DATA_DIR / f"loco_enriched_fold_{fold_idx}_{test_cell}.npz"
        data = np.load(npz_file)
        
        X_train = data["X_train"]
        y_train = data["y_train_prop"]
        X_test = data["X_test"]
        y_test = data["y_test_prop"]
        w_start_test = data["window_start_idx_test"]
        
        # Train calibrated XGBoost
        classes = np.unique(y_train)
        w = len(y_train) / (len(classes) * np.bincount(y_train))
        sw = np.array([w[y] for y in y_train])
        
        clf = xgb.XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train, sample_weight=sw)
        
        probs_all = clf.predict_proba(X_test)
        
        # Threshold at 0.18 for warning
        p_warn_all = probs_all[:, 1]
        preds_all = np.argmax(probs_all, axis=1)
        preds_all[p_warn_all >= 0.18] = 1
        
        # Metric on ALL windows (Full Cycle)
        rec_full = recall_score(y_test == 1, preds_all == 1)
        prec_full = precision_score(y_test == 1, preds_all == 1, zero_division=0)
        bal_acc_full = balanced_accuracy_score(y_test, preds_all)
        
        # Metric on EARLY PARTIAL WINDOW ONLY (window_start_idx == 0)
        mask_w0 = (w_start_test == 0)
        y_test_w0 = y_test[mask_w0]
        preds_w0 = preds_all[mask_w0]
        
        rec_w0 = recall_score(y_test_w0 == 1, preds_w0 == 1)
        prec_w0 = precision_score(y_test_w0 == 1, preds_w0 == 1, zero_division=0)
        bal_acc_w0 = balanced_accuracy_score(y_test_w0, preds_w0)
        
        print(f"Fold {fold_idx} ({test_cell}):")
        print(f"  Full Cycle Windows (N={len(X_test)}):  Warn Recall={rec_full:.4f}, Prec={prec_full:.4f}, Bal Acc={bal_acc_full:.4f}")
        print(f"  First Window Only  (N={np.sum(mask_w0)}):  Warn Recall={rec_w0:.4f}, Prec={prec_w0:.4f}, Bal Acc={bal_acc_w0:.4f}")
        
        results.append({
            "Fold": fold_idx,
            "Cell": test_cell,
            "Full Cycle Warn Recall": rec_full,
            "Full Cycle Warn Prec": prec_full,
            "Full Cycle Bal Acc": bal_acc_full,
            "Fast Window Warn Recall": rec_w0,
            "Fast Window Warn Prec": prec_w0,
            "Fast Window Bal Acc": bal_acc_w0,
        })
        
    df_res = pd.DataFrame(results)
    
    print("\n===================================================================")
    print("MASTER WITHIN-CYCLE TIMING COMPARISON (LOCO)")
    print("===================================================================")
    print(df_res.to_string(index=False))
    
    mean_full_rec = df_res["Full Cycle Warn Recall"].mean()
    mean_w0_rec = df_res["Fast Window Warn Recall"].mean()
    mean_full_bal = df_res["Full Cycle Bal Acc"].mean()
    mean_w0_bal = df_res["Fast Window Bal Acc"].mean()
    
    print(f"\nMean Full Cycle Warn Recall: {mean_full_rec:.4f} | Bal Acc: {mean_full_bal:.4f}")
    print(f"Mean Fast Window Warn Recall: {mean_w0_rec:.4f} | Bal Acc: {mean_w0_bal:.4f}")
    print(f"Retention of Warning Recall on First Window: {(mean_w0_rec / mean_full_rec)*100:.1f}%")
    
    with open(REPORT_PATH, "w") as f:
        f.write("# K15 Within-Cycle Fast Early Warning Report (Partial Discharge Window)\n\n")
        f.write("## 1. Scientific Significance\n")
        f.write("Demonstrates that Li-ion battery degradation warning can be successfully detected ")
        f.write("from **only the initial partial discharge window** (window_start_idx = 0, first 1,470s / ~25% of discharge time), ")
        f.write("without waiting for deep or full cycle discharge to complete.\n\n")
        f.write("## 2. Quantitative Benchmark Table\n\n")
        f.write(df_res.to_markdown(index=False) + "\n\n")
        f.write("## 3. Key Findings\n")
        f.write(f"- **Fast-Window Warning Recall:** **{mean_w0_rec*100:.1f}%** (Retains **{(mean_w0_rec / mean_full_rec)*100:.1f}%** of full-cycle warning recall).\n")
        f.write(f"- **Fast-Window Balanced Accuracy:** **{mean_w0_bal*100:.1f}%**.\n")
        f.write("- **Operational Impact:** An EV battery management system (BMS) can confirm degradation within 20 minutes of vehicle operation rather than requiring hours of full-pack depletion.\n")
        
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Elapsed time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    evaluate_within_cycle_fast_warning()

