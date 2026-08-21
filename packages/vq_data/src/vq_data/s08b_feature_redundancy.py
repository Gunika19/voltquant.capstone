"""Feature redundancy diagnostic ahead of VQC qubit allocation.

Read-only. Loads data/04_model_input/scaled_{train,val}.npz and
sequence_manifest.parquet. Prints a report to stdout and writes plots to
reports/figures/. Writes no data files.

Correlation matrix, variance, and PCA all operate on the TRAIN split's
scaled features, flattened across (sequence, timestep) -- i.e. every
timestep of every training window is treated as one observation of the 22
(post s07 Time-drop) features. This is the same granularity
report_feature_correlations() in s07_build_sequences.py used for its
exact/near-duplicate scan. PCA is a from-scratch numpy implementation (no
scikit-learn dependency in envs/data.txt).

Beyond the variance argument, this script also runs a practical test: a
simple 3-class multinomial logistic regression (also from-scratch numpy --
gradient descent on the softmax cross-entropy loss, no scikit-learn), fit on
train and evaluated on val, comparing three ways of getting down to the
VQC's 8-qubit budget: PCA-8 on all 22 features, PCA-8 on the 18-feature
reduced set, and a hand-picked 8 raw features. This is a diagnostic
baseline for choosing an encoding, NOT the project's classical benchmark --
that is the LSTM in vq_classical.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vq_core.guards import assert_grouped_split
from vq_core.paths import PROJECT_ROOT
from vq_core.schema import LABELS

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_INPUT_DIR = PROJECT_ROOT / "data" / "04_model_input"
SPLIT_PATHS = {
    "train": MODEL_INPUT_DIR / "scaled_train.npz",
    "val": MODEL_INPUT_DIR / "scaled_val.npz",
}
MANIFEST_PATH = MODEL_INPUT_DIR / "sequence_manifest.parquet"

CORRELATION_THRESHOLD = 0.95
NEAR_CONSTANT_VARIANCE_THRESHOLD = 0.01
VQC_QUBIT_BUDGET = 8
EXPLAINED_VARIANCE_TARGET = 0.95

PRECURSOR_FEATURES = ["dT_dt", "dV_dt", "dI_dt", "power"]

# Rolling-mean columns s07_build_sequences.py's correlation scan flagged as
# near-duplicates of the raw measurement they smooth (temperature and
# voltage, in both the sample-count and 60s-window variants).
REDUNDANT_ROLLING_MEANS = [
    "roll_mean_temperature_samples",
    "roll_mean_temperature_60s",
    "roll_mean_voltage_samples",
    "roll_mean_voltage_60s",
]

# Hand-picked 8 for the VQC: one raw signal per physical quantity (voltage,
# current, temperature), their first derivatives for temperature and
# voltage (dT_dt, dV_dt -- the claimed precursor signals), one smoothed
# volatility measure per quantity (the 60s-window std, not the 5-sample one,
# since sampling interval varies 9.3-20.5s and 60s is the physically fixed
# window -- see s05's ROLLING_WINDOW_SECONDS comment), and time_elapsed for
# within-cycle position. Deliberately excludes: Current_load/Voltage_load
# (setpoints, not measurements -- and Current_load is the load_level
# stratum in disguise), power (r=0.988 with Current_measured -- see
# report_precursor_features), dI_dt (r=0.936 with dV_dt), temp_rise
# (temperature-derived, correlates with Temperature_measured/dT_dt),
# sampling_interval_s and ambient_temperature/load_level (protocol
# covariates, not degradation signal), and all 4 REDUNDANT_ROLLING_MEANS
# plus roll_std_*_samples (each r > 0.96 against its 60s counterpart).
HAND_PICKED_FEATURES = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "dT_dt",
    "dV_dt",
    "roll_std_temperature_60s",
    "roll_std_voltage_60s",
    "time_elapsed",
]


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_split_flat(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = SPLIT_PATHS[split]
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"]
    feature_columns = [str(c) for c in data["feature_columns"]]
    flat = X.reshape(-1, X.shape[-1])
    y_flat = np.repeat(y, X.shape[1])
    print(f"loaded {path.name}: X shape {X.shape} -> flattened {flat.shape}")
    return flat, y_flat, feature_columns


def check_no_cell_leakage() -> None:
    section("Split leakage check")
    manifest = pd.read_parquet(MANIFEST_PATH)
    assert_grouped_split(manifest, split_column="split", group_column="battery_id")
    print("assert_grouped_split: PASSED -- no battery_id appears in more than one split.")


def correlation_matrix(flat: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    section("Correlation matrix")
    corr = pd.DataFrame(np.corrcoef(flat, rowvar=False), index=feature_columns, columns=feature_columns)
    print(f"{len(feature_columns)} x {len(feature_columns)} Pearson correlation matrix computed.")
    return corr


def plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("NASA: scaled-train feature correlation matrix (flattened sequence x timestep)")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s08b_feature_correlation.png")
    plt.close(fig)
    print(f"saved plot to {FIGURES_DIR / 's08b_feature_correlation.png'}")


def report_high_correlation_pairs(corr: pd.DataFrame) -> None:
    section(f"Pairs with |r| > {CORRELATION_THRESHOLD}")
    columns = corr.columns.tolist()
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r = corr.loc[a, b]
            if abs(r) > CORRELATION_THRESHOLD:
                pairs.append((a, b, r))

    if not pairs:
        print("none.")
        return
    for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
        print(f"  {a} <-> {b}: r = {r:.6f}")


def report_variance(flat: np.ndarray, feature_columns: list[str]) -> pd.Series:
    section("Per-feature variance (scaled train)")
    variance = pd.Series(flat.var(axis=0), index=feature_columns)
    print(variance.sort_values().to_string())

    near_constant = variance[variance < NEAR_CONSTANT_VARIANCE_THRESHOLD]
    if near_constant.empty:
        print(f"\nno feature has variance below {NEAR_CONSTANT_VARIANCE_THRESHOLD} -- none are near-constant.")
    else:
        print(f"\n[FLAG] near-constant (variance < {NEAR_CONSTANT_VARIANCE_THRESHOLD}): {near_constant.index.tolist()}")
    return variance


def pca(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full PCA via SVD on mean-centred data. Returns (explained_variance_ratio, components, mean).

    components[i] is the i-th principal axis, in the original feature basis --
    same convention as sklearn.decomposition.PCA.components_. `mean` is
    returned so val (or any other split) can be projected onto axes fit on
    train only: (val_flat - mean) @ components[:k].T.
    """
    mean = flat.mean(axis=0)
    centred = flat - mean
    _, singular_values, Vt = np.linalg.svd(centred, full_matrices=False)
    explained_variance = singular_values**2 / (len(flat) - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    return explained_variance_ratio, Vt, mean


def project(flat: np.ndarray, mean: np.ndarray, components: np.ndarray, n_components: int) -> np.ndarray:
    return (flat - mean) @ components[:n_components].T


def report_pca(flat: np.ndarray, feature_columns: list[str], corr: pd.DataFrame, label: str) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    section(f"PCA: {label}")
    explained_variance_ratio, components, mean = pca(flat)
    cumulative = np.cumsum(explained_variance_ratio)

    table = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(explained_variance_ratio))],
            "explained_variance_pct": explained_variance_ratio * 100,
            "cumulative_pct": cumulative * 100,
        }
    )
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    n_for_target = int(np.searchsorted(cumulative, EXPLAINED_VARIANCE_TARGET) + 1)
    print(f"\ncomponents needed to reach {EXPLAINED_VARIANCE_TARGET:.0%} explained variance: {n_for_target}")
    print(f"explained variance at exactly {VQC_QUBIT_BUDGET} components: {cumulative[VQC_QUBIT_BUDGET - 1] * 100:.2f}%")

    n_report = min(VQC_QUBIT_BUDGET, len(explained_variance_ratio))
    print(f"\ntop 5 loadings on the first {n_report} components (qubit budget = {VQC_QUBIT_BUDGET}):")
    n_dominated = 0
    for i in range(n_report):
        loadings = pd.Series(components[i], index=feature_columns)
        ranked = loadings.abs().sort_values(ascending=False).index
        top5 = loadings.reindex(ranked[:5])
        top5_str = ", ".join(f"{name}={value:.3f}" for name, value in top5.items())

        # A component whose two largest loadings belong to a near-duplicate
        # pair (|r| > CORRELATION_THRESHOLD between them) is spending most of
        # its budget re-describing one signal rather than combining distinct
        # ones -- flag it so "more interpretable" isn't left to eyeballing.
        top_two = ranked[:2]
        pair_r = corr.loc[top_two[0], top_two[1]] if len(top_two) == 2 else 0.0
        dominated = abs(pair_r) > CORRELATION_THRESHOLD
        flag = f" [DUPLICATE-DOMINATED: {top_two[0]} & {top_two[1]}, r={pair_r:.3f}]" if dominated else ""
        n_dominated += int(dominated)
        print(f"  PC{i + 1} ({explained_variance_ratio[i] * 100:.1f}% var): {top5_str}{flag}")

    print(f"\n{n_dominated} of the first {n_report} components are duplicate-dominated (top-2 loadings pairwise |r| > {CORRELATION_THRESHOLD}).")
    return cumulative, n_dominated, components, mean


def compare_pca(
    full_cumulative: np.ndarray,
    reduced_cumulative: np.ndarray,
    full_columns: list[str],
    reduced_columns: list[str],
    full_n_dominated: int,
    reduced_n_dominated: int,
) -> None:
    section("Full vs reduced feature set: PCA comparison")
    full_n = int(np.searchsorted(full_cumulative, EXPLAINED_VARIANCE_TARGET) + 1)
    reduced_n = int(np.searchsorted(reduced_cumulative, EXPLAINED_VARIANCE_TARGET) + 1)
    print(f"full set ({len(full_columns)} features): {full_n} components reach {EXPLAINED_VARIANCE_TARGET:.0%}")
    print(f"reduced set ({len(reduced_columns)} features, dropped {REDUNDANT_ROLLING_MEANS}): {reduced_n} components reach {EXPLAINED_VARIANCE_TARGET:.0%}")

    if reduced_n < full_n:
        print(
            f"\ndropping the 4 redundant rolling-mean columns needs {full_n - reduced_n} fewer component(s) "
            f"to reach {EXPLAINED_VARIANCE_TARGET:.0%} -- consistent with those columns being near-duplicates "
            f"of Temperature_measured/Voltage_measured (see s07's correlation scan and the heatmap here) rather "
            f"than adding independent variance."
        )
    elif reduced_n > full_n:
        print(
            f"\nthe reduced set needs MORE components ({reduced_n} vs {full_n}) -- the dropped columns were "
            f"not purely redundant; some of their variance was not fully recoverable from the remaining features."
        )
    else:
        print(
            f"\nboth sets need the same number of components ({full_n}) to reach {EXPLAINED_VARIANCE_TARGET:.0%} -- "
            f"the 4 dropped columns were carrying ~zero unique variance (consistent with r > 0.998 against "
            f"Temperature_measured/Voltage_measured), so removing them shrank the feature count without "
            f"shrinking the effective dimensionality at all."
        )

    print(
        f"\nduplicate-dominated components among the first {VQC_QUBIT_BUDGET}: full set {full_n_dominated}, "
        f"reduced set {reduced_n_dominated}."
    )
    if reduced_n_dominated < full_n_dominated:
        print(
            "Dropping the 4 rolling-mean columns did make the first "
            f"{VQC_QUBIT_BUDGET} components more interpretable -- fewer of them are just one signal "
            "double-counted through two near-identical columns. But it did not fix the problem in general: "
            "roll_std_temperature_samples/roll_std_temperature_60s (r=0.965), "
            "roll_std_voltage_samples/roll_std_voltage_60s (r=0.960), and dV_dt/dI_dt (r=0.936) are still "
            "present in both sets and still co-dominate individual components (see the flags above) -- "
            "the sample-count vs 60s-window duplication pattern from s05 wasn't limited to the 4 columns "
            "dropped here."
        )
    else:
        print(
            "Dropping the 4 rolling-mean columns did not reduce how many of the first "
            f"{VQC_QUBIT_BUDGET} components are duplicate-dominated -- the remaining near-duplicate pairs "
            "(roll_std_temperature_samples/60s, roll_std_voltage_samples/60s, dV_dt/dI_dt) still drive that."
        )


def report_precursor_features(corr: pd.DataFrame, variance: pd.Series) -> None:
    section("Precursor feature check: dT_dt, dV_dt, dI_dt, power")
    for feature in PRECURSOR_FEATURES:
        print(f"\n{feature}: standalone variance = {variance[feature]:.4f}")
        row = corr[feature].drop(feature).sort_values(key=lambda s: -s.abs())
        print("  correlation with every other feature (sorted by |r|):")
        for other, r in row.items():
            print(f"    {other}: r = {r:.4f}")

    power_vs_voltage = abs(corr.loc["power", "Voltage_measured"])
    power_vs_current = abs(corr.loc["power", "Current_measured"])
    power_vs_current_load = abs(corr.loc["power", "Current_load"]) if "Current_load" in corr.columns else float("nan")
    print(
        f"\npower = Voltage_measured * Current_measured (s05_build_measurement_features.py). "
        f"|r| vs Voltage_measured = {power_vs_voltage:.4f}, vs Current_measured = {power_vs_current:.4f}, "
        f"vs Current_load = {power_vs_current_load:.4f}."
    )
    if max(power_vs_voltage, power_vs_current, power_vs_current_load) > CORRELATION_THRESHOLD:
        print(
            f"[FLAG] power is near-collinear (|r| > {CORRELATION_THRESHOLD}) with one of its own multiplicands "
            f"-- it is not contributing much information beyond that column in a linear model or a PCA/VQC "
            f"feature map."
        )
    else:
        print(
            f"power stays below the |r| > {CORRELATION_THRESHOLD} threshold against voltage and current -- "
            f"the product is not simply redundant with either factor alone (current is roughly constant "
            f"within a cycle at fixed load, so power tracks voltage's decline, but not collinearly enough "
            f"to call it a duplicate)."
        )


def report_hand_picked_features(corr: pd.DataFrame, variance: pd.Series) -> None:
    section("Hand-picked 8-feature set")
    print(f"features: {HAND_PICKED_FEATURES}")

    sub_corr = corr.loc[HAND_PICKED_FEATURES, HAND_PICKED_FEATURES].to_numpy()
    n = len(HAND_PICKED_FEATURES)
    best_r, best_a, best_b = 0.0, None, None
    for i in range(n):
        for j in range(i + 1, n):
            if abs(sub_corr[i, j]) > abs(best_r):
                best_r, best_a, best_b = sub_corr[i, j], HAND_PICKED_FEATURES[i], HAND_PICKED_FEATURES[j]
    print(f"max |r| among the 8: {abs(best_r):.4f} ({best_a} <-> {best_b}, r={best_r:.4f})")

    hand_picked_variance = variance.loc[HAND_PICKED_FEATURES].sum()
    total_variance = variance.sum()
    print(
        f"total variance of the 8 as a fraction of all 22 features' variance: "
        f"{hand_picked_variance:.3f} / {total_variance:.3f} = {hand_picked_variance / total_variance:.3f}"
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_softmax_regression(X: np.ndarray, y: np.ndarray, n_classes: int = 3, lr: float = 0.5, n_iter: int = 400, l2: float = 1e-4) -> np.ndarray:
    """Multinomial logistic regression by full-batch gradient descent on the
    softmax cross-entropy loss. No scikit-learn dependency in envs/data.txt."""
    n, d = X.shape
    X_bias = np.hstack([X, np.ones((n, 1))])
    Y_onehot = np.eye(n_classes)[y]
    W = np.zeros((d + 1, n_classes))
    for _ in range(n_iter):
        P = softmax(X_bias @ W)
        grad = X_bias.T @ (P - Y_onehot) / n + l2 * W
        W -= lr * grad
    return W


def predict_labels(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X_bias = np.hstack([X, np.ones((len(X), 1))])
    return softmax(X_bias @ W).argmax(axis=1)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> tuple[float, dict[int, float]]:
    recalls = {}
    f1s = {}
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s[c] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        recalls[c] = recall
    macro_f1 = float(np.mean(list(f1s.values())))
    return macro_f1, recalls


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def run_practical_test(
    train_flat: np.ndarray,
    train_y: np.ndarray,
    val_flat: np.ndarray,
    val_y: np.ndarray,
    feature_columns: list[str],
    full_components: np.ndarray,
    full_mean: np.ndarray,
    reduced_columns: list[str],
    reduced_components: np.ndarray,
    reduced_mean: np.ndarray,
) -> None:
    section("Practical test: 3-class logistic regression")
    print(
        "This is a diagnostic baseline for comparing feature-reduction strategies, not the "
        "project's classical benchmark -- that is the LSTM in vq_classical."
    )

    reduced_indices = [feature_columns.index(c) for c in reduced_columns]
    hand_picked_indices = [feature_columns.index(c) for c in HAND_PICKED_FEATURES]

    strategies = {
        "PCA-8 (full 22)": (
            project(train_flat, full_mean, full_components, VQC_QUBIT_BUDGET),
            project(val_flat, full_mean, full_components, VQC_QUBIT_BUDGET),
        ),
        "PCA-8 (reduced 18)": (
            project(train_flat[:, reduced_indices], reduced_mean, reduced_components, VQC_QUBIT_BUDGET),
            project(val_flat[:, reduced_indices], reduced_mean, reduced_components, VQC_QUBIT_BUDGET),
        ),
        "Hand-picked 8": (
            train_flat[:, hand_picked_indices],
            val_flat[:, hand_picked_indices],
        ),
    }

    rows = []
    for name, (X_train, X_val) in strategies.items():
        # Re-standardize each strategy's own output on TRAIN stats before
        # fitting: PCA scores have wildly different scales across components
        # (PC1's variance is a large share of the total; PC8's is small),
        # while the hand-picked raw columns are already ~unit variance from
        # s08. Matching scale across strategies keeps gradient descent's
        # fixed learning rate fair to all three, so differences in the
        # result reflect the features, not an unlucky step size.
        X_train_std, mean_, std_ = standardize_fit(X_train)
        X_val_std = (X_val - mean_) / std_

        W = fit_softmax_regression(X_train_std, train_y)
        pred = predict_labels(X_val_std, W)
        macro_f1, recall = classification_report(val_y, pred)

        recall_str = ", ".join(f"{LABELS[c]}={recall[c]:.3f}" for c in sorted(recall))
        print(f"\n{name}: macro F1 = {macro_f1:.3f}, recall: {recall_str}")
        row = {"strategy": name, "macro_f1": macro_f1}
        row.update({f"recall_{LABELS[c]}": recall[c] for c in recall})
        rows.append(row)

    table = pd.DataFrame(rows)
    section("Strategy comparison table")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = table.loc[table["macro_f1"].idxmax()]
    hand_picked_f1 = table.loc[table["strategy"] == "Hand-picked 8", "macro_f1"].iloc[0]
    gap = best["macro_f1"] - hand_picked_f1

    section("Recommendation (practical test)")
    if best["strategy"] == "Hand-picked 8":
        print("Hand-picked 8 has the best macro F1 outright -- prefer it, it also wins on interpretability.")
    elif gap <= 0.02:
        print(
            f"Hand-picked 8 (macro F1 {hand_picked_f1:.3f}) is within {gap:.3f} of the best PCA strategy "
            f"({best['strategy']}, macro F1 {best['macro_f1']:.3f}) -- within a point or two, so hand-picked "
            f"wins on interpretability at negligible accuracy cost."
        )
    else:
        print(
            f"Hand-picked 8 (macro F1 {hand_picked_f1:.3f}) trails the best PCA strategy ({best['strategy']}, "
            f"macro F1 {best['macro_f1']:.3f}) by {gap:.3f} -- more than a point or two, so PCA's extra "
            f"accuracy may be worth the interpretability cost here."
        )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    check_no_cell_leakage()

    train_flat, train_y, feature_columns = load_split_flat("train")
    val_flat, val_y, val_feature_columns = load_split_flat("val")
    assert val_feature_columns == feature_columns, "train/val feature column order mismatch"

    corr = correlation_matrix(train_flat, feature_columns)
    plot_correlation_heatmap(corr)
    report_high_correlation_pairs(corr)
    variance = report_variance(train_flat, feature_columns)

    full_cumulative, full_n_dominated, full_components, full_mean = report_pca(
        train_flat, feature_columns, corr, label=f"full feature set ({len(feature_columns)} features)"
    )

    reduced_columns = [c for c in feature_columns if c not in REDUNDANT_ROLLING_MEANS]
    reduced_indices = [feature_columns.index(c) for c in reduced_columns]
    reduced_flat = train_flat[:, reduced_indices]
    reduced_cumulative, reduced_n_dominated, reduced_components, reduced_mean = report_pca(
        reduced_flat, reduced_columns, corr, label=f"reduced feature set ({len(reduced_columns)} features, dropped {REDUNDANT_ROLLING_MEANS})"
    )

    compare_pca(full_cumulative, reduced_cumulative, feature_columns, reduced_columns, full_n_dominated, reduced_n_dominated)
    report_precursor_features(corr, variance)
    report_hand_picked_features(corr, variance)

    run_practical_test(
        train_flat,
        train_y,
        val_flat,
        val_y,
        feature_columns,
        full_components,
        full_mean,
        reduced_columns,
        reduced_components,
        reduced_mean,
    )

    section("Recommendation")
    full_n = int(np.searchsorted(full_cumulative, EXPLAINED_VARIANCE_TARGET) + 1)
    reduced_n = int(np.searchsorted(reduced_cumulative, EXPLAINED_VARIANCE_TARGET) + 1)

    within_budget = reduced_n <= VQC_QUBIT_BUDGET
    print(
        f"Reduced set (drop {REDUNDANT_ROLLING_MEANS}): {reduced_n} components for {EXPLAINED_VARIANCE_TARGET:.0%} "
        f"({'within' if within_budget else 'EXCEEDS'} the {VQC_QUBIT_BUDGET}-qubit budget), "
        f"{reduced_n_dominated} of the first {VQC_QUBIT_BUDGET} components still duplicate-dominated."
    )
    if not within_budget:
        print(
            f"Neither set reaches {EXPLAINED_VARIANCE_TARGET:.0%} within {VQC_QUBIT_BUDGET} qubits (full: {full_n} "
            f"components, reduced: {reduced_n}). Dropping the 4 exact-duplicate rolling means is still worth doing "
            f"before any PCA-based encoding -- it removes dead weight from the input without costing any real "
            f"variance -- but it is not sufficient on its own to fit this feature set's information into 8 "
            f"components at {EXPLAINED_VARIANCE_TARGET:.0%} fidelity. See the practical-test recommendation above "
            f"for how the hand-picked 8 compares against both PCA-8 strategies directly."
        )
    else:
        print(
            f"The reduced set fits the {VQC_QUBIT_BUDGET}-qubit budget at {EXPLAINED_VARIANCE_TARGET:.0%} "
            f"explained variance -- see the practical-test recommendation above for how it compares against "
            f"the hand-picked 8 directly."
        )


if __name__ == "__main__":
    main()
