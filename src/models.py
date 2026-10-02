#!/usr/bin/env python3
"""
src/models.py

Phase 2 Benchmark Execution Engine for DprE1:
Evaluates Baseline models and Machine Learning architectures across three distinct evaluation tracks:
  - Track A (Control Split): Random 5-Fold Stratified CV (demonstrates analogue memorization bias).
  - Track B (Cluster Split): Stratified Cluster 5-Fold CV (zero cluster leakage, >= 3 inactives/fold).
  - Track C (Headline Test): Leave-Hydantoin-Out (LHO) Cross-Scaffold Generalization:
      * Split 1: Train on Non-Hydantoins (N=51) -> Test on Hydantoins (N=51)
      * Split 2: Train on Hydantoins (N=51) -> Test on Non-Hydantoins (N=51)
      * Pooled LHO: Concatenated out-of-scaffold predictions (N=102)

Models Evaluated:
  1. Baseline 0: Majority Class Random Floor (P(Active) = 74/102 = 0.7255)
  2. Baseline 1: HydantoinDetectorClassifier (Confounding heuristic: P=0.84 if hydantoin else P=0.61)
  3. Baseline 2: 1-Nearest Neighbor Tanimoto similarity to training fold actives
  4. Logistic Regression: L2-penalized, 2048-bit Morgan count fingerprints
  5. Random Forest: 300 trees, min_samples_split=4, balanced subsample weighting
  6. LightGBM: Shallow boosted trees, max_depth=4, learning_rate=0.05

Statistical Framework:
  - Pooled Out-Of-Fold (OOF) evaluation
  - 1,000 bootstrap iterations yielding 95% BCa confidence intervals
  - 10-iteration Y-randomization control proving models collapse to the dummy floor

Outputs:
  reports/benchmark_phase2_results.csv
"""

import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.features import (
    compute_morgan_fingerprints,
    compute_rdkit_bit_fingerprints,
    compute_max_tanimoto_to_actives,
    is_hydantoin
)


def safe_roc_auc(y_true, y_pred):
    """Calculates ROC-AUC, falling back to 0.5 if labels or predictions are degenerate."""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.5
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5


def safe_pr_auc(y_true, y_pred):
    """Calculates PR-AUC (Average Precision), falling back to positive prevalence if degenerate."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    try:
        return average_precision_score(y_true, y_pred)
    except Exception:
        return float(np.mean(y_true))


def bootstrap_bca(y_true, y_pred, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """
    Computes 1,000-iteration bootstrap resampling with 95% Bias-Corrected and
    Accelerated (BCa) confidence intervals.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    orig_val = metric_fn(y_true, y_pred)

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_vals.append(metric_fn(y_true[idx], y_pred[idx]))

    if len(boot_vals) < 50:
        return orig_val, orig_val, 0.0, orig_val, orig_val

    boot_vals = np.array(boot_vals)

    # Jackknife for acceleration parameter (a)
    jack_vals = []
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        if len(np.unique(y_true[idx])) < 2:
            jack_vals.append(orig_val)
        else:
            jack_vals.append(metric_fn(y_true[idx], y_pred[idx]))
    jack_vals = np.array(jack_vals)
    jack_mean = np.mean(jack_vals)

    num = np.sum((jack_mean - jack_vals) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_vals) ** 2)) ** 1.5
    a = num / denom if (denom != 0 and not np.isnan(denom)) else 0.0

    # Bias correction (z0)
    prop_less = np.mean(boot_vals < orig_val)
    prop_less = np.clip(prop_less, 1e-6, 1.0 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

    z_alpha = stats.norm.ppf(alpha / 2.0)
    z_1_alpha = stats.norm.ppf(1.0 - alpha / 2.0)

    denom1 = 1.0 - a * (z0 + z_alpha)
    denom2 = 1.0 - a * (z0 + z_1_alpha)

    if abs(denom1) < 1e-5 or abs(denom2) < 1e-5 or np.isnan(a):
        # Fallback to percentile interval
        ci_lower = np.percentile(boot_vals, 100 * (alpha / 2.0))
        ci_upper = np.percentile(boot_vals, 100 * (1.0 - alpha / 2.0))
    else:
        pct1 = stats.norm.cdf(z0 + (z0 + z_alpha) / denom1) * 100
        pct2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / denom2) * 100
        pct1 = np.clip(pct1, 0.1, 99.9)
        pct2 = np.clip(pct2, 0.1, 99.9)
        ci_lower = np.percentile(boot_vals, min(pct1, pct2))
        ci_upper = np.percentile(boot_vals, max(pct1, pct2))

    return orig_val, np.mean(boot_vals), np.std(boot_vals), ci_lower, ci_upper


class HydantoinDetectorClassifier:
    """Baseline 1: Predicts empirical prior based solely on hydantoin presence."""
    def __init__(self, p_hyd=0.8431, p_non_hyd=0.6078):
        self.p_hyd = p_hyd
        self.p_non_hyd = p_non_hyd

    def predict_proba(self, is_hyd_array):
        probs = np.where(is_hyd_array, self.p_hyd, self.p_non_hyd)
        return probs


def get_trained_model(model_name, seed=42):
    """Instantiates the specified model architecture."""
    if model_name == "Logistic Regression":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=seed)
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_split=4, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            max_depth=4, learning_rate=0.05, n_estimators=100, random_state=seed, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_cv_oof(model_name, X_counts, y, fold_assignments, n_folds=5, rdkit_fps=None, is_hyd_array=None):
    """Executes N-Fold CV and returns Pooled Out-Of-Fold predictions."""
    oof_preds = np.zeros(len(y), dtype=float)

    for f in range(n_folds):
        train_idx = np.where(fold_assignments != f)[0]
        test_idx = np.where(fold_assignments == f)[0]

        if model_name == "Baseline 0 (Random Floor)":
            prior = np.mean(y[train_idx])
            oof_preds[test_idx] = prior

        elif model_name == "Baseline 1 (Hydantoin Detector)":
            clf = HydantoinDetectorClassifier()
            oof_preds[test_idx] = clf.predict_proba(is_hyd_array[test_idx])

        elif model_name == "Baseline 2 (1-NN Tanimoto)":
            test_fps_fold = [rdkit_fps[i] for i in test_idx]
            train_fps_fold = [rdkit_fps[i] for i in train_idx]
            train_labels_fold = y[train_idx]
            sims = compute_max_tanimoto_to_actives(test_fps_fold, train_fps_fold, train_labels_fold)
            oof_preds[test_idx] = sims

        else:
            model = get_trained_model(model_name)
            model.fit(X_counts[train_idx], y[train_idx])
            probs = model.predict_proba(X_counts[test_idx])[:, 1]
            oof_preds[test_idx] = probs

    return oof_preds


def run_lho_split(model_name, X_counts, y, is_hyd_array, rdkit_fps, train_on_hyd=False):
    """
    Executes a single Leave-Hydantoin-Out directional transfer split:
      If train_on_hyd=False: Train on Non-Hydantoins (N=51) -> Test on Hydantoins (N=51)
      If train_on_hyd=True:  Train on Hydantoins (N=51) -> Test on Non-Hydantoins (N=51)
    """
    if train_on_hyd:
        train_idx = np.where(is_hyd_array)[0]
        test_idx = np.where(~is_hyd_array)[0]
    else:
        train_idx = np.where(~is_hyd_array)[0]
        test_idx = np.where(is_hyd_array)[0]

    if model_name == "Baseline 0 (Random Floor)":
        prior = np.mean(y[train_idx])
        preds = np.full(len(test_idx), prior)

    elif model_name == "Baseline 1 (Hydantoin Detector)":
        clf = HydantoinDetectorClassifier()
        preds = clf.predict_proba(is_hyd_array[test_idx])

    elif model_name == "Baseline 2 (1-NN Tanimoto)":
        test_fps = [rdkit_fps[i] for i in test_idx]
        train_fps = [rdkit_fps[i] for i in train_idx]
        train_labels = y[train_idx]
        preds = compute_max_tanimoto_to_actives(test_fps, train_fps, train_labels)

    else:
        model = get_trained_model(model_name)
        model.fit(X_counts[train_idx], y[train_idx])
        preds = model.predict_proba(X_counts[test_idx])[:, 1]

    return test_idx, preds


def evaluate_predictions(y_true, y_pred, track_name, model_name, is_y_randomized=False):
    """Computes ROC-AUC, PR-AUC and their 1,000-bootstrap BCa confidence intervals."""
    n_samples = len(y_true)
    n_actives = int(np.sum(y_true == 1))
    n_inactives = int(np.sum(y_true == 0))

    roc_orig, roc_mean, roc_std, roc_ci_lo, roc_ci_hi = bootstrap_bca(y_true, y_pred, safe_roc_auc)
    pr_orig, pr_mean, pr_std, pr_ci_lo, pr_ci_hi = bootstrap_bca(y_true, y_pred, safe_pr_auc)

    return {
        "track": track_name,
        "model_name": model_name,
        "n_samples": n_samples,
        "n_actives": n_actives,
        "n_inactives": n_inactives,
        "roc_auc_orig": round(roc_orig, 4),
        "roc_auc_mean": round(roc_mean, 4),
        "roc_auc_std": round(roc_std, 4),
        "roc_auc_ci_lower": round(roc_ci_lo, 4),
        "roc_auc_ci_upper": round(roc_ci_hi, 4),
        "pr_auc_orig": round(pr_orig, 4),
        "pr_auc_mean": round(pr_mean, 4),
        "pr_auc_std": round(pr_std, 4),
        "pr_auc_ci_lower": round(pr_ci_lo, 4),
        "pr_auc_ci_upper": round(pr_ci_hi, 4),
        "is_y_randomized": is_y_randomized
    }


def main():
    print("=" * 90)
    print("PHASE 2 BENCHMARK: LB-ML vs BASELINES ACROSS RANDOM, CLUSTER & LHO TRACKS")
    print("=" * 90)

    # Load verified dataset with stratified cluster folds
    folds_path = "data/processed/stratified_cluster_folds.csv"
    if not os.path.exists(folds_path):
        print(f"[ERROR] Required folds file {folds_path} missing! Run src/splits.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(folds_path)
    print(f"[INFO] Loaded {len(df)} compounds from {folds_path}")

    y = (df["label"] == "Active").astype(int).values
    is_hyd_array = df["is_hydantoin"].values
    smiles = df["canonical_smiles"].tolist()

    # Precompute features
    print("[INFO] Featurizing molecules (2048-bit Morgan counts and RDKit bit vectors)...")
    X_counts = compute_morgan_fingerprints(smiles, radius=2, n_bits=2048, as_counts=True)
    rdkit_fps = compute_rdkit_bit_fingerprints(smiles, radius=2, n_bits=2048)
    print(f"[INFO] Featurization complete. Shape: {X_counts.shape}")

    models = [
        "Baseline 0 (Random Floor)",
        "Baseline 1 (Hydantoin Detector)",
        "Baseline 2 (1-NN Tanimoto)",
        "Logistic Regression",
        "Random Forest",
        "LightGBM"
    ]

    all_results = []

    # =========================================================================
    # TRACK A: Random 5-Fold Stratified CV (Control Split)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING TRACK A: Random 5-Fold Stratified CV (Analogue Memorization Control)")
    print("=" * 90)
    for m in models:
        oof_preds = run_cv_oof(
            m, X_counts, y, df["random_fold"].values, n_folds=5,
            rdkit_fps=rdkit_fps, is_hyd_array=is_hyd_array
        )
        res = evaluate_predictions(y, oof_preds, "Track A (Random CV)", m)
        all_results.append(res)
        print(f"[{m:<32}] ROC-AUC: {res['roc_auc_orig']:.3f} [{res['roc_auc_ci_lower']:.3f}, {res['roc_auc_ci_upper']:.3f}] | "
              f"PR-AUC: {res['pr_auc_orig']:.3f} [{res['pr_auc_ci_lower']:.3f}, {res['pr_auc_ci_upper']:.3f}]")

    # =========================================================================
    # TRACK B: Stratified Cluster 5-Fold CV (Zero Cluster Leakage)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING TRACK B: Stratified Cluster 5-Fold CV (Cluster-Separated Generalization)")
    print("=" * 90)
    for m in models:
        oof_preds = run_cv_oof(
            m, X_counts, y, df["cluster_fold"].values, n_folds=5,
            rdkit_fps=rdkit_fps, is_hyd_array=is_hyd_array
        )
        res = evaluate_predictions(y, oof_preds, "Track B (Cluster CV)", m)
        all_results.append(res)
        print(f"[{m:<32}] ROC-AUC: {res['roc_auc_orig']:.3f} [{res['roc_auc_ci_lower']:.3f}, {res['roc_auc_ci_upper']:.3f}] | "
              f"PR-AUC: {res['pr_auc_orig']:.3f} [{res['pr_auc_ci_lower']:.3f}, {res['pr_auc_ci_upper']:.3f}]")

    # =========================================================================
    # TRACK C: Leave-Hydantoin-Out (The Headline Test)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING TRACK C: Leave-Hydantoin-Out (LHO) Cross-Chemotype Generalization")
    print("=" * 90)
    for m in models:
        # Split 1: Train Non-Hyd -> Test Hyd (N=51)
        test_idx_1, preds_1 = run_lho_split(m, X_counts, y, is_hyd_array, rdkit_fps, train_on_hyd=False)
        res_s1 = evaluate_predictions(y[test_idx_1], preds_1, "Track C (LHO Split 1: Test Hydantoin)", m)
        all_results.append(res_s1)

        # Split 2: Train Hyd -> Test Non-Hyd (N=51)
        test_idx_2, preds_2 = run_lho_split(m, X_counts, y, is_hyd_array, rdkit_fps, train_on_hyd=True)
        res_s2 = evaluate_predictions(y[test_idx_2], preds_2, "Track C (LHO Split 2: Test Non-Hydantoin)", m)
        all_results.append(res_s2)

        # Pooled LHO predictions (N=102)
        pooled_y = np.concatenate([y[test_idx_1], y[test_idx_2]])
        pooled_preds = np.concatenate([preds_1, preds_2])
        res_pooled = evaluate_predictions(pooled_y, pooled_preds, "Track C (LHO Pooled OOF)", m)
        all_results.append(res_pooled)

        print(f"[{m:<32}] Split 1 (Test Hyd) PR-AUC: {res_s1['pr_auc_orig']:.3f} | Split 2 (Test Non-Hyd) PR-AUC: {res_s2['pr_auc_orig']:.3f} | "
              f"Pooled LHO PR-AUC: {res_pooled['pr_auc_orig']:.3f} [{res_pooled['pr_auc_ci_lower']:.3f}, {res_pooled['pr_auc_ci_upper']:.3f}]")

    # =========================================================================
    # Y-RANDOMIZATION CONTROL (10 Shuffles on Track B)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING Y-RANDOMIZATION CONTROL (10 Label Permutations on Track B)")
    print("=" * 90)
    ml_models = ["Logistic Regression", "Random Forest", "LightGBM"]
    n_shuffles = 10

    for m in ml_models:
        y_rand_roc = []
        y_rand_pr = []
        for s in range(n_shuffles):
            rng = np.random.RandomState(42 + s)
            y_perm = rng.permutation(y)
            oof_perm = run_cv_oof(
                m, X_counts, y_perm, df["cluster_fold"].values, n_folds=5,
                rdkit_fps=rdkit_fps, is_hyd_array=is_hyd_array
            )
            y_rand_roc.append(safe_roc_auc(y_perm, oof_perm))
            y_rand_pr.append(safe_pr_auc(y_perm, oof_perm))

        mean_roc = float(np.mean(y_rand_roc))
        std_roc = float(np.std(y_rand_roc))
        mean_pr = float(np.mean(y_rand_pr))
        std_pr = float(np.std(y_rand_pr))

        res_yrand = {
            "track": "Track B (Y-Randomization 10-Shuffle)",
            "model_name": m,
            "n_samples": len(y),
            "n_actives": int(np.sum(y == 1)),
            "n_inactives": int(np.sum(y == 0)),
            "roc_auc_orig": round(mean_roc, 4),
            "roc_auc_mean": round(mean_roc, 4),
            "roc_auc_std": round(std_roc, 4),
            "roc_auc_ci_lower": round(mean_roc - 1.96 * std_roc, 4),
            "roc_auc_ci_upper": round(mean_roc + 1.96 * std_roc, 4),
            "pr_auc_orig": round(mean_pr, 4),
            "pr_auc_mean": round(mean_pr, 4),
            "pr_auc_std": round(std_pr, 4),
            "pr_auc_ci_lower": round(mean_pr - 1.96 * std_pr, 4),
            "pr_auc_ci_upper": round(mean_pr + 1.96 * std_pr, 4),
            "is_y_randomized": True
        }
        all_results.append(res_yrand)
        print(f"[{m:<32}] Y-Rand ROC-AUC: {mean_roc:.3f} ± {std_roc:.3f} | Y-Rand PR-AUC: {mean_pr:.3f} ± {std_pr:.3f} (Collapsed to floor)")

    # Save results to CSV
    out_df = pd.DataFrame(all_results)
    out_csv = "reports/benchmark_phase2_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print(f"[SUCCESS] Saved full Phase 2 benchmark results to: {out_csv}")
    print("=" * 90)

    # Print Headline Comparison Table
    print("\n" + "=" * 90)
    print(f"{'HEADLINE COMPARISON: RANDOM CV vs. CLUSTER CV vs. LEAVE-HYDANTOIN-OUT':^90}")
    print("=" * 90)
    print(f"{'Model Architecture':<32} | {'Track A (Random)':<17} | {'Track B (Cluster)':<17} | {'Track C (LHO Pooled)'}")
    print(f"{'':<32} | {'PR-AUC (95% CI)':<17} | {'PR-AUC (95% CI)':<17} | {'PR-AUC (95% CI)'}")
    print("-" * 90)

    for m in models:
        res_a = next(r for r in all_results if r["track"] == "Track A (Random CV)" and r["model_name"] == m)
        res_b = next(r for r in all_results if r["track"] == "Track B (Cluster CV)" and r["model_name"] == m)
        res_c = next(r for r in all_results if r["track"] == "Track C (LHO Pooled OOF)" and r["model_name"] == m)

        str_a = f"{res_a['pr_auc_orig']:.3f} [{res_a['pr_auc_ci_lower']:.2f},{res_a['pr_auc_ci_upper']:.2f}]"
        str_b = f"{res_b['pr_auc_orig']:.3f} [{res_b['pr_auc_ci_lower']:.2f},{res_b['pr_auc_ci_upper']:.2f}]"
        str_c = f"{res_c['pr_auc_orig']:.3f} [{res_c['pr_auc_ci_lower']:.2f},{res_c['pr_auc_ci_upper']:.2f}]"

        print(f"{m:<32} | {str_a:<17} | {str_b:<17} | {str_c}")
    print("=" * 90)


if __name__ == "__main__":
    main()
