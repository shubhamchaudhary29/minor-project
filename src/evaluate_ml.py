#!/usr/bin/env python3
"""
src/evaluate_ml.py

Phase 3 Comprehensive Benchmark Evaluation Engine:
Evaluates Baseline models and Machine Learning architectures across three distinct evaluation tracks:
  - Track A (Control Split): Random 5-Fold Stratified CV (demonstrates analogue memorization bias).
  - Track B (Cluster Split): Stratified Cluster 5-Fold CV (zero cluster leakage, >= 3 inactives/fold).
  - Track C (Headline Test): Leave-Hydantoin-Out (LHO) Cross-Scaffold Generalization:
      * Split 1: Train on Non-Hydantoins (N=51) -> Test on Hydantoins (N=51)
      * Split 2: Train on Hydantoins (N=51) -> Test on Non-Hydantoins (N=51)
      * Pooled LHO: Concatenated out-of-scaffold predictions (N=102)

Models & Representations Evaluated:
  1. Baseline 0: Random Floor
  2. Baseline 1: Hydantoin Detector (Confounding Heuristic)
  3. Baseline 2: 1-NN Tanimoto Similarity
  4. Logistic Regression (ECFP4 Counts, inner-tuned C)
  5. Random Forest (ECFP4 Counts)
  6. LightGBM (ECFP4 Counts)
  7. Random Forest (ECFP4 Bits)
  8. Random Forest (MACCS Keys)
  9. Random Forest (RDKit 2D Descriptors)
  10. Logistic Regression (RDKit 2D Descriptors)

Outputs:
  reports/benchmark_phase3_ml_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_rdkit_bit_fingerprints
from src.baselines import RandomFloorClassifier, HydantoinDetectorClassifier, Tanimoto1NNClassifier
from src.models import get_model


def safe_roc_auc(y_true, y_pred):
    """Calculates ROC-AUC, falling back to 0.5 if labels or predictions are degenerate."""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.5


def safe_pr_auc(y_true, y_pred):
    """Calculates PR-AUC (Average Precision), falling back to positive prevalence if degenerate."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    try:
        return float(average_precision_score(y_true, y_pred))
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

    return orig_val, float(np.mean(boot_vals)), float(np.std(boot_vals)), float(ci_lower), float(ci_upper)


def evaluate_predictions(y_true, y_pred, track_name, model_display_name):
    """Computes ROC-AUC, PR-AUC and their 1,000-bootstrap BCa confidence intervals."""
    n_samples = len(y_true)
    n_actives = int(np.sum(y_true == 1))
    n_inactives = int(np.sum(y_true == 0))

    roc_orig, roc_mean, roc_std, roc_ci_lo, roc_ci_hi = bootstrap_bca(y_true, y_pred, safe_roc_auc)
    pr_orig, pr_mean, pr_std, pr_ci_lo, pr_ci_hi = bootstrap_bca(y_true, y_pred, safe_pr_auc)

    return {
        "track": track_name,
        "model_name": model_display_name,
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
        "pr_auc_ci_upper": round(pr_ci_hi, 4)
    }


def run_cv_oof(model_spec, feature_dict, y, fold_assignments, rdkit_fps, is_hyd_array, n_folds=5):
    """Executes N-Fold CV and returns Pooled Out-Of-Fold predictions."""
    oof_preds = np.zeros(len(y), dtype=float)
    m_type, f_type = model_spec

    for f in range(n_folds):
        train_idx = np.where(fold_assignments != f)[0]
        test_idx = np.where(fold_assignments == f)[0]

        if m_type == "Baseline 0 (Random Floor)":
            clf = RandomFloorClassifier().fit(None, y[train_idx])
            oof_preds[test_idx] = clf.predict_proba(test_idx)[:, 1]

        elif m_type == "Baseline 1 (Hydantoin Detector)":
            clf = HydantoinDetectorClassifier()
            oof_preds[test_idx] = clf.predict_proba(is_hyd_array[test_idx])[:, 1]

        elif m_type == "Baseline 2 (1-NN Tanimoto)":
            train_fps = [rdkit_fps[i] for i in train_idx]
            test_fps = [rdkit_fps[i] for i in test_idx]
            clf = Tanimoto1NNClassifier().fit(train_fps, y[train_idx])
            oof_preds[test_idx] = clf.predict_proba(test_fps)[:, 1]

        else:
            X = feature_dict[f_type]
            model = get_model(m_type, feature_type=f_type, seed=42 + f)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    return oof_preds


def run_lho_split(model_spec, feature_dict, y, is_hyd_array, rdkit_fps, train_on_hyd=False):
    """Executes a single Leave-Hydantoin-Out directional transfer split."""
    m_type, f_type = model_spec
    if train_on_hyd:
        train_idx = np.where(is_hyd_array)[0]
        test_idx = np.where(~is_hyd_array)[0]
    else:
        train_idx = np.where(~is_hyd_array)[0]
        test_idx = np.where(is_hyd_array)[0]

    if m_type == "Baseline 0 (Random Floor)":
        clf = RandomFloorClassifier().fit(None, y[train_idx])
        preds = clf.predict_proba(test_idx)[:, 1]

    elif m_type == "Baseline 1 (Hydantoin Detector)":
        clf = HydantoinDetectorClassifier()
        preds = clf.predict_proba(is_hyd_array[test_idx])[:, 1]

    elif m_type == "Baseline 2 (1-NN Tanimoto)":
        train_fps = [rdkit_fps[i] for i in train_idx]
        test_fps = [rdkit_fps[i] for i in test_idx]
        clf = Tanimoto1NNClassifier().fit(train_fps, y[train_idx])
        preds = clf.predict_proba(test_fps)[:, 1]

    else:
        X = feature_dict[f_type]
        model = get_model(m_type, feature_type=f_type, seed=42)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[test_idx])[:, 1]

    return test_idx, preds


def main():
    print("=" * 95)
    print("PHASE 3: LIGAND-BASED MACHINE LEARNING BENCHMARK EVALUATION")
    print("=" * 95)

    dataset_path = "data/processed/stratified_cluster_folds.csv"
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Loaded {len(df)} compounds from {dataset_path}")

    y = (df["label"] == "Active").astype(int).values
    is_hyd_array = df["is_hydantoin"].values
    smiles = df["canonical_smiles"].tolist()

    # Load precomputed features from data/interim/features/
    feat_dir = "data/interim/features"
    feature_dict = {
        "ecfp4_counts": np.load(os.path.join(feat_dir, "ecfp4_counts.npy")),
        "ecfp4_bits": np.load(os.path.join(feat_dir, "ecfp4_bits.npy")),
        "maccs": np.load(os.path.join(feat_dir, "maccs.npy")),
        "rdkit_2d": np.load(os.path.join(feat_dir, "rdkit_2d.npy")),
    }
    rdkit_fps = compute_rdkit_bit_fingerprints(smiles, radius=2, n_bits=2048)
    print(f"[INFO] Loaded precomputed feature matrices from {feat_dir}:")
    for k, v in feature_dict.items():
        print(f"  - {k:<15}: shape {v.shape}")

    # Model specifications: (Model Name, Feature Type, Display Name)
    model_configs = [
        ("Baseline 0 (Random Floor)", None, "Baseline 0 (Random Floor)"),
        ("Baseline 1 (Hydantoin Detector)", None, "Baseline 1 (Hydantoin Detector)"),
        ("Baseline 2 (1-NN Tanimoto)", None, "Baseline 2 (1-NN Tanimoto)"),
        ("Logistic Regression", "ecfp4_counts", "Logistic Regression (ECFP4 Counts)"),
        ("Random Forest", "ecfp4_counts", "Random Forest (ECFP4 Counts)"),
        ("LightGBM", "ecfp4_counts", "LightGBM (ECFP4 Counts)"),
        ("Random Forest", "ecfp4_bits", "Random Forest (ECFP4 Bits)"),
        ("Random Forest", "maccs", "Random Forest (MACCS Keys)"),
        ("Random Forest", "rdkit_2d", "Random Forest (RDKit 2D)"),
        ("Logistic Regression", "rdkit_2d", "Logistic Regression (RDKit 2D)")
    ]

    all_results = []

    # =========================================================================
    # TRACK A: Random 5-Fold Stratified CV
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXECUTING TRACK A: Random 5-Fold Stratified CV (Analogue Memorization Control)")
    print("=" * 95)
    for m_type, f_type, d_name in model_configs:
        oof_preds = run_cv_oof(
            (m_type, f_type), feature_dict, y, df["random_fold"].values,
            rdkit_fps, is_hyd_array, n_folds=5
        )
        res = evaluate_predictions(y, oof_preds, "Track A (Random CV)", d_name)
        all_results.append(res)
        print(f"[{d_name:<36}] ROC-AUC: {res['roc_auc_orig']:.3f} [{res['roc_auc_ci_lower']:.3f}, {res['roc_auc_ci_upper']:.3f}] | "
              f"PR-AUC: {res['pr_auc_orig']:.3f} [{res['pr_auc_ci_lower']:.3f}, {res['pr_auc_ci_upper']:.3f}]")

    # =========================================================================
    # TRACK B: Stratified Cluster 5-Fold CV
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXECUTING TRACK B: Stratified Cluster 5-Fold CV (Zero Cluster Leakage)")
    print("=" * 95)
    for m_type, f_type, d_name in model_configs:
        oof_preds = run_cv_oof(
            (m_type, f_type), feature_dict, y, df["cluster_fold"].values,
            rdkit_fps, is_hyd_array, n_folds=5
        )
        res = evaluate_predictions(y, oof_preds, "Track B (Cluster CV)", d_name)
        all_results.append(res)
        print(f"[{d_name:<36}] ROC-AUC: {res['roc_auc_orig']:.3f} [{res['roc_auc_ci_lower']:.3f}, {res['roc_auc_ci_upper']:.3f}] | "
              f"PR-AUC: {res['pr_auc_orig']:.3f} [{res['pr_auc_ci_lower']:.3f}, {res['pr_auc_ci_upper']:.3f}]")

    # =========================================================================
    # TRACK C: Leave-Hydantoin-Out (LHO)
    # =========================================================================
    print("\n" + "=" * 95)
    print("EXECUTING TRACK C: Leave-Hydantoin-Out (Cross-Chemotype Generalization)")
    print("=" * 95)
    for m_type, f_type, d_name in model_configs:
        # Split 1: Train Non-Hyd -> Test Hyd (N=51)
        test_idx_1, preds_1 = run_lho_split(
            (m_type, f_type), feature_dict, y, is_hyd_array, rdkit_fps, train_on_hyd=False
        )
        res_s1 = evaluate_predictions(y[test_idx_1], preds_1, "Track C (LHO Split 1: Test Hydantoin)", d_name)
        all_results.append(res_s1)

        # Split 2: Train Hyd -> Test Non-Hyd (N=51)
        test_idx_2, preds_2 = run_lho_split(
            (m_type, f_type), feature_dict, y, is_hyd_array, rdkit_fps, train_on_hyd=True
        )
        res_s2 = evaluate_predictions(y[test_idx_2], preds_2, "Track C (LHO Split 2: Test Non-Hydantoin)", d_name)
        all_results.append(res_s2)

        # Pooled LHO predictions (N=102)
        pooled_y = np.concatenate([y[test_idx_1], y[test_idx_2]])
        pooled_preds = np.concatenate([preds_1, preds_2])
        res_pooled = evaluate_predictions(pooled_y, pooled_preds, "Track C (LHO Pooled OOF)", d_name)
        all_results.append(res_pooled)

        print(f"[{d_name:<36}] Split 1 PR-AUC: {res_s1['pr_auc_orig']:.3f} | Split 2 PR-AUC: {res_s2['pr_auc_orig']:.3f} | "
              f"Pooled LHO PR-AUC: {res_pooled['pr_auc_orig']:.3f} [{res_pooled['pr_auc_ci_lower']:.3f}, {res_pooled['pr_auc_ci_upper']:.3f}]")

    # Save results to CSV
    out_df = pd.DataFrame(all_results)
    out_csv = "reports/benchmark_phase3_ml_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print("\n" + "=" * 95)
    print(f"[SUCCESS] Saved structured Phase 3 ML results to: {out_csv}")
    print("=" * 95)

    # Print Headline Comparison Table
    print("\n" + "=" * 95)
    print(f"{'HEADLINE COMPARISON: RANDOM CV vs. CLUSTER CV vs. LEAVE-HYDANTOIN-OUT':^95}")
    print("=" * 95)
    print(f"{'Model Architecture':<36} | {'Track A (Random)':<17} | {'Track B (Cluster)':<17} | {'Track C (LHO Pooled)'}")
    print(f"{'':<36} | {'PR-AUC (95% CI)':<17} | {'PR-AUC (95% CI)':<17} | {'PR-AUC (95% CI)'}")
    print("-" * 95)

    for _, _, d_name in model_configs:
        res_a = next(r for r in all_results if r["track"] == "Track A (Random CV)" and r["model_name"] == d_name)
        res_b = next(r for r in all_results if r["track"] == "Track B (Cluster CV)" and r["model_name"] == d_name)
        res_c = next(r for r in all_results if r["track"] == "Track C (LHO Pooled OOF)" and r["model_name"] == d_name)

        str_a = f"{res_a['pr_auc_orig']:.3f} [{res_a['pr_auc_ci_lower']:.2f},{res_a['pr_auc_ci_upper']:.2f}]"
        str_b = f"{res_b['pr_auc_orig']:.3f} [{res_b['pr_auc_ci_lower']:.2f},{res_b['pr_auc_ci_upper']:.2f}]"
        str_c = f"{res_c['pr_auc_orig']:.3f} [{res_c['pr_auc_ci_lower']:.2f},{res_c['pr_auc_ci_upper']:.2f}]"

        print(f"{d_name:<36} | {str_a:<17} | {str_b:<17} | {str_c}")
    print("=" * 95)


if __name__ == "__main__":
    main()
