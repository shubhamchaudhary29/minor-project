#!/usr/bin/env python3
"""
src/evaluate_integration.py

Phase 5 Master Benchmark Evaluation Suite:
Computes ROC-AUC, PR-AUC, EF_10%, and EF_1% with 1,000 bootstrap iterations
(95% BCa/percentile confidence intervals) across:
  - Track B: Stratified Cluster 5-Fold CV (N=93)
  - Track C: Leave-Hydantoin-Out Cross-Chemotype Generalization (N=93)
  - Non-Hydantoin Independent Test Subset under LHO (N=43: 25 active, 18 inactive)

Directly evaluates whether Hybrid Rank Fusion (RRF, MPR, Z-Score) beats:
  1. Pure ML alone (Random Forest, Logistic Regression)
  2. Pure Docking alone (Raw Vina, Ligand Efficiency)
  3. Naive Baselines (Random Floor, Hydantoin Detector, 1-NN Tanimoto)

Computes paired empirical Delta (Delta = Hybrid - ML) to test for statistical significance.

Outputs:
  reports/phase5_benchmark_master.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_ef(y_true, scores, fraction=0.10):
    """Computes Enrichment Factor at given fraction."""
    n = len(y_true)
    n_act = int(np.sum(y_true == 1))
    if n_act == 0 or n == 0:
        return 0.0
    k = max(1, int(round(fraction * n)))
    top_k_idx = np.argsort(-scores)[:k]
    hits = int(np.sum(y_true[top_k_idx] == 1))
    return float((hits / k) / (n_act / n))


def bootstrap_metric_ci(y_true, scores, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Computes 1,000 bootstrap iterations yielding mean, std, and 95% BCa/percentile CIs."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    orig_val = float(metric_fn(y_true, scores))

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_vals.append(float(metric_fn(y_true[idx], scores[idx])))

    if len(boot_vals) < 50:
        return orig_val, orig_val, 0.0, orig_val, orig_val

    boot_vals = np.array(boot_vals)
    ci_lower = float(np.percentile(boot_vals, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_vals, 100 * (1.0 - alpha / 2.0)))

    return orig_val, float(np.mean(boot_vals)), float(np.std(boot_vals)), ci_lower, ci_upper


def bootstrap_delta_ci(y_true, scores_hybrid, scores_ml, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Computes paired bootstrap confidence interval for Delta = Metric(Hybrid) - Metric(ML)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    orig_delta = float(metric_fn(y_true, scores_hybrid) - metric_fn(y_true, scores_ml))

    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        d = float(metric_fn(y_true[idx], scores_hybrid[idx]) - metric_fn(y_true[idx], scores_ml[idx]))
        boot_deltas.append(d)

    if len(boot_deltas) < 50:
        return orig_delta, orig_delta, 0.0, orig_delta, orig_delta

    boot_deltas = np.array(boot_deltas)
    ci_lower = float(np.percentile(boot_deltas, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_deltas, 100 * (1.0 - alpha / 2.0)))
    return orig_delta, float(np.mean(boot_deltas)), float(np.std(boot_deltas)), ci_lower, ci_upper


def evaluate_cohort(df_subset, cohort_label, track_type="cluster"):
    """Evaluates all standalone and hybrid methods on a given dataset partition."""
    y = (df_subset["label"] == "Active").astype(int).values
    n = len(df_subset)
    n_act = int(np.sum(y == 1))
    n_inact = int(np.sum(y == 0))

    # Define score mappings based on track type
    if track_type == "cluster":
        methods = [
            ("Baseline 0 (Random Floor)", np.full(n, n_act / n), None),
            ("Baseline 1 (Hydantoin Detector)", np.where(df_subset["is_hydantoin"], 0.8431, 0.6078), None),
            ("Baseline 2 (1-NN Tanimoto)", df_subset["prob_tanimoto_cluster"].values, None),
            ("AutoDock Vina (Raw Affinity)", df_subset["vina_score"].values, None),
            ("AutoDock Vina (Ligand Efficiency)", df_subset["ligand_efficiency"].values, None),
            ("Random Forest (ECFP4 Counts)", df_subset["prob_rf_cluster"].values, None),
            ("Logistic Regression (ECFP4 Counts)", df_subset["prob_lr_cluster"].values, None),
            ("RRF (RF + Vina)", df_subset["rrf_rf_vina_cluster"].values, df_subset["prob_rf_cluster"].values),
            ("RRF (LR + Vina)", df_subset["rrf_lr_vina_cluster"].values, df_subset["prob_lr_cluster"].values),
            ("RRF (RF + LE-Vina)", df_subset["rrf_rf_le_cluster"].values, df_subset["prob_rf_cluster"].values),
            ("MPR (RF + Vina)", df_subset["mpr_rf_vina_cluster"].values, df_subset["prob_rf_cluster"].values),
            ("Z-Score Sum (RF + Vina)", df_subset["zscore_rf_vina_cluster"].values, df_subset["prob_rf_cluster"].values),
        ]
    else:  # LHO track
        methods = [
            ("Baseline 0 (Random Floor)", np.full(n, n_act / n), None),
            ("Baseline 1 (Hydantoin Detector)", np.where(df_subset["is_hydantoin"], 0.8431, 0.6078), None),
            ("Baseline 2 (1-NN Tanimoto)", df_subset["prob_tanimoto_lho"].values, None),
            ("AutoDock Vina (Raw Affinity)", df_subset["vina_score"].values, None),
            ("AutoDock Vina (Ligand Efficiency)", df_subset["ligand_efficiency"].values, None),
            ("Random Forest (ECFP4 Counts)", df_subset["prob_rf_lho"].values, None),
            ("Logistic Regression (ECFP4 Counts)", df_subset["prob_lr_lho"].values, None),
            ("RRF (RF + Vina)", df_subset["rrf_rf_vina_lho"].values, df_subset["prob_rf_lho"].values),
            ("RRF (LR + Vina)", df_subset["rrf_lr_vina_lho"].values, df_subset["prob_lr_lho"].values),
            ("RRF (RF + LE-Vina)", df_subset["rrf_rf_le_lho"].values, df_subset["prob_rf_lho"].values),
            ("MPR (RF + Vina)", df_subset["mpr_rf_vina_lho"].values, df_subset["prob_rf_lho"].values),
            ("Z-Score Sum (RF + Vina)", df_subset["zscore_rf_vina_lho"].values, df_subset["prob_rf_lho"].values),
        ]

    records = []
    print(f"\n" + "=" * 90)
    print(f"COHORT: {cohort_label} (N={n}: {n_act} Act, {n_inact} Inact | Prior: {n_act/n:.3f})")
    print("=" * 90)
    print(f"{'Method Name':<34} | {'ROC-AUC [95% CI]':<20} | {'PR-AUC [95% CI]':<20} | {'EF_10%':<6} | {'Delta ROC [95% CI]'}")
    print("-" * 90)

    for m_name, scores, ref_ml_scores in methods:
        roc_orig, _, _, roc_lo, roc_hi = bootstrap_metric_ci(y, scores, roc_auc_score)
        pr_orig, _, _, pr_lo, pr_hi = bootstrap_metric_ci(y, scores, average_precision_score)
        ef10_orig, _, _, ef10_lo, ef10_hi = bootstrap_metric_ci(y, scores, lambda y_t, s: compute_ef(y_t, s, 0.10))
        ef1_orig = compute_ef(y, scores, 0.01)

        delta_roc_str = "—"
        delta_roc_orig, delta_roc_lo, delta_roc_hi = np.nan, np.nan, np.nan
        delta_pr_orig, delta_pr_lo, delta_pr_hi = np.nan, np.nan, np.nan

        if ref_ml_scores is not None:
            d_roc, _, _, d_roc_lo, d_roc_hi = bootstrap_delta_ci(y, scores, ref_ml_scores, roc_auc_score)
            d_pr, _, _, d_pr_lo, d_pr_hi = bootstrap_delta_ci(y, scores, ref_ml_scores, average_precision_score)
            delta_roc_orig, delta_roc_lo, delta_roc_hi = d_roc, d_roc_lo, d_roc_hi
            delta_pr_orig, delta_pr_lo, delta_pr_hi = d_pr, d_pr_lo, d_pr_hi
            delta_roc_str = f"{d_roc:+.3f} [{d_roc_lo:+.2f}, {d_roc_hi:+.2f}]"

        records.append({
            "cohort": cohort_label,
            "method": m_name,
            "n_samples": n,
            "n_actives": n_act,
            "n_inactives": n_inact,
            "roc_auc_orig": round(roc_orig, 4),
            "roc_auc_ci_lower": round(roc_lo, 4),
            "roc_auc_ci_upper": round(roc_hi, 4),
            "pr_auc_orig": round(pr_orig, 4),
            "pr_auc_ci_lower": round(pr_lo, 4),
            "pr_auc_ci_upper": round(pr_hi, 4),
            "ef_10_orig": round(ef10_orig, 4),
            "ef_10_ci_lower": round(ef10_lo, 4),
            "ef_10_ci_upper": round(ef10_hi, 4),
            "ef_1_orig": round(ef1_orig, 4),
            "delta_roc_orig": round(delta_roc_orig, 4) if not np.isnan(delta_roc_orig) else None,
            "delta_roc_ci_lower": round(delta_roc_lo, 4) if not np.isnan(delta_roc_lo) else None,
            "delta_roc_ci_upper": round(delta_roc_hi, 4) if not np.isnan(delta_roc_hi) else None,
            "delta_pr_orig": round(delta_pr_orig, 4) if not np.isnan(delta_pr_orig) else None,
            "delta_pr_ci_lower": round(delta_pr_lo, 4) if not np.isnan(delta_pr_lo) else None,
            "delta_pr_ci_upper": round(delta_pr_hi, 4) if not np.isnan(delta_pr_hi) else None,
        })

        roc_fmt = f"{roc_orig:.3f} [{roc_lo:.3f},{roc_hi:.3f}]"
        pr_fmt = f"{pr_orig:.3f} [{pr_lo:.3f},{pr_hi:.3f}]"
        print(f"{m_name:<34} | {roc_fmt:<20} | {pr_fmt:<20} | {ef10_orig:>5.2f}x | {delta_roc_str}")

    return records


def main():
    print("=" * 90)
    print("PHASE 5: MASTER BENCHMARK EVALUATION (STANDALONE vs. HYBRID FUSION)")
    print("=" * 90)

    pred_csv = "data/processed/master_predictions_phase5.csv"
    if not os.path.exists(pred_csv):
        print(f"[ERROR] Missing {pred_csv}. Run src/integrate.py first!", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pred_csv)
    print(f"[INFO] Loaded {len(df)} aligned non-covalent molecules.")

    all_records = []

    # 1. Track B: Stratified Cluster 5-Fold CV (N=93)
    rec_b = evaluate_cohort(df, "Track B: Cluster 5-Fold CV (N=93)", track_type="cluster")
    all_records.extend(rec_b)

    # 2. Track C: Leave-Hydantoin-Out (N=93)
    rec_c = evaluate_cohort(df, "Track C: Leave-Hydantoin-Out (N=93)", track_type="lho")
    all_records.extend(rec_c)

    # 3. Non-Hydantoin Independent Test Subset under LHO (N=43)
    df_non_hyd = df[~df["is_hydantoin"]].copy().reset_index(drop=True)
    rec_non_hyd = evaluate_cohort(df_non_hyd, "Non-Hydantoin Independent Test Subset (N=43)", track_type="lho")
    all_records.extend(rec_non_hyd)

    out_csv = "reports/phase5_benchmark_master.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Master benchmark results saved to: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
