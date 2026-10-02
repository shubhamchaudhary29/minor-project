#!/usr/bin/env python3
"""
src/controls.py

Phase 3 Experimental Controls Suite for DprE1 Benchmark:
  1. Y-Randomization Test:
     - 10 independent label permutations evaluated via 5-Fold Cluster CV.
     - Verifies statistical collapse of ML models to dummy random floor.
     - Saves to reports/y_randomization_results.csv.
  2. Applicability Domain Decay Analysis:
     - Measures maximum Tanimoto similarity to training fold actives/compounds.
     - Evaluates performance across similarity tiers: [0.0-0.3), [0.3-0.5), [0.5-0.7), [0.7-1.0].
     - Plots PR-AUC and Brier score decay -> reports/figures/applicability_domain_decay.png.
  3. Feature Importance & Substructure Attribution:
     - Extracts Gini importances from whole-dataset Random Forest model.
     - Identifies top 5 positive and top 5 negative ECFP4 substructural bits.
     - Renders atom environments with RDKit -> reports/figures/top_ecfp4_substructures.png.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator, DataStructs

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_rdkit_bit_fingerprints, compute_max_tanimoto_to_train
from src.models import get_model


def safe_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.5


def safe_pr_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    try:
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        return float(np.mean(y_true))


def run_y_randomization(df, X_counts, y, n_shuffles=10, out_csv="reports/y_randomization_results.csv"):
    """
    Executes 10-iteration Y-randomization control on Cluster CV folds.
    """
    print("\n" + "=" * 80)
    print("1. EXECUTING Y-RANDOMIZATION CONTROL (10 Permutations on Cluster CV)")
    print("=" * 80)

    models_to_test = ["Logistic Regression", "Random Forest"]
    records = []

    for m_name in models_to_test:
        rocs = []
        prs = []
        for s in range(n_shuffles):
            rng = np.random.RandomState(100 + s)
            y_perm = rng.permutation(y)
            oof_preds = np.zeros(len(y), dtype=float)

            for f in range(5):
                train_idx = np.where(df["cluster_fold"] != f)[0]
                test_idx = np.where(df["cluster_fold"] == f)[0]
                model = get_model(m_name, feature_type="ecfp4_counts", seed=100 + s)
                model.fit(X_counts[train_idx], y_perm[train_idx])
                oof_preds[test_idx] = model.predict_proba(X_counts[test_idx])[:, 1]

            roc = safe_roc_auc(y_perm, oof_preds)
            pr = safe_pr_auc(y_perm, oof_preds)
            rocs.append(roc)
            prs.append(pr)
            records.append({
                "iteration": s + 1,
                "model_name": m_name,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4)
            })

        print(f"[{m_name:<20}] 10-Shuffle Mean ROC-AUC: {np.mean(rocs):.3f} ± {np.std(rocs):.3f} | "
              f"Mean PR-AUC: {np.mean(prs):.3f} ± {np.std(prs):.3f} (True Random Floor: 0.725)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_y = pd.DataFrame(records)
    df_y.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved Y-randomization results to: {out_csv}")
    return df_y


def run_applicability_domain_analysis(df, X_counts, y, rdkit_fps, out_png="reports/figures/applicability_domain_decay.png"):
    """
    Computes performance metrics stratified across similarity tiers to document applicability domain decay.
    """
    print("\n" + "=" * 80)
    print("2. EXECUTING APPLICABILITY DOMAIN ANALYSIS")
    print("=" * 80)

    # 1. Compute maximum training similarity for each test compound under Cluster CV
    max_train_sims = np.zeros(len(df), dtype=float)
    rf_oof_preds = np.zeros(len(df), dtype=float)
    lr_oof_preds = np.zeros(len(df), dtype=float)

    for f in range(5):
        train_idx = np.where(df["cluster_fold"] != f)[0]
        test_idx = np.where(df["cluster_fold"] == f)[0]

        train_fps = [rdkit_fps[i] for i in train_idx]
        test_fps = [rdkit_fps[i] for i in test_idx]

        sims = compute_max_tanimoto_to_train(test_fps, train_fps)
        max_train_sims[test_idx] = sims

        # Fit models on fold
        rf = get_model("Random Forest", feature_type="ecfp4_counts", seed=42 + f)
        rf.fit(X_counts[train_idx], y[train_idx])
        rf_oof_preds[test_idx] = rf.predict_proba(X_counts[test_idx])[:, 1]

        lr = get_model("Logistic Regression", feature_type="ecfp4_counts", seed=42 + f)
        lr.fit(X_counts[train_idx], y[train_idx])
        lr_oof_preds[test_idx] = lr.predict_proba(X_counts[test_idx])[:, 1]

    # Assign bins
    bins = [0.0, 0.3, 0.5, 0.7, 1.0]
    tier_labels = ["[0.0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 1.0]"]
    tier_series = pd.cut(max_train_sims, bins=bins, labels=tier_labels, include_lowest=True)

    tier_stats = []
    print(f"{'Similarity Tier':<15} | {'N':<4} | {'Act':<4} | {'Inact':<5} | {'RF PR-AUC':<10} | {'RF Brier':<10} | {'LR PR-AUC':<10} | {'LR Brier':<10}")
    print("-" * 80)

    for t in tier_labels:
        idx = np.where(tier_series == t)[0]
        n_t = len(idx)
        if n_t == 0:
            continue
        y_t = y[idx]
        n_act = int(np.sum(y_t == 1))
        n_inact = int(np.sum(y_t == 0))

        rf_pr = safe_pr_auc(y_t, rf_oof_preds[idx])
        rf_brier = float(brier_score_loss(y_t, rf_oof_preds[idx]))

        lr_pr = safe_pr_auc(y_t, lr_oof_preds[idx])
        lr_brier = float(brier_score_loss(y_t, lr_oof_preds[idx]))

        tier_stats.append({
            "tier": t,
            "n": n_t,
            "n_actives": n_act,
            "n_inactives": n_inact,
            "rf_pr_auc": rf_pr,
            "rf_brier": rf_brier,
            "lr_pr_auc": lr_pr,
            "lr_brier": lr_brier
        })
        print(f"{t:<15} | {n_t:<4} | {n_act:<4} | {n_inact:<5} | {rf_pr:<10.3f} | {rf_brier:<10.3f} | {lr_pr:<10.3f} | {lr_brier:<10.3f}")
    print("-" * 80)

    df_tiers = pd.DataFrame(tier_stats)

    # Plot Applicability Domain Decay figure
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: PR-AUC vs Similarity Tier
    x_pos = np.arange(len(df_tiers))
    width = 0.35

    axes[0].bar(x_pos - width/2, df_tiers["rf_pr_auc"], width, label="Random Forest", color="#1f77b4", alpha=0.9, edgecolor="black")
    axes[0].bar(x_pos + width/2, df_tiers["lr_pr_auc"], width, label="Logistic Regression", color="#ff7f0e", alpha=0.9, edgecolor="black")
    axes[0].axhline(y=0.725, color="red", linestyle="--", linewidth=1.5, label="Random Prior Floor (0.725)")
    axes[0].set_ylabel("Precision-Recall AUC (PR-AUC)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Maximum Tanimoto Similarity to Training Split", fontsize=12, fontweight="bold")
    axes[0].set_title("PR-AUC Generalization vs. Structural Similarity", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"{r['tier']}\n(N={r['n']})" for _, r in df_tiers.iterrows()], fontsize=10)
    axes[0].set_ylim(0.4, 1.05)
    axes[0].legend(loc="lower right", frameon=True)

    # Subplot 2: Brier Score Error vs Similarity Tier
    axes[1].bar(x_pos - width/2, df_tiers["rf_brier"], width, label="Random Forest", color="#1f77b4", alpha=0.9, edgecolor="black")
    axes[1].bar(x_pos + width/2, df_tiers["lr_brier"], width, label="Logistic Regression", color="#ff7f0e", alpha=0.9, edgecolor="black")
    axes[1].set_ylabel("Brier Score Loss (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Maximum Tanimoto Similarity to Training Split", fontsize=12, fontweight="bold")
    axes[1].set_title("Calibration Error vs. Structural Similarity", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"{r['tier']}\n(N={r['n']})" for _, r in df_tiers.iterrows()], fontsize=10)
    axes[1].set_ylim(0.0, 0.40)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Applicability Domain Decay plot to: {out_png}")
    return df_tiers


def run_feature_importance_attribution(df, smiles_list, y, out_png="reports/figures/top_ecfp4_substructures.png"):
    """
    Extracts Gini feature importances from whole-dataset Random Forest model,
    identifies top 5 positive and negative ECFP4 bits, and visualizes atom environments.
    """
    print("\n" + "=" * 80)
    print("3. EXECUTING FEATURE IMPORTANCE & SUBSTRUCTURE ATTRIBUTION")
    print("=" * 80)

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    bit_infos = []
    fps_np = []
    for m in mols:
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        fp = gen.GetFingerprint(m, additionalOutput=ao)
        info = ao.GetBitInfoMap()
        bit_infos.append(info)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps_np.append(arr)

    X_bits = np.vstack(fps_np)

    # Train whole-dataset Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_split=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    rf.fit(X_bits, y)
    importances = rf.feature_importances_

    # Determine correlation directionality with activity
    corrs = np.zeros(2048)
    for b in range(2048):
        if np.std(X_bits[:, b]) > 0:
            corrs[b] = np.corrcoef(X_bits[:, b], y)[0, 1]

    # Top 5 positive bits (promotes activity)
    pos_idx = np.where(corrs > 0.05)[0]
    top_pos_bits = pos_idx[np.argsort(-importances[pos_idx])[:5]]

    # Top 5 negative bits (associated with inactivity)
    neg_idx = np.where(corrs < -0.05)[0]
    top_neg_bits = neg_idx[np.argsort(-importances[neg_idx])[:5]]

    print("Top 5 Positive ECFP4 Bits (Activity-Promoting):")
    for b in top_pos_bits:
        print(f"  - Bit {b:<4}: Gini Importance = {importances[b]:.4f} | Pearson r = {corrs[b]:+.3f}")

    print("\nTop 5 Negative ECFP4 Bits (Inactivity-Associated):")
    for b in top_neg_bits:
        print(f"  - Bit {b:<4}: Gini Importance = {importances[b]:.4f} | Pearson r = {corrs[b]:+.3f}")

    # Build DrawMorganBits item list: (mol, bit_id, bit_info)
    items_to_draw = []
    legends = []

    for b in top_pos_bits:
        for i, info in enumerate(bit_infos):
            if b in info:
                items_to_draw.append((mols[i], b, info))
                legends.append(f"Bit {b} (+Act)\nGini: {importances[b]:.3f}")
                break

    for b in top_neg_bits:
        for i, info in enumerate(bit_infos):
            if b in info:
                items_to_draw.append((mols[i], b, info))
                legends.append(f"Bit {b} (-Inact)\nGini: {importances[b]:.3f}")
                break

    img_bytes = Draw.DrawMorganBits(
        items_to_draw, molsPerRow=5, subImgSize=(250, 250), legends=legends, useSVG=False
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with open(out_png, "wb") as f:
        f.write(img_bytes)

    print(f"\n[SUCCESS] Rendered top 10 predictive ECFP4 substructures to: {out_png}")


def main():
    print("=" * 80)
    print("PHASE 3 EXPERIMENTAL CONTROLS ENGINE")
    print("=" * 80)

    dataset_path = "data/processed/stratified_cluster_folds.csv"
    df = pd.read_csv(dataset_path)
    y = (df["label"] == "Active").astype(int).values
    smiles = df["canonical_smiles"].tolist()

    # Load precomputed features
    feat_dir = "data/interim/features"
    X_counts = np.load(os.path.join(feat_dir, "ecfp4_counts.npy"))
    rdkit_fps = compute_rdkit_bit_fingerprints(smiles, radius=2, n_bits=2048)

    # 1. Y-Randomization
    run_y_randomization(df, X_counts, y, n_shuffles=10, out_csv="reports/y_randomization_results.csv")

    # 2. Applicability Domain Decay
    run_applicability_domain_analysis(df, X_counts, y, rdkit_fps, out_png="reports/figures/applicability_domain_decay.png")

    # 3. Feature Importance & Substructure Attribution
    run_feature_importance_attribution(df, smiles, y, out_png="reports/figures/top_ecfp4_substructures.png")

    print("\n" + "=" * 80)
    print("[SUCCESS] All Phase 3 Experimental Controls Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
