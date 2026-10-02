#!/usr/bin/env python3
"""
src/integrate.py

Phase 5 Pipeline Alignment & Multi-Modal Integration:
Aligns ligand-based machine learning predictions (Phase 3) with structure-based
AutoDock Vina docking scores (Phase 4) on the exact non-covalent cohort (N=93).

Executes and aligns:
  1. Primary Docking Scores:
     - Raw Vina Binding Affinity (kcal/mol)
     - Vina Ranking Score (-affinity, higher = better)
     - Ligand Efficiency (LE = -affinity / heavy_atom_count)
  2. Machine Learning OOF Predictions:
     - Track B (Stratified Cluster 5-Fold CV): Random Forest, Logistic Regression, 1-NN Tanimoto
     - Track C (Leave-Hydantoin-Out Cross-Chemotype Generalization): RF, LR, 1-NN Tanimoto
  3. Hybrid Fusion Strategies (No post-hoc tuning):
     - Reciprocal Rank Fusion (RRF, k=60)
     - Mean Percentile Rank (MPR)
     - Standardized Z-Score Sum (Z_comb = z_ML + z_Dock)

Outputs:
  data/processed/master_predictions_phase5.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import compute_morgan_fingerprints, compute_rdkit_bit_fingerprints, compute_max_tanimoto_to_train
from src.baselines import Tanimoto1NNClassifier
from src.models import get_model


def compute_rrf(scores_a, scores_b, k=60):
    """
    Reciprocal Rank Fusion of two score arrays (higher score is better for both).
    RRF(d) = 1 / (k + rank_a) + 1 / (k + rank_b)
    """
    rank_a = stats.rankdata(-scores_a, method="average")
    rank_b = stats.rankdata(-scores_b, method="average")
    return (1.0 / (k + rank_a)) + (1.0 / (k + rank_b))


def compute_mpr(scores_a, scores_b):
    """
    Mean Percentile Rank combination of two score arrays.
    Scores are converted to percentiles in [0.0, 1.0] where 1.0 is highest rank.
    """
    n = len(scores_a)
    rank_a = stats.rankdata(-scores_a, method="average")
    rank_b = stats.rankdata(-scores_b, method="average")
    pct_a = 1.0 - (rank_a - 1.0) / max(1.0, float(n - 1))
    pct_b = 1.0 - (rank_b - 1.0) / max(1.0, float(n - 1))
    return 0.5 * (pct_a + pct_b)


def compute_zscore_sum(scores_a, scores_b):
    """Standardized Z-Score combination: z_a + z_b."""
    z_a = stats.zscore(scores_a)
    z_b = stats.zscore(scores_b)
    return z_a + z_b


def main():
    print("=" * 85)
    print("PHASE 5: DATASET ALIGNMENT & MULTI-MODAL INTEGRATION ENGINE")
    print("=" * 85)

    folds_path = "data/processed/stratified_cluster_folds.csv"
    vina_path = "reports/vina_docking_scores.csv"

    if not os.path.exists(folds_path) or not os.path.exists(vina_path):
        print("[ERROR] Required input files missing!", file=sys.stderr)
        sys.exit(1)

    df_folds = pd.read_csv(folds_path)
    df_vina = pd.read_csv(vina_path)

    # Filter strictly to the 93 non-covalent cohort
    df_merged = pd.merge(
        df_folds[~df_folds["nitro_aromatic_warhead"]].copy(),
        df_vina[["molecule_chembl_ids", "vina_affinity", "vina_score", "status"]],
        on="molecule_chembl_ids"
    ).reset_index(drop=True)

    print(f"[INFO] Aligned {len(df_merged)} non-covalent molecules.")
    print(f"  - Active:   {(df_merged['label'] == 'Active').sum()}")
    print(f"  - Inactive: {(df_merged['label'] == 'Inactive').sum()}")
    print(f"  - Hydantoins: {df_merged['is_hydantoin'].sum()}")
    print(f"  - Non-Hydantoins: {(~df_merged['is_hydantoin']).sum()}")

    # 1. Physicochemical & Structural Descriptors
    mols = [Chem.MolFromSmiles(s) for s in df_merged["canonical_smiles"]]
    heavy_atoms = [m.GetNumHeavyAtoms() for m in mols]
    mol_wts = [float(Descriptors.MolWt(m)) for m in mols]

    df_merged["heavy_atom_count"] = heavy_atoms
    df_merged["molecular_weight"] = np.round(mol_wts, 2)
    df_merged["ligand_efficiency"] = np.round(df_merged["vina_score"] / df_merged["heavy_atom_count"], 4)

    y = (df_merged["label"] == "Active").astype(int).values
    is_hyd = df_merged["is_hydantoin"].values
    smiles = df_merged["canonical_smiles"].tolist()

    # Precompute Morgan count fingerprints and bit vectors
    print("[INFO] Computing molecular representations...")
    X_counts = compute_morgan_fingerprints(smiles, radius=2, n_bits=2048, as_counts=True)
    rdkit_fps = compute_rdkit_bit_fingerprints(smiles, radius=2, n_bits=2048)

    # 2. Track B: Stratified Cluster 5-Fold OOF Predictions
    print("[INFO] Executing Track B (Cluster 5-Fold CV) OOF inferences...")
    prob_rf_cluster = np.zeros(len(df_merged), dtype=float)
    prob_lr_cluster = np.zeros(len(df_merged), dtype=float)
    prob_tanimoto_cluster = np.zeros(len(df_merged), dtype=float)
    sim_to_train_cluster = np.zeros(len(df_merged), dtype=float)

    for f in range(5):
        train_idx = np.where(df_merged["cluster_fold"] != f)[0]
        test_idx = np.where(df_merged["cluster_fold"] == f)[0]

        # Random Forest
        rf = get_model("Random Forest", feature_type="ecfp4_counts", seed=42 + f)
        rf.fit(X_counts[train_idx], y[train_idx])
        prob_rf_cluster[test_idx] = rf.predict_proba(X_counts[test_idx])[:, 1]

        # Logistic Regression (tuned C)
        lr = get_model("Logistic Regression", feature_type="ecfp4_counts", seed=42 + f)
        lr.fit(X_counts[train_idx], y[train_idx])
        prob_lr_cluster[test_idx] = lr.predict_proba(X_counts[test_idx])[:, 1]

        # 1-NN Tanimoto
        train_fps = [rdkit_fps[i] for i in train_idx]
        test_fps = [rdkit_fps[i] for i in test_idx]
        t1nn = Tanimoto1NNClassifier().fit(train_fps, y[train_idx])
        prob_tanimoto_cluster[test_idx] = t1nn.predict_proba(test_fps)[:, 1]

        # Applicability domain similarity
        sim_to_train_cluster[test_idx] = compute_max_tanimoto_to_train(test_fps, train_fps)

    df_merged["prob_rf_cluster"] = np.round(prob_rf_cluster, 4)
    df_merged["prob_lr_cluster"] = np.round(prob_lr_cluster, 4)
    df_merged["prob_tanimoto_cluster"] = np.round(prob_tanimoto_cluster, 4)
    df_merged["sim_to_train_cluster"] = np.round(sim_to_train_cluster, 4)

    # 3. Track C: Leave-Hydantoin-Out Inferences
    print("[INFO] Executing Track C (Leave-Hydantoin-Out) cross-chemotype inferences...")
    prob_rf_lho = np.zeros(len(df_merged), dtype=float)
    prob_lr_lho = np.zeros(len(df_merged), dtype=float)
    prob_tanimoto_lho = np.zeros(len(df_merged), dtype=float)

    # Split 1: Train Non-Hyd (N=43) -> Test Hyd (N=50)
    train_idx_1 = np.where(~is_hyd)[0]
    test_idx_1 = np.where(is_hyd)[0]

    rf_s1 = get_model("Random Forest", feature_type="ecfp4_counts", seed=42).fit(X_counts[train_idx_1], y[train_idx_1])
    lr_s1 = get_model("Logistic Regression", feature_type="ecfp4_counts", seed=42).fit(X_counts[train_idx_1], y[train_idx_1])
    t_s1 = Tanimoto1NNClassifier().fit([rdkit_fps[i] for i in train_idx_1], y[train_idx_1])

    prob_rf_lho[test_idx_1] = rf_s1.predict_proba(X_counts[test_idx_1])[:, 1]
    prob_lr_lho[test_idx_1] = lr_s1.predict_proba(X_counts[test_idx_1])[:, 1]
    prob_tanimoto_lho[test_idx_1] = t_s1.predict_proba([rdkit_fps[i] for i in test_idx_1])[:, 1]

    # Split 2: Train Hyd (N=50) -> Test Non-Hyd (N=43)
    train_idx_2 = np.where(is_hyd)[0]
    test_idx_2 = np.where(~is_hyd)[0]

    rf_s2 = get_model("Random Forest", feature_type="ecfp4_counts", seed=42).fit(X_counts[train_idx_2], y[train_idx_2])
    lr_s2 = get_model("Logistic Regression", feature_type="ecfp4_counts", seed=42).fit(X_counts[train_idx_2], y[train_idx_2])
    t_s2 = Tanimoto1NNClassifier().fit([rdkit_fps[i] for i in train_idx_2], y[train_idx_2])

    prob_rf_lho[test_idx_2] = rf_s2.predict_proba(X_counts[test_idx_2])[:, 1]
    prob_lr_lho[test_idx_2] = lr_s2.predict_proba(X_counts[test_idx_2])[:, 1]
    prob_tanimoto_lho[test_idx_2] = t_s2.predict_proba([rdkit_fps[i] for i in test_idx_2])[:, 1]

    df_merged["prob_rf_lho"] = np.round(prob_rf_lho, 4)
    df_merged["prob_lr_lho"] = np.round(prob_lr_lho, 4)
    df_merged["prob_tanimoto_lho"] = np.round(prob_tanimoto_lho, 4)

    # 4. Multi-Modal Rank & Score Combinations
    print("[INFO] Computing rank fusion and score combinations (RRF, MPR, Z-Score)...")
    vina_score = df_merged["vina_score"].values
    le_score = df_merged["ligand_efficiency"].values

    # Track B Combinations
    df_merged["rrf_rf_vina_cluster"] = np.round(compute_rrf(prob_rf_cluster, vina_score, k=60), 6)
    df_merged["rrf_lr_vina_cluster"] = np.round(compute_rrf(prob_lr_cluster, vina_score, k=60), 6)
    df_merged["rrf_rf_le_cluster"] = np.round(compute_rrf(prob_rf_cluster, le_score, k=60), 6)
    df_merged["mpr_rf_vina_cluster"] = np.round(compute_mpr(prob_rf_cluster, vina_score), 6)
    df_merged["mpr_lr_vina_cluster"] = np.round(compute_mpr(prob_lr_cluster, vina_score), 6)
    df_merged["zscore_rf_vina_cluster"] = np.round(compute_zscore_sum(prob_rf_cluster, vina_score), 4)

    # Track C (LHO) Combinations
    df_merged["rrf_rf_vina_lho"] = np.round(compute_rrf(prob_rf_lho, vina_score, k=60), 6)
    df_merged["rrf_lr_vina_lho"] = np.round(compute_rrf(prob_lr_lho, vina_score, k=60), 6)
    df_merged["rrf_rf_le_lho"] = np.round(compute_rrf(prob_rf_lho, le_score, k=60), 6)
    df_merged["mpr_rf_vina_lho"] = np.round(compute_mpr(prob_rf_lho, vina_score), 6)
    df_merged["mpr_lr_vina_lho"] = np.round(compute_mpr(prob_lr_lho, vina_score), 6)
    df_merged["zscore_rf_vina_lho"] = np.round(compute_zscore_sum(prob_rf_lho, vina_score), 4)

    # Reorder columns cleanly
    cols_order = [
        "molecule_chembl_ids", "canonical_smiles", "label", "is_hydantoin", "cluster_fold",
        "molecular_weight", "heavy_atom_count", "vina_affinity", "vina_score", "ligand_efficiency",
        "sim_to_train_cluster",
        "prob_rf_cluster", "prob_lr_cluster", "prob_tanimoto_cluster",
        "prob_rf_lho", "prob_lr_lho", "prob_tanimoto_lho",
        "rrf_rf_vina_cluster", "rrf_lr_vina_cluster", "rrf_rf_le_cluster", "mpr_rf_vina_cluster", "zscore_rf_vina_cluster",
        "rrf_rf_vina_lho", "rrf_lr_vina_lho", "rrf_rf_le_lho", "mpr_rf_vina_lho", "zscore_rf_vina_lho"
    ]
    df_out = df_merged[cols_order].rename(columns={"cluster_fold": "fold"})

    out_csv = "data/processed/master_predictions_phase5.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved master integrated predictions to: {out_csv}")
    print("=" * 85)


if __name__ == "__main__":
    main()
