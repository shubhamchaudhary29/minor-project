#!/usr/bin/env python3
"""
src/analyze_hydantoins.py

Addresses the dominant hydantoin chemotype series in DprE1 biochemical data:
  1. Inspects the 34 compounds sharing the dominant hydantoin scaffold:
     O=C(CN1C(=O)NC(c2ccccc2)C1=O)c1ccccc1
  2. Quantifies the severe fold imbalance caused by naive Bemis-Murcko GroupKFold
     (34 compounds in one fold vs ~17 compounds across others).
  3. Implements a cluster-aware fold generator using Butina clustering (ECFP4, Tanimoto
     distance threshold 0.55) with greedy size-balancing to prevent intra-cluster leakage.
  4. Plans and prototypes the Leave-Hydantoin-Out (LHO) evaluation scheme:
     - Train on non-hydantoins (N=68) -> Test on hydantoins (N=34)
     - Train on hydantoins (N=34) -> Test on non-hydantoins (N=68)
  5. Saves fold assignments and analysis to reports/balanced_folds.csv and
     reports/hydantoin_cluster_analysis.json.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold


def get_ecfp4_fingerprints(smiles_list, radius=2, fp_size=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fps.append(gen.GetFingerprint(m))
    return fps


def butina_clustering(fps, dist_thresh=0.55):
    n = len(fps)
    dist_matrix = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dist_matrix.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dist_matrix, n, distThresh=dist_thresh, isDistData=True)
    return clusters


def greedy_balanced_split(clusters, df_labeled, n_splits=5):
    # Sort clusters descending by size
    sorted_clusters = sorted(clusters, key=len, reverse=True)
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    fold_actives = [0] * n_splits

    for c in sorted_clusters:
        # Assign entire cluster to fold with lowest compound count
        best_fold = int(np.argmin(fold_sizes))
        folds[best_fold].extend(c)
        fold_sizes[best_fold] += len(c)
        n_act = sum(1 for idx in c if df_labeled.loc[idx, "label"] == "Active")
        fold_actives[best_fold] += n_act

    # Create fold array
    fold_assignments = np.zeros(len(df_labeled), dtype=int)
    for f_idx, fold_members in enumerate(folds):
        for m_idx in fold_members:
            fold_assignments[m_idx] = f_idx

    return fold_assignments, fold_sizes, fold_actives


def main():
    print("=" * 80)
    print("HYDANTOIN CLUSTER ANALYSIS & BALANCED PARTITION STRATEGY")
    print("=" * 80)

    input_path = "data/interim/gate1_compounds.csv"
    if not os.path.exists(input_path):
        input_path = "data/processed/gate1_compounds.csv"
    df = pd.read_csv(input_path)

    # Focus on labeled evaluation set (Actives & Inactives outside gray zone)
    labeled = df[df["label"].isin(["Active", "Inactive"])].reset_index(drop=True)
    print(f"Total labeled compounds for benchmarking: {len(labeled)}")
    print(f"  - Active:   {(labeled['label'] == 'Active').sum()}")
    print(f"  - Inactive: {(labeled['label'] == 'Inactive').sum()}")

    # 1. Inspect Top Scaffolds
    scaff_counts = labeled["murcko_scaffold"].value_counts()
    dominant_scaff = scaff_counts.index[0]
    n_dominant = scaff_counts.iloc[0]
    print(f"\n[DOMINANT CHEMOTYPE DETECTED]")
    print(f"  Scaffold SMILES: {dominant_scaff}")
    print(f"  Compound Count:  {n_dominant} ({n_dominant / len(labeled) * 100:.1f}% of labeled set)")
    print(f"  Activity breakdown in dominant scaffold:")
    dom_df = labeled[labeled["murcko_scaffold"] == dominant_scaff]
    print(f"    - Active:   {(dom_df['label'] == 'Active').sum()}")
    print(f"    - Inactive: {(dom_df['label'] == 'Inactive').sum()}")

    # Hydantoin SMARTS verification
    hyd_smarts = "O=C1NC(=O)NC1"
    pat_hyd = Chem.MolFromSmarts(hyd_smarts)
    is_hydantoin_list = []
    for s in labeled["canonical_smiles"]:
        m = Chem.MolFromSmiles(s)
        is_hydantoin_list.append(m.HasSubstructMatch(pat_hyd) if m else False)
    labeled["is_hydantoin"] = is_hydantoin_list
    print(f"\nTotal labeled compounds matching hydantoin core: {sum(is_hydantoin_list)}")

    # 2. Simulate Naive Bemis-Murcko GroupKFold
    gkf = GroupKFold(n_splits=5)
    groups = labeled["murcko_scaffold"]
    naive_fold_sizes = []
    naive_fold_actives = []
    for train_idx, test_idx in gkf.split(labeled, labeled["label"], groups):
        naive_fold_sizes.append(len(test_idx))
        naive_fold_actives.append((labeled.loc[test_idx, "label"] == "Active").sum())

    print("\n--- Naive Bemis-Murcko GroupKFold (5-Fold) ---")
    for i in range(5):
        print(f"  Fold {i}: Size = {naive_fold_sizes[i]:>2} | Actives = {naive_fold_actives[i]:>2} | Inactives = {naive_fold_sizes[i] - naive_fold_actives[i]:>2}")
    print(f"  Standard Deviation of Fold Sizes: {np.std(naive_fold_sizes):.2f}")
    print("  Observation: Dominant scaffold forces 34 compounds into a single fold,")
    print("  leaving other folds starved (~17 compounds), causing severe train-test variance.")

    # 3. Butina Clustering + Greedy Size-Balanced Partitioning
    fps = get_ecfp4_fingerprints(labeled["canonical_smiles"])
    clusters = butina_clustering(fps, dist_thresh=0.55)
    print(f"\n--- Butina Clustering (ECFP4, Tanimoto Distance Threshold = 0.55) ---")
    print(f"  Total clusters identified: {len(clusters)}")
    top_cluster_sizes = [len(c) for c in clusters[:5]]
    print(f"  Top 5 cluster sizes: {top_cluster_sizes}")

    balanced_folds, bal_sizes, bal_actives = greedy_balanced_split(clusters, labeled, n_splits=5)
    labeled["cluster_fold"] = balanced_folds

    print("\n--- Cluster-Aware Balanced Partition (5-Fold) ---")
    for i in range(5):
        print(f"  Fold {i}: Size = {bal_sizes[i]:>2} | Actives = {bal_actives[i]:>2} | Inactives = {bal_sizes[i] - bal_actives[i]:>2}")
    print(f"  Zero intra-cluster leakage guaranteed across all folds.")

    # 4. Leave-Hydantoin-Out (LHO) Strategy Plan
    lho_train_a = labeled[~labeled["is_hydantoin"]].index.tolist()
    lho_test_a = labeled[labeled["is_hydantoin"]].index.tolist()

    lho_train_b = lho_test_a
    lho_test_b = lho_train_a

    print("\n--- Leave-Hydantoin-Out (LHO) Evaluation Scheme ---")
    print(f"  Track A (Train on Non-Hydantoins -> Test on Hydantoins):")
    print(f"    - Training Set:   {len(lho_train_a)} compounds (Actives: {(labeled.loc[lho_train_a, 'label'] == 'Active').sum()}, Inactives: {(labeled.loc[lho_train_a, 'label'] == 'Inactive').sum()})")
    print(f"    - Testing Set:    {len(lho_test_a)} compounds (Actives: {(labeled.loc[lho_test_a, 'label'] == 'Active').sum()}, Inactives: {(labeled.loc[lho_test_a, 'label'] == 'Inactive').sum()})")
    print(f"  Track B (Train on Hydantoins -> Test on Non-Hydantoins):")
    print(f"    - Training Set:   {len(lho_train_b)} compounds")
    print(f"    - Testing Set:    {len(lho_test_b)} compounds")
    print("  Rationale: Provides an uncompromised test of out-of-distribution chemotype transfer.")

    # Save outputs
    out_csv = "reports/balanced_folds.csv"
    labeled.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved balanced fold assignments to: {out_csv}")

    analysis_data = {
        "total_labeled_compounds": len(labeled),
        "dominant_scaffold": dominant_scaff,
        "dominant_scaffold_count": int(n_dominant),
        "hydantoin_substructure_matches": int(sum(is_hydantoin_list)),
        "naive_groupkfold_sizes": [int(x) for x in naive_fold_sizes],
        "butina_cluster_count": len(clusters),
        "balanced_cluster_fold_sizes": [int(x) for x in bal_sizes],
        "balanced_cluster_fold_actives": [int(x) for x in bal_actives],
        "lho_non_hydantoin_count": len(lho_train_a),
        "lho_hydantoin_count": len(lho_test_a)
    }
    out_json = "reports/hydantoin_cluster_analysis.json"
    with open(out_json, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"[SUCCESS] Saved hydantoin cluster analysis to: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
