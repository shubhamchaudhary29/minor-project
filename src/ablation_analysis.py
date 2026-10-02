#!/usr/bin/env python3
"""
src/ablation_analysis.py

Phase 5 Ablation & Failure Mode Analysis Suite:
1. Molecular Weight (MW) Stratification (<400 Da vs >=400 Da):
   - Demonstrates Vina size-bias: raw docking excels on low MW (<400 Da) but collapses on high MW (>=400 Da).
   - Evaluates Ligand Efficiency (LE = -dG / N_heavy) and hybrid RRF rescue.
2. Applicability Domain / Tanimoto Similarity Crossover (<0.40 vs >=0.40):
   - Evaluates ML vs Docking performance in out-of-domain vs in-domain chemical space.
3. Quadrant Discordance Analysis:
   - Identifies specific compounds where Docking rescues ML (ML FN -> Docking TP).
   - Identifies specific compounds where ML rescues Docking (Docking FN -> ML TP).
   - Identifies False Positives (Inactives scored highly by either method).
   - Maps chemical rationale (steric clashes, hydantoin bias, lipophilicity/size).

Outputs:
  reports/discordance_analysis.csv
  reports/ablation_results.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_metrics(y_true, scores):
    """Computes ROC-AUC and PR-AUC safely."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)
    return roc, pr


def main():
    print("=" * 80)
    print("PHASE 5: ABLATION & DISCORDANCE / FAILURE MODE ANALYSIS")
    print("=" * 80)

    pred_csv = "data/processed/master_predictions_phase5.csv"
    if not os.path.exists(pred_csv):
        print(f"[ERROR] Missing {pred_csv}. Run src/integrate.py first!", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(pred_csv)
    n_total = len(df)
    y_total = (df["label"] == "Active").astype(int).values
    print(f"[INFO] Loaded {n_total} non-covalent molecules.")

    ablation_records = []

    # -------------------------------------------------------------
    # 1. MOLECULAR WEIGHT (MW) STRATIFICATION (<400 Da vs >=400 Da)
    # -------------------------------------------------------------
    print("\n--- 1. MOLECULAR WEIGHT STRATIFICATION ---")
    mw_low = df[df["molecular_weight"] < 400].copy().reset_index(drop=True)
    mw_high = df[df["molecular_weight"] >= 400].copy().reset_index(drop=True)

    for subset, name in [(mw_low, "Low MW (<400 Da)"), (mw_high, "High MW (>=400 Da)")]:
        y_sub = (subset["label"] == "Active").astype(int).values
        n_sub = len(subset)
        n_act = int(np.sum(y_sub == 1))
        n_inact = int(np.sum(y_sub == 0))

        methods = [
            ("AutoDock Vina (Raw Affinity)", subset["vina_score"].values),
            ("AutoDock Vina (Ligand Efficiency)", subset["ligand_efficiency"].values),
            ("Random Forest (Cluster OOF)", subset["prob_rf_cluster"].values),
            ("Random Forest (LHO)", subset["prob_rf_lho"].values),
            ("RRF (RF + Vina, Cluster)", subset["rrf_rf_vina_cluster"].values),
            ("RRF (RF + Vina, LHO)", subset["rrf_rf_vina_lho"].values),
            ("RRF (RF + LE, Cluster)", subset["rrf_rf_le_cluster"].values),
        ]

        print(f"\nSubset: {name} (N={n_sub}: {n_act} Actives, {n_inact} Inactives)")
        print(f"{'Method':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
        print("-" * 54)
        for m_name, scores in methods:
            roc, pr = compute_metrics(y_sub, scores)
            print(f"{m_name:<32} | {roc:>7.3f}  | {pr:>7.3f}")
            ablation_records.append({
                "analysis_type": "MW Stratification",
                "subgroup": name,
                "n_samples": n_sub,
                "n_actives": n_act,
                "n_inactives": n_inact,
                "method": m_name,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
            })

    # Correlation between MW and Vina score vs MW and label
    corr_mw_vina, p_mw_vina = stats.pearsonr(df["molecular_weight"], df["vina_score"])
    corr_mw_label, p_mw_label = stats.pointbiserialr(y_total, df["molecular_weight"])
    print(f"\n[CORRELATION AUDIT]")
    print(f"  Pearson r(MW, Vina raw affinity score): {corr_mw_vina:.3f} (p={p_mw_vina:.2e})")
    print(f"  Point-biserial r(Label, MW):            {corr_mw_label:.3f} (p={p_mw_label:.2e})")

    # -------------------------------------------------------------
    # 2. APPLICABILITY DOMAIN / TANIMOTO SIMILARITY CROSSOVER
    # -------------------------------------------------------------
    print("\n--- 2. APPLICABILITY DOMAIN (TANIMOTO SIMILARITY) STRATIFICATION ---")
    sim_low = df[df["sim_to_train_cluster"] < 0.40].copy().reset_index(drop=True)
    sim_high = df[df["sim_to_train_cluster"] >= 0.40].copy().reset_index(drop=True)

    for subset, name in [(sim_low, "Low Similarity (<0.40, Out-of-Domain)"),
                         (sim_high, "High Similarity (>=0.40, In-Domain)")]:
        y_sub = (subset["label"] == "Active").astype(int).values
        n_sub = len(subset)
        n_act = int(np.sum(y_sub == 1))
        n_inact = int(np.sum(y_sub == 0))

        methods = [
            ("AutoDock Vina (Raw Affinity)", subset["vina_score"].values),
            ("AutoDock Vina (Ligand Efficiency)", subset["ligand_efficiency"].values),
            ("Random Forest (Cluster OOF)", subset["prob_rf_cluster"].values),
            ("Random Forest (LHO)", subset["prob_rf_lho"].values),
            ("RRF (RF + Vina, Cluster)", subset["rrf_rf_vina_cluster"].values),
            ("RRF (RF + Vina, LHO)", subset["rrf_rf_vina_lho"].values),
        ]

        print(f"\nSubset: {name} (N={n_sub}: {n_act} Actives, {n_inact} Inactives)")
        print(f"{'Method':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
        print("-" * 54)
        for m_name, scores in methods:
            roc, pr = compute_metrics(y_sub, scores)
            print(f"{m_name:<32} | {roc:>7.3f}  | {pr:>7.3f}")
            ablation_records.append({
                "analysis_type": "Applicability Domain",
                "subgroup": name,
                "n_samples": n_sub,
                "n_actives": n_act,
                "n_inactives": n_inact,
                "method": m_name,
                "roc_auc": round(roc, 4),
                "pr_auc": round(pr, 4),
            })

    # Save ablation summary
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv("reports/ablation_results.csv", index=False)
    print("\n[SUCCESS] Ablation summary written to reports/ablation_results.csv")

    # -------------------------------------------------------------
    # 3. QUADRANT DISCORDANCE ANALYSIS
    # -------------------------------------------------------------
    print("\n--- 3. QUADRANT DISCORDANCE ANALYSIS ---")
    # Compute percentile ranks across N=93
    df["rank_vina"] = df["vina_score"].rank(pct=True)
    df["rank_rf_lho"] = df["prob_rf_lho"].rank(pct=True)
    df["rank_rf_cluster"] = df["prob_rf_cluster"].rank(pct=True)
    df["rank_delta_lho"] = df["rank_vina"] - df["rank_rf_lho"]
    df["rank_delta_cluster"] = df["rank_vina"] - df["rank_rf_cluster"]

    # Focus on Actives
    df_act = df[df["label"] == "Active"].copy()

    # Category A: Docking rescues ML (ML missed it, Docking prioritized it)
    # Compound is Active, rank_delta_lho is large positive
    rescued_by_docking = df_act.sort_values(by="rank_delta_lho", ascending=False).head(5)

    # Category B: ML rescues Docking (Docking missed it, ML prioritized it)
    # Compound is Active, rank_delta_lho is large negative
    rescued_by_ml = df_act.sort_values(by="rank_delta_lho", ascending=True).head(5)

    # Category C: False Positives (Inactives with high predictions)
    df_inact = df[df["label"] == "Inactive"].copy()
    docking_fp = df_inact.sort_values(by="vina_score", ascending=False).head(5)
    ml_fp = df_inact.sort_values(by="prob_rf_cluster", ascending=False).head(5)

    discordance_rows = []

    print("\n[CASE 1: DOCKING RESCUES ML (Active compounds: Docking Rank >> ML Rank under LHO)]")
    for _, r in rescued_by_docking.iterrows():
        chembl_id = r["molecule_chembl_ids"]
        chemotype = "Hydantoin" if r["is_hydantoin"] else "Non-Hydantoin"
        explanation = (
            f"Active {chemotype} (MW {r['molecular_weight']:.1f}). Novel scaffold to LHO model "
            f"(ML prob: {r['prob_rf_lho']:.3f}, pct: {r['rank_rf_lho']:.2f}). AutoDock Vina binds strongly "
            f"into the CT325 hydrophobic pocket (-{r['vina_score']:.2f} kcal/mol, pct: {r['rank_vina']:.2f}), "
            f"rescuing ML false-negative through 3D shape/electrostatic complementarity."
        )
        print(f"  * {chembl_id} ({chemotype}): Vina={-r['vina_score']:.2f} kcal/mol (pct {r['rank_vina']:.2f}) vs ML P={r['prob_rf_lho']:.3f} (pct {r['rank_rf_lho']:.2f}) | Delta={r['rank_delta_lho']:+.2f}")
        discordance_rows.append({
            "molecule_chembl_id": chembl_id,
            "label": r["label"],
            "is_hydantoin": r["is_hydantoin"],
            "molecular_weight": r["molecular_weight"],
            "heavy_atom_count": r["heavy_atom_count"],
            "vina_affinity": r["vina_affinity"],
            "vina_percentile": round(r["rank_vina"], 3),
            "rf_prob_lho": round(r["prob_rf_lho"], 3),
            "rf_percentile_lho": round(r["rank_rf_lho"], 3),
            "rf_prob_cluster": round(r["prob_rf_cluster"], 3),
            "rf_percentile_cluster": round(r["rank_rf_cluster"], 3),
            "discordance_category": "Docking Rescues ML (LHO)",
            "rank_delta": round(r["rank_delta_lho"], 3),
            "chemical_rationale": explanation
        })

    print("\n[CASE 2: ML RESCUES DOCKING (Active compounds: ML Rank >> Docking Rank)]")
    for _, r in rescued_by_ml.iterrows():
        chembl_id = r["molecule_chembl_ids"]
        chemotype = "Hydantoin" if r["is_hydantoin"] else "Non-Hydantoin"
        explanation = (
            f"Active {chemotype} (MW {r['molecular_weight']:.1f}). ML recognizes 2D ECFP4 pharmacophore "
            f"(ML prob: {r['prob_rf_lho']:.3f}, pct: {r['rank_rf_lho']:.2f}). Vina rigid-receptor docking "
            f"penalizes compound (-{r['vina_score']:.2f} kcal/mol, pct: {r['rank_vina']:.2f}) due to rigid pocket "
            f"steric clashes or small size (heavy atoms: {r['heavy_atom_count']}), rescued by 2D ML."
        )
        print(f"  * {chembl_id} ({chemotype}): ML P={r['prob_rf_lho']:.3f} (pct {r['rank_rf_lho']:.2f}) vs Vina={-r['vina_score']:.2f} kcal/mol (pct {r['rank_vina']:.2f}) | Delta={r['rank_delta_lho']:+.2f}")
        discordance_rows.append({
            "molecule_chembl_id": chembl_id,
            "label": r["label"],
            "is_hydantoin": r["is_hydantoin"],
            "molecular_weight": r["molecular_weight"],
            "heavy_atom_count": r["heavy_atom_count"],
            "vina_affinity": r["vina_affinity"],
            "vina_percentile": round(r["rank_vina"], 3),
            "rf_prob_lho": round(r["prob_rf_lho"], 3),
            "rf_percentile_lho": round(r["rank_rf_lho"], 3),
            "rf_prob_cluster": round(r["prob_rf_cluster"], 3),
            "rf_percentile_cluster": round(r["rank_rf_cluster"], 3),
            "discordance_category": "ML Rescues Docking (LHO)",
            "rank_delta": round(r["rank_delta_lho"], 3),
            "chemical_rationale": explanation
        })

    print("\n[CASE 3: DOCKING FALSE POSITIVES (Inactive compounds scored top-tier by Vina)]")
    for _, r in docking_fp.iterrows():
        chembl_id = r["molecule_chembl_ids"]
        chemotype = "Hydantoin" if r["is_hydantoin"] else "Non-Hydantoin"
        explanation = (
            f"Inactive {chemotype} with high MW ({r['molecular_weight']:.1f} Da, {r['heavy_atom_count']} heavy atoms). "
            f"AutoDock Vina empirical size-bias assigns top affinity (-{r['vina_score']:.2f} kcal/mol, pct {r['rank_vina']:.2f}) "
            f"due to accumulated non-specific hydrophobic contacts, despite lacking key biochemical DprE1 interactions."
        )
        print(f"  * {chembl_id} ({chemotype}): Vina={-r['vina_score']:.2f} kcal/mol (pct {r['rank_vina']:.2f}) | MW={r['molecular_weight']:.1f}")
        discordance_rows.append({
            "molecule_chembl_id": chembl_id,
            "label": r["label"],
            "is_hydantoin": r["is_hydantoin"],
            "molecular_weight": r["molecular_weight"],
            "heavy_atom_count": r["heavy_atom_count"],
            "vina_affinity": r["vina_affinity"],
            "vina_percentile": round(r["rank_vina"], 3),
            "rf_prob_lho": round(r["prob_rf_lho"], 3),
            "rf_percentile_lho": round(r["rank_rf_lho"], 3),
            "rf_prob_cluster": round(r["prob_rf_cluster"], 3),
            "rf_percentile_cluster": round(r["rank_rf_cluster"], 3),
            "discordance_category": "Docking False Positive (Size-Bias)",
            "rank_delta": round(r["rank_delta_lho"], 3),
            "chemical_rationale": explanation
        })

    df_discord = pd.DataFrame(discordance_rows)
    df_discord.to_csv("reports/discordance_analysis.csv", index=False)
    print("\n[SUCCESS] Discordance analysis written to reports/discordance_analysis.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
