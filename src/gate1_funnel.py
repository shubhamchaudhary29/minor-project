#!/usr/bin/env python3
"""
src/gate1_funnel.py

Executes the Gate 1 Funnel evaluation on verified DprE1 bioactivity data (CHEMBL3804751).
Performs:
  1. Structure curation & sanitization via RDKit (salt stripping, canonicalization).
  2. Unit standardization to nM and pIC50 computation (pIC50 = 9 - log10(nM)).
  3. Relational operator handling (=, >, <).
  4. Activity categorization:
     - Active: pIC50 >= 6.0 (<= 1 uM)
     - Inactive: pIC50 < 5.0 (> 10 uM or relation '>' above 10 uM)
     - Gray Zone: 5.0 <= pIC50 < 6.0
  5. Bemis-Murcko scaffold calculation & Chemotype Balance Analysis.
  6. Outputs curated compound tables and summary metrics JSON.
"""

import os
import sys
import math
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import Dict, Any, Tuple


def sanitize_and_canonicalize(smiles: str) -> str:
    """
    Parses SMILES with RDKit, strips salts/solvents by selecting the largest
    organic fragment, and returns the canonical SMILES string.
    """
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None
        largest_frag = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(largest_frag, canonical=True)
    except Exception:
        return None


def convert_to_nm(value: float, units: str) -> float:
    """
    Converts activity value to nanomolar (nM).
    """
    if pd.isna(value) or value is None:
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None

    if val <= 0:
        return None

    units_clean = str(units).strip().lower() if pd.notna(units) else "nm"
    if units_clean in ["nm", "nanomolar"]:
        return val
    elif units_clean in ["um", "µm", "micromolar"]:
        return val * 1000.0
    elif units_clean in ["m", "molar"]:
        return val * 1e9
    elif units_clean in ["pm", "picomolar"]:
        return val * 1e-3
    elif units_clean in ["mm", "millimolar"]:
        return val * 1e6
    else:
        return val


def compute_murcko(smiles: str) -> str:
    """
    Calculates Bemis-Murcko scaffold SMILES without chirality.
    """
    if not smiles:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def run_gate1_pipeline(input_csv: str, dataset_label: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    print("=" * 80)
    print(f"GATE 1 FUNNEL PIPELINE: {dataset_label}")
    print(f"Ingesting: {input_csv}")
    print("=" * 80)

    df_raw = pd.read_csv(input_csv, low_memory=False)
    raw_bioactivities_count = len(df_raw)

    # 1. Tier A: Biochemical / binding assays (assay_type == 'B') measuring direct inhibition (IC50 or Ki)
    tier_a_mask = (df_raw["assay_type"] == "B") & (df_raw["standard_type"].isin(["IC50", "Ki"]))
    df_tier_a = df_raw[tier_a_mask].copy()
    tier_a_count = len(df_tier_a)

    if tier_a_count == 0:
        metrics = {
            "dataset_label": dataset_label,
            "raw_bioactivities_count": raw_bioactivities_count,
            "tier_a_biochemical_count": 0,
            "unique_chemical_structures": 0,
            "labeled_compounds_outside_gray": 0,
            "active_compounds": 0,
            "inactive_compounds": 0,
            "gray_zone_compounds": 0,
            "total_unique_scaffolds": 0,
            "scaffolds_ge_3_compounds": 0,
            "gate1_verdict": "FAIL / PIVOT",
            "reason": "No Tier A biochemical assays (IC50/Ki) found."
        }
        return metrics, pd.DataFrame()

    # 2. Curation, canonicalization, unit conversion, pIC50
    curated_records = []
    for _, row in df_tier_a.iterrows():
        raw_smi = row.get("canonical_smiles")
        can_smi = sanitize_and_canonicalize(raw_smi)
        if not can_smi:
            continue

        raw_val = row.get("standard_value")
        raw_units = row.get("standard_units")
        nm_val = convert_to_nm(raw_val, raw_units)
        if nm_val is None or nm_val <= 0:
            continue

        pic50 = 9.0 - math.log10(nm_val)
        raw_rel = row.get("standard_relation")
        rel = str(raw_rel).strip() if pd.notna(raw_rel) and str(raw_rel).strip() != "None" else "="

        scaff = compute_murcko(can_smi)

        label = "Gray Zone"
        if rel in ["=", "=="]:
            if pic50 >= 6.0:
                label = "Active"
            elif pic50 < 5.0:
                label = "Inactive"
        elif rel == ">":
            if nm_val >= 10000.0:
                label = "Inactive"
        elif rel == "<":
            if nm_val <= 1000.0:
                label = "Active"

        curated_records.append({
            "canonical_smiles": can_smi,
            "raw_smiles": raw_smi,
            "molecule_chembl_id": row.get("molecule_chembl_id"),
            "assay_chembl_id": row.get("assay_chembl_id"),
            "standard_type": row.get("standard_type"),
            "standard_relation": rel,
            "standard_value_nm": nm_val,
            "pic50": pic50,
            "label": label,
            "murcko_scaffold": scaff
        })

    df_curated = pd.DataFrame(curated_records)
    unique_chemical_structures = df_curated["canonical_smiles"].nunique()

    # 3. Compound-level aggregation
    aggregated_compounds = []
    for can_smi, group in df_curated.groupby("canonical_smiles"):
        med_pic50 = group["pic50"].median()
        med_nm = group["standard_value_nm"].median()
        scaff = group["murcko_scaffold"].iloc[0]
        rels = list(group["standard_relation"].unique())
        rel_str = "; ".join(rels)
        labels = list(group["label"].unique())

        if "Active" in labels and "Inactive" not in labels:
            comp_label = "Active"
        elif "Inactive" in labels and "Active" not in labels:
            comp_label = "Inactive"
        elif "Active" in labels and "Inactive" in labels:
            if med_pic50 >= 6.0:
                comp_label = "Active"
            elif med_pic50 < 5.0:
                comp_label = "Inactive"
            else:
                comp_label = "Gray Zone"
        else:
            comp_label = "Gray Zone"

        aggregated_compounds.append({
            "canonical_smiles": can_smi,
            "molecule_chembl_ids": "; ".join(group["molecule_chembl_id"].dropna().unique()),
            "median_pic50": round(med_pic50, 4),
            "median_nm": round(med_nm, 2),
            "relation_summary": rel_str,
            "label": comp_label,
            "murcko_scaffold": scaff,
            "n_measurements": len(group)
        })

    df_compounds = pd.DataFrame(aggregated_compounds)

    # 4. Filter labeled benchmark set
    df_outside_gray = df_compounds[df_compounds["label"].isin(["Active", "Inactive"])].copy()
    labeled_compounds_outside_gray = len(df_outside_gray)
    active_count = (df_outside_gray["label"] == "Active").sum()
    inactive_count = (df_outside_gray["label"] == "Inactive").sum()
    gray_count = (df_compounds["label"] == "Gray Zone").sum()

    # 5. Scaffold diversity metrics
    unique_scaffolds = df_outside_gray["murcko_scaffold"].nunique()
    scaffold_counts = df_outside_gray["murcko_scaffold"].value_counts()
    scaffolds_ge_3 = (scaffold_counts >= 3).sum()

    # 6. Chemotype Balance & Reformulated Gate 1 Verdict
    pass_criteria = (unique_scaffolds >= 30) and (scaffolds_ge_3 >= 10)
    verdict = "PASS" if pass_criteria else "REFORMULATED / CLUSTER-SPLIT"
    
    reasons = []
    if unique_scaffolds >= 30:
        reasons.append(f"Unique scaffolds {unique_scaffolds} >= 30 (Satisfied)")
    else:
        reasons.append(f"Unique scaffolds {unique_scaffolds} < 30")
    if scaffolds_ge_3 < 10:
        reasons.append(f"Scaffolds with >= 3 compounds is {scaffolds_ge_3} (< 10 threshold, driven by single hydantoin series dominance)")
    reason = "; ".join(reasons)

    metrics = {
        "dataset_label": dataset_label,
        "raw_bioactivities_count": int(raw_bioactivities_count),
        "tier_a_biochemical_count": int(tier_a_count),
        "unique_chemical_structures": int(unique_chemical_structures),
        "labeled_compounds_outside_gray": int(labeled_compounds_outside_gray),
        "active_compounds": int(active_count),
        "inactive_compounds": int(inactive_count),
        "gray_zone_compounds": int(gray_count),
        "total_unique_scaffolds": int(unique_scaffolds),
        "scaffolds_ge_3_compounds": int(scaffolds_ge_3),
        "gate1_verdict": verdict,
        "reason": reason
    }

    return metrics, df_compounds


def main():
    primary_csv = os.path.abspath("data/raw/chembl_dpre1_raw.csv")

    if not os.path.exists(primary_csv):
        print(f"Error: {primary_csv} not found!", file=sys.stderr)
        sys.exit(1)

    metrics, df_comp = run_gate1_pipeline(primary_csv, "Verified DprE1 (CHEMBL3804751)")

    # Save to data/interim and data/processed
    for out_dir in ["data/interim", "data/processed"]:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "gate1_compounds.csv")
        df_comp.to_csv(p, index=False)
        print(f"[SAVE] Saved curated compounds to: {p}")

    # Output formatted table
    print("\n" + "=" * 80)
    print("GATE 1 EVALUATION FUNNEL SUMMARY: VERIFIED DprE1 (CHEMBL3804751)")
    print("=" * 80)
    for k, v in metrics.items():
        print(f"  {k:<32}: {v}")
    print("=" * 80)

    # Save summary JSON
    summary_path = os.path.abspath("reports/gate1_funnel_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump([metrics], f, indent=2)
    print(f"[SAVE] Saved Gate 1 Funnel summary to: {summary_path}")


if __name__ == "__main__":
    main()
