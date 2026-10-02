#!/usr/bin/env python3
"""
src/audit_covalent.py

Audits DprE1 chemical structures for covalent suicide-inhibitor warheads:
  1. Nitro-aromatics reduced by FAD to attack Cys387: c[N+](=O)[O-]
  2. Classic 1,3-benzothiazin-4-one (BTZ) core: c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2

Partitions the dataset into:
  - Non-covalent evaluation set (eligible for standard rigid AutoDock Vina docking)
  - Covalent set (suicide-inhibitors requiring pre-reaction proximity modeling or covalent docking)

Saves detailed compound-level audit and summary table to reports/covalent_audit.csv.
"""

import os
import sys
import pandas as pd
from rdkit import Chem


def detect_covalent_motifs(smiles: str):
    """
    Evaluates SMARTS patterns for covalent DprE1 mechanisms.
    """
    if not smiles or pd.isna(smiles):
        return False, "Invalid SMILES"
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False, "RDKit parse error"

    # 1. Classic 1,3-benzothiazin-4-one core (BTZ)
    smarts_btz = "c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2"
    pat_btz = Chem.MolFromSmarts(smarts_btz)

    # 2. General nitro-aromatic warhead (activated aromatic nitro reduced by FAD)
    smarts_nitro_aromatic = "c[N+](=O)[O-]"
    pat_nitro_aro = Chem.MolFromSmarts(smarts_nitro_aromatic)

    # 3. Nitroso / hydroxylamine / covalent adduct forms
    smarts_nitroso = "c[N]=O"
    pat_nitroso = Chem.MolFromSmarts(smarts_nitroso)

    has_btz = mol.HasSubstructMatch(pat_btz)
    has_nitro_aro = mol.HasSubstructMatch(pat_nitro_aro)
    has_nitroso = mol.HasSubstructMatch(pat_nitroso)

    if has_btz:
        return True, "1,3-Benzothiazin-4-one (BTZ) core"
    elif has_nitro_aro:
        return True, "Aromatic nitro warhead (FAD-reduced Cys387 adduct)"
    elif has_nitroso:
        return True, "Aromatic nitroso reactive intermediate"
    else:
        return False, "Non-Covalent"


def main():
    print("=" * 80)
    print("COVALENT VS. NON-COVALENT SUBSTRUCTURE AUDIT (DprE1 Benchmark)")
    print("=" * 80)

    input_paths = [
        "data/interim/gate1_compounds.csv",
        "data/processed/gate1_compounds.csv"
    ]
    input_csv = None
    for p in input_paths:
        if os.path.exists(p):
            input_csv = p
            break

    if not input_csv:
        print(f"[ERROR] Could not find gate1_compounds.csv in data/interim/ or data/processed/!", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Ingesting curated compounds from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {len(df)} total curated compounds.")

    covalent_flags = []
    mechanism_types = []
    for _, row in df.iterrows():
        smi = row["canonical_smiles"]
        is_cov, mech = detect_covalent_motifs(smi)
        covalent_flags.append(is_cov)
        mechanism_types.append(mech)

    df["is_covalent"] = covalent_flags
    df["mechanism_annotation"] = mechanism_types

    # Partition by primary benchmark labeling
    df_labeled = df[df["label"].isin(["Active", "Inactive"])].copy()
    df_gray = df[df["label"] == "Gray Zone"].copy()

    # Breakdown metrics
    n_act = (df_labeled["label"] == "Active").sum()
    n_inact = (df_labeled["label"] == "Inactive").sum()
    n_gray = len(df_gray)

    act_cov = ((df_labeled["label"] == "Active") & (df_labeled["is_covalent"])).sum()
    act_non_cov = ((df_labeled["label"] == "Active") & (~df_labeled["is_covalent"])).sum()

    inact_cov = ((df_labeled["label"] == "Inactive") & (df_labeled["is_covalent"])).sum()
    inact_non_cov = ((df_labeled["label"] == "Inactive") & (~df_labeled["is_covalent"])).sum()

    gray_cov = (df_gray["is_covalent"]).sum()
    gray_non_cov = (~df_gray["is_covalent"]).sum()

    total_non_covalent_labeled = act_non_cov + inact_non_cov
    total_covalent_labeled = act_cov + inact_cov

    print("\n" + "=" * 80)
    print("SUBSTRUCTURE MECHANISM CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"{'Activity Category':<25} | {'Non-Covalent':<15} | {'Covalent':<12} | {'Total':<10}")
    print("-" * 80)
    print(f"{'Active (pIC50 >= 6.0)':<25} | {act_non_cov:<15} | {act_cov:<12} | {n_act:<10}")
    print(f"{'Inactive (pIC50 < 5.0)':<25} | {inact_non_cov:<15} | {inact_cov:<12} | {n_inact:<10}")
    print(f"{'Subtotal (Labeled Benchmark)':<25} | {total_non_covalent_labeled:<15} | {total_covalent_labeled:<12} | {len(df_labeled):<10}")
    print(f"{'Gray Zone (5.0 <= pIC50 < 6.0)':<25} | {gray_non_cov:<15} | {gray_cov:<12} | {n_gray:<10}")
    print("-" * 80)
    print(f"{'Total Dataset':<25} | {(~df['is_covalent']).sum():<15} | {df['is_covalent'].sum():<12} | {len(df):<10}")
    print("=" * 80)

    print(f"\n[KEY FINDING] Non-Covalent Molecules Eligible for Rigid Vina Docking: {total_non_covalent_labeled}")
    print(f"  - Active Non-Covalent:   {act_non_cov}")
    print(f"  - Inactive Non-Covalent: {inact_non_cov}")
    print(f"  - Covalent Subset (Separate proximity track): {total_covalent_labeled}")

    # Save to reports/covalent_audit.csv
    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "covalent_audit.csv")

    cols_to_save = [
        "canonical_smiles", "molecule_chembl_ids", "median_pic50",
        "median_nm", "label", "is_covalent", "mechanism_annotation",
        "murcko_scaffold", "n_measurements"
    ]
    df[cols_to_save].to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved full covalent audit catalog to: {out_path}")

    # Also update interim/processed datasets with is_covalent flag for Phase 2
    for p in ["data/interim/gate1_compounds.csv", "data/processed/gate1_compounds.csv"]:
        if os.path.exists(p):
            df.to_csv(p, index=False)
            print(f"[SUCCESS] Updated {p} with is_covalent column.")


if __name__ == "__main__":
    main()
