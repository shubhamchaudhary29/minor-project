#!/usr/bin/env python3
"""
src/dock.py

Phase 4 Molecular Docking & Virtual Screening Pipeline for DprE1:
  1. Receptor Preparation (--step prep):
     - Downloads PDB 4P8K (2.49 Å resolution) if missing.
     - Selectively extracts Chain A protein residues + essential FAD cofactor.
     - Extracts crystallographic reference ligand Ty38c (PDB ID: 38C / CT325) to docking/ligands/crystal_38C.pdb.
     - Strips solvent and buffer ions; converts receptor to PDBQT with polar hydrogens via OpenBabel.
  2. Gate 2 Redocking Validation (--step redock):
     - Prepares flexible ligand probe from SMILES using ETKDGv3 + MMFF94 and Meeko.
     - Executes AutoDock Vina redocking into active site centroid (17.07, -20.26, 1.49) Å.
     - Computes symmetry-corrected heavy-atom RMSD (rdMolAlign.GetBestRMS) against crystal_38C.pdb.
     - Asserts Gate 2: RMSD < 2.0 Å. Records decision to reports/gate2_redocking.txt.
  3. Batch Virtual Screening (--step batch):
     - Docks all non-covalent labeled molecules (N=93) using Vina Python API (exhaustiveness=16, cpu=4).
     - Saves docked poses and binding affinities (kcal/mol) to reports/vina_docking_scores.csv.
  4. Enrichment & Size-Bias Evaluation (--step evaluate):
     - Computes PR-AUC, ROC-AUC, and EF_10% with 1,000 bootstrap iterations (95% BCa CIs)
       for both the full non-covalent cohort (N=93) and non-hydantoin subset (N=43).
     - Computes Pearson r and Spearman rho correlations between molecular weight and docking affinity.
     - Generates scatter plot with linear regression fit -> reports/figures/vina_score_vs_mw.png.
"""

import os
import sys
import argparse
import urllib.request
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, Descriptors, rdFingerprintGenerator
import meeko
from meeko import MoleculePreparation, RDKitMolCreate
from vina import Vina

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Grid box parameters centered on ligand 38C in PDB 4P8K Chain A
GRID_CENTER = [17.07, -20.26, 1.49]
GRID_SIZE = [22.0, 22.0, 22.0]
SMILES_38C = "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1"


def download_pdb_if_missing(pdb_id="4P8K", dest_path="docking/receptors/4P8K.pdb"):
    """Downloads PDB file from RCSB if not already present."""
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[INFO] Downloading {pdb_id}.pdb from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[SUCCESS] Downloaded to {dest_path}")
    else:
        print(f"[INFO] Receptor PDB already exists at {dest_path}")


def prepare_receptor_and_reference_ligand():
    """
    Cleans PDB 4P8K:
      - Extracts Chain A protein residues and essential cofactor FAD.
      - Extracts ligand 38C into docking/ligands/crystal_38C.pdb with correct bond orders.
      - Converts cleaned receptor to PDBQT using OpenBabel.
    """
    print("\n" + "=" * 80)
    print("STEP 1: RECEPTOR PREPARATION & REFERENCE LIGAND EXTRACTION")
    print("=" * 80)

    raw_pdb = "docking/receptors/4P8K.pdb"
    download_pdb_if_missing("4P8K", raw_pdb)

    with open(raw_pdb) as f:
        lines = f.readlines()

    clean_receptor_lines = []
    ligand_lines = []

    for line in lines:
        if line.startswith("ATOM") and line[21] == "A":
            clean_receptor_lines.append(line)
        elif line.startswith("HETATM"):
            res = line[17:20].strip()
            ch = line[21]
            if res == "FAD" and ch == "A":
                clean_receptor_lines.append(line)
            elif res in ["38C", "TY3"] and ch == "A":
                ligand_lines.append(line)

    clean_pdb_path = "docking/receptors/4P8K_clean.pdb"
    with open(clean_pdb_path, "w") as f:
        f.writelines(clean_receptor_lines)
    print(f"[INFO] Wrote cleaned receptor PDB (Chain A + FAD) to: {clean_pdb_path}")

    # Process crystal 38C
    raw_lig_path = "docking/ligands/crystal_38C_raw.pdb"
    os.makedirs(os.path.dirname(raw_lig_path), exist_ok=True)
    with open(raw_lig_path, "w") as f:
        f.writelines(ligand_lines)

    # Assign proper bond orders using SMILES template
    template = Chem.MolFromSmiles(SMILES_38C)
    raw_lig_mol = Chem.MolFromPDBFile(raw_lig_path, sanitize=False)
    ref_lig_mol = AllChem.AssignBondOrdersFromTemplate(template, raw_lig_mol)
    Chem.SanitizeMol(ref_lig_mol)

    crystal_lig_path = "docking/ligands/crystal_38C.pdb"
    Chem.MolToPDBFile(ref_lig_mol, crystal_lig_path)
    print(f"[INFO] Wrote sanitized reference ligand 38C (CT325) to: {crystal_lig_path}")

    # Convert cleaned receptor to PDBQT using OpenBabel with polar hydrogens (-p 7.4 -xr)
    receptor_pdbqt = "docking/receptors/4P8K_receptor.pdbqt"
    obabel_cmd = f"obabel -ipdb {clean_pdb_path} -opdbqt -O {receptor_pdbqt} -xr -p 7.4"
    print(f"[INFO] Converting receptor to PDBQT via OpenBabel: {obabel_cmd}")
    res = subprocess.run(obabel_cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[WARNING] OpenBabel exited with code {res.returncode}: {res.stderr}")

    assert os.path.exists(receptor_pdbqt) and os.path.getsize(receptor_pdbqt) > 0, "Failed to create 4P8K_receptor.pdbqt"

    with open(receptor_pdbqt) as f:
        pdbqt_lines = f.readlines()
    fad_count = sum(1 for l in pdbqt_lines if "FAD" in l)
    print(f"[SUCCESS] Generated {receptor_pdbqt} ({len(pdbqt_lines)} lines, {fad_count} FAD lines).")
    return receptor_pdbqt, crystal_lig_path


def prepare_ligand_pdbqt_from_mol(mol, out_pdbqt_path=None):
    """Prepares flexible PDBQT string from an RDKit Mol using Meeko."""
    mol_h = Chem.AddHs(mol)
    # Generate 3D conformer with ETKDGv3
    res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if res != 0:
        # Fallback to standard ETKDG or random coords
        res2 = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if res2 != 0:
            AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol_h)
    if len(mol_setups) == 0:
        raise ValueError("Meeko could not prepare molecule setup")

    pdbqt_res = meeko.PDBQTWriterLegacy.write_string(mol_setups[0])
    pdbqt_str = pdbqt_res[0] if isinstance(pdbqt_res, tuple) else pdbqt_res

    if out_pdbqt_path:
        os.makedirs(os.path.dirname(out_pdbqt_path), exist_ok=True)
        with open(out_pdbqt_path, "w") as f:
            f.write(pdbqt_str)

    return pdbqt_str


def validate_gate2_redocking():
    """
    Performs crystallographic redocking validation of 38C (CT325) into 4P8K:
      - Docks probe 38C into prepared receptor.
      - Calculates symmetry-corrected heavy-atom RMSD against crystal_38C.pdb.
      - Evaluates Gate 2: RMSD < 2.0 Å.
    """
    print("\n" + "=" * 80)
    print("STEP 2: GATE 2 CRYSTALLOGRAPHIC REDOCKING VALIDATION")
    print("=" * 80)

    receptor_pdbqt = "docking/receptors/4P8K_receptor.pdbqt"
    crystal_pdb = "docking/ligands/crystal_38C.pdb"
    if not os.path.exists(receptor_pdbqt) or not os.path.exists(crystal_pdb):
        prepare_receptor_and_reference_ligand()

    # Prepare probe 38C
    probe_mol = Chem.MolFromSmiles(SMILES_38C)
    probe_pdbqt = "docking/ligands/probe_38C.pdbqt"
    prepare_ligand_pdbqt_from_mol(probe_mol, probe_pdbqt)
    print(f"[INFO] Prepared probe PDBQT: {probe_pdbqt}")

    # Run Vina redocking
    v = Vina(sf_name="vina", cpu=4, seed=42)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(probe_pdbqt)
    v.compute_vina_maps(center=GRID_CENTER, box_size=GRID_SIZE)
    v.dock(exhaustiveness=16, n_poses=9)

    out_poses = "docking/outputs/redock_38C_poses.pdbqt"
    os.makedirs(os.path.dirname(out_poses), exist_ok=True)
    v.write_poses(out_poses, n_poses=9, overwrite=True)

    # Compute RMSD against crystal reference
    ref_mol = Chem.MolFromPDBFile(crystal_pdb, removeHs=True)
    pdbqt_mol = meeko.PDBQTMolecule.from_file(out_poses, is_dlg=False, skip_typing=True)
    docked_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

    top_pose = docked_mols[0]
    top_pose_no_h = Chem.RemoveHs(top_pose)
    top_rmsd = float(rdMolAlign.GetBestRMS(top_pose_no_h, ref_mol))

    # Parse top affinity
    top_affinity = None
    with open(out_poses) as f:
        for line in f:
            if "REMARK VINA RESULT:" in line:
                parts = line.split()
                top_affinity = float(parts[3])
                break

    gate2_passed = top_rmsd < 2.0
    gate2_status = "PASS" if gate2_passed else "FAIL"

    report_lines = [
        "================================================================================",
        "GATE 2 REDOCKING VALIDATION REPORT",
        "================================================================================",
        f"Receptor Structure:         PDB 4P8K (Chain A, 2.49 A resolution)",
        f"Essential Cofactor:         FAD (Retained & rigid in active site)",
        f"Reference Ligand:           Ty38c (PDB ID: 38C / CT325)",
        f"Ligand Formula:             C18 H14 F3 N3 O3 (27 heavy atoms)",
        f"Active Site Grid Centroid:  X={GRID_CENTER[0]:.2f}, Y={GRID_CENTER[1]:.2f}, Z={GRID_CENTER[2]:.2f} A",
        f"Search Box Dimensions:      X={GRID_SIZE[0]:.1f}, Y={GRID_SIZE[1]:.1f}, Z={GRID_SIZE[2]:.1f} A",
        f"Top Vina Binding Affinity:  {top_affinity:.3f} kcal/mol",
        f"Top Heavy-Atom RMSD:        {top_rmsd:.3f} A",
        f"Gate 2 Threshold:           RMSD < 2.0 A",
        f"Gate 2 Decision:            {gate2_status}",
        "================================================================================"
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_report = "reports/gate2_redocking.txt"
    os.makedirs(os.path.dirname(out_report), exist_ok=True)
    with open(out_report, "w") as f:
        f.write(report_text + "\n")
    print(f"[SUCCESS] Saved Gate 2 audit to: {out_report}")

    if not gate2_passed:
        print("[ERROR] Gate 2 FAILED: RMSD >= 2.0 A. Halting execution.", file=sys.stderr)
        sys.exit(1)

    return top_rmsd, top_affinity


def run_batch_virtual_screening():
    """
    Performs batch AutoDock Vina screening across all non-covalent labeled molecules (N=93).
    Saves scores to reports/vina_docking_scores.csv.
    """
    print("\n" + "=" * 80)
    print("STEP 3: BATCH VIRTUAL SCREENING (AutoDock Vina on Non-Covalent Cohort)")
    print("=" * 80)

    dataset_path = "data/processed/stratified_cluster_folds.csv"
    df = pd.read_csv(dataset_path)

    # Filter strictly to non-covalent molecules
    non_cov_mask = (~df["nitro_aromatic_warhead"]).values
    df_dock = df[non_cov_mask].copy().reset_index(drop=True)
    print(f"[INFO] Total non-covalent compounds for docking: {len(df_dock)}")
    print(f"  - Active:   {(df_dock['label'] == 'Active').sum()}")
    print(f"  - Inactive: {(df_dock['label'] == 'Inactive').sum()}")
    print(f"  - Hydantoins: {df_dock['is_hydantoin'].sum()}")
    print(f"  - Non-Hydantoins: {(~df_dock['is_hydantoin']).sum()}")

    receptor_pdbqt = "docking/receptors/4P8K_receptor.pdbqt"
    if not os.path.exists(receptor_pdbqt):
        prepare_receptor_and_reference_ligand()

    poses_dir = "docking/outputs/poses"
    os.makedirs(poses_dir, exist_ok=True)

    # Initialize Vina engine once
    v = Vina(sf_name="vina", cpu=4, seed=42)
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=GRID_CENTER, box_size=GRID_SIZE)

    docking_results = []
    total = len(df_dock)

    for i, row in df_dock.iterrows():
        smi = row["canonical_smiles"]
        cid = row["molecule_chembl_ids"]
        lbl = row["label"]
        is_hyd = row["is_hydantoin"]
        mol = Chem.MolFromSmiles(smi)
        mw = float(Descriptors.MolWt(mol))

        try:
            pdbqt_str = prepare_ligand_pdbqt_from_mol(mol)
            v.set_ligand_from_string(pdbqt_str)
            v.dock(exhaustiveness=16, n_poses=9)

            pose_path = os.path.join(poses_dir, f"dock_{i:03d}_{cid}.pdbqt")
            v.write_poses(pose_path, n_poses=9, overwrite=True)

            # Read top affinity from written PDBQT
            top_affinity = None
            with open(pose_path) as f:
                for line in f:
                    if "REMARK VINA RESULT:" in line:
                        top_affinity = float(line.split()[3])
                        break

            docking_results.append({
                "canonical_smiles": smi,
                "molecule_chembl_ids": cid,
                "label": lbl,
                "is_active": 1 if lbl == "Active" else 0,
                "is_hydantoin": is_hyd,
                "molecular_weight": round(mw, 2),
                "vina_affinity": top_affinity,
                "vina_score": -top_affinity if top_affinity is not None else np.nan,  # More positive = stronger binding
                "pose_path": pose_path,
                "status": "SUCCESS"
            })
            print(f"[{i+1:>2}/{total}] {cid:<15} | MW: {mw:>6.1f} | Affinity: {top_affinity:>6.2f} kcal/mol | {lbl}")

        except Exception as e:
            print(f"[{i+1:>2}/{total}] {cid:<15} | ERROR: {e}", file=sys.stderr)
            docking_results.append({
                "canonical_smiles": smi,
                "molecule_chembl_ids": cid,
                "label": lbl,
                "is_active": 1 if lbl == "Active" else 0,
                "is_hydantoin": is_hyd,
                "molecular_weight": round(mw, 2),
                "vina_affinity": np.nan,
                "vina_score": np.nan,
                "pose_path": None,
                "status": f"FAILED: {e}"
            })

    out_csv = "reports/vina_docking_scores.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res = pd.DataFrame(docking_results)
    df_res.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Completed batch docking. Saved results to: {out_csv}")
    return df_res


def compute_ef(y_true, scores, fraction=0.10):
    """Computes Enrichment Factor at given fraction (default 10%)."""
    n = len(y_true)
    n_act = int(np.sum(y_true == 1))
    if n_act == 0 or n == 0:
        return 0.0
    k = max(1, int(round(fraction * n)))
    top_k_idx = np.argsort(-scores)[:k]
    hits = int(np.sum(y_true[top_k_idx] == 1))
    ef = (hits / k) / (n_act / n)
    return float(ef)


def bootstrap_metric(y_true, scores, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Computes 1,000 bootstrap iterations reporting mean, std, and 95% BCa/percentile CIs."""
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


def evaluate_docking_enrichment_and_size_bias():
    """
    Evaluates virtual screening enrichment (PR-AUC, ROC-AUC, EF_10%) and molecular weight bias.
    """
    print("\n" + "=" * 80)
    print("STEP 4: ENRICHMENT METRICS & SIZE-BIAS AUDITING")
    print("=" * 80)

    scores_path = "reports/vina_docking_scores.csv"
    if not os.path.exists(scores_path):
        run_batch_virtual_screening()

    df = pd.read_csv(scores_path)
    df = df[df["status"] == "SUCCESS"].copy().reset_index(drop=True)
    print(f"[INFO] Ingested {len(df)} successfully docked molecules.")

    # Higher score = stronger binding
    # Vina returns affinity in kcal/mol (negative); ranking score is -vina_affinity
    scores = df["vina_score"].values
    y = df["is_active"].values

    cohorts = [
        ("Full Non-Covalent Docking Set", df),
        ("Non-Covalent, Non-Hydantoin Subset", df[~df["is_hydantoin"]].copy().reset_index(drop=True))
    ]

    metric_records = []

    for cohort_name, c_df in cohorts:
        c_scores = c_df["vina_score"].values
        c_y = c_df["is_active"].values
        n = len(c_df)
        n_act = int(np.sum(c_y == 1))
        n_inact = int(np.sum(c_y == 0))

        roc_orig, roc_mean, roc_std, roc_ci_lo, roc_ci_hi = bootstrap_metric(
            c_y, c_scores, lambda y_t, s: roc_auc_score(y_t, s)
        )
        pr_orig, pr_mean, pr_std, pr_ci_lo, pr_ci_hi = bootstrap_metric(
            c_y, c_scores, lambda y_t, s: average_precision_score(y_t, s)
        )
        ef_orig, ef_mean, ef_std, ef_ci_lo, ef_ci_hi = bootstrap_metric(
            c_y, c_scores, lambda y_t, s: compute_ef(y_t, s, fraction=0.10)
        )

        metric_records.append({
            "cohort": cohort_name,
            "n_samples": n,
            "n_actives": n_act,
            "n_inactives": n_inact,
            "roc_auc_orig": round(roc_orig, 4),
            "roc_auc_ci_lower": round(roc_ci_lo, 4),
            "roc_auc_ci_upper": round(roc_ci_hi, 4),
            "pr_auc_orig": round(pr_orig, 4),
            "pr_auc_ci_lower": round(pr_ci_lo, 4),
            "pr_auc_ci_upper": round(pr_ci_hi, 4),
            "ef_10_orig": round(ef_orig, 4),
            "ef_10_ci_lower": round(ef_ci_lo, 4),
            "ef_10_ci_upper": round(ef_ci_hi, 4)
        })

        print(f"\n--- {cohort_name} (N={n}: {n_act} Act, {n_inact} Inact) ---")
        print(f"  ROC-AUC:  {roc_orig:.3f} [{roc_ci_lo:.3f}, {roc_ci_hi:.3f}]")
        print(f"  PR-AUC:   {pr_orig:.3f} [{pr_ci_lo:.3f}, {pr_ci_hi:.3f}] (Prior: {n_act/n:.3f})")
        print(f"  EF_10%:   {ef_orig:.2f}x [{ef_ci_lo:.2f}, {ef_ci_hi:.2f}]")

    out_metrics_csv = "reports/docking_enrichment_metrics.csv"
    os.makedirs(os.path.dirname(out_metrics_csv), exist_ok=True)
    df_metrics = pd.DataFrame(metric_records)
    df_metrics.to_csv(out_metrics_csv, index=False)
    print(f"\n[SUCCESS] Saved docking enrichment metrics to: {out_metrics_csv}")

    # =========================================================================
    # SIZE-BIAS AUDITING (Molecular Weight vs. Vina Affinity)
    # =========================================================================
    mw = df["molecular_weight"].values
    aff = df["vina_affinity"].values

    pearson_r, pearson_p = stats.pearsonr(mw, aff)
    spearman_rho, spearman_p = stats.spearmanr(mw, aff)

    print("\n" + "=" * 80)
    print("MOLECULAR WEIGHT SIZE-BIAS AUDIT")
    print("=" * 80)
    print(f"Pearson Correlation (MW vs. Vina Affinity):  r = {pearson_r:+.3f} (p = {pearson_p:.2e})")
    print(f"Spearman Correlation (MW vs. Vina Affinity): rho = {spearman_rho:+.3f} (p = {spearman_p:.2e})")
    print("Scientific Interpretation: Significant negative correlation indicates standard Vina scoring function")
    print("exhibits molecular weight bias (larger ligands receive artificially more favorable kcal/mol scores).")

    # Generate scatter plot
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot actives and inactives separately
    sns.scatterplot(
        data=df,
        x="molecular_weight",
        y="vina_affinity",
        hue="label",
        palette={"Active": "#2ca02c", "Inactive": "#d62728"},
        alpha=0.85,
        s=70,
        edgecolor="black"
    )

    # Regression fit line
    sns.regplot(
        data=df,
        x="molecular_weight",
        y="vina_affinity",
        scatter=False,
        color="#1f77b4",
        line_kws={"linewidth": 2, "linestyle": "--", "label": f"Fit: r = {pearson_r:.2f} (p = {pearson_p:.1e})"}
    )

    plt.xlabel("Molecular Weight (g/mol)", fontsize=12, fontweight="bold")
    plt.ylabel("AutoDock Vina Affinity (kcal/mol)", fontsize=12, fontweight="bold")
    plt.title(f"DprE1 Vina Affinity vs. Molecular Weight (N={len(df)})\nPearson r = {pearson_r:+.2f}, Spearman ρ = {spearman_rho:+.2f}", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out_fig = "reports/figures/vina_score_vs_mw.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved size-bias audit figure to: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 AutoDock Vina Docking Pipeline for DprE1")
    parser.add_argument(
        "--step",
        choices=["prep", "redock", "batch", "evaluate", "all"],
        default="all",
        help="Pipeline step to execute"
    )
    args = parser.parse_args()

    if args.step in ["prep", "all"]:
        prepare_receptor_and_reference_ligand()

    if args.step in ["redock", "all"]:
        validate_gate2_redocking()

    if args.step in ["batch", "all"]:
        run_batch_virtual_screening()

    if args.step in ["evaluate", "all"]:
        evaluate_docking_enrichment_and_size_bias()


if __name__ == "__main__":
    main()
