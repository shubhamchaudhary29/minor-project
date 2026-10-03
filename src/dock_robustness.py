#!/usr/bin/env python3
"""
src/dock_robustness.py

Multi-Seed Crystallographic Redocking Robustness Verification:
Evaluates Gate 2 redocking stability across three independent random seeds (42, 101, 2024)
using AutoDock Vina (exhaustiveness=16) on PDB 4P8K reference ligand 38C (Ty38c).

Outputs:
  reports/docking_robustness_seeds.txt
"""

import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import meeko
from meeko import RDKitMolCreate
from vina import Vina

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

GRID_CENTER = [17.07, -20.26, 1.49]
GRID_SIZE = [22.0, 22.0, 22.0]


def run_seed_docking(receptor_pdbqt, probe_pdbqt, crystal_pdb, seed, exhaustiveness=16):
    """Executes AutoDock Vina redocking for a given random seed and computes heavy-atom RMSD."""
    v = Vina(sf_name="vina", cpu=4, seed=seed)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(probe_pdbqt)
    v.compute_vina_maps(center=GRID_CENTER, box_size=GRID_SIZE)
    v.dock(exhaustiveness=exhaustiveness, n_poses=9)

    out_poses = f"docking/outputs/redock_38C_seed_{seed}.pdbqt"
    os.makedirs(os.path.dirname(out_poses), exist_ok=True)
    v.write_poses(out_poses, n_poses=9, overwrite=True)

    # Compute RMSD against crystal reference
    ref_mol = Chem.MolFromPDBFile(crystal_pdb, removeHs=True)
    pdbqt_mol = meeko.PDBQTMolecule.from_file(out_poses, is_dlg=False, skip_typing=True)
    docked_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)

    top_pose_no_h = Chem.RemoveHs(docked_mols[0])
    top_rmsd = float(rdMolAlign.GetBestRMS(top_pose_no_h, ref_mol))
    top_affinity = float(v.energies(n_poses=1)[0][0])

    return top_affinity, top_rmsd


def main():
    print("=" * 80)
    print("MULTI-SEED DOCKING ROBUSTNESS EVALUATION (GATE 2 POSE REPRODUCIBILITY)")
    print("=" * 80)

    receptor_pdbqt = "docking/receptors/4P8K_receptor.pdbqt"
    probe_pdbqt = "docking/ligands/probe_38C.pdbqt"
    crystal_pdb = "docking/ligands/crystal_38C.pdb"

    assert os.path.exists(receptor_pdbqt), f"Missing {receptor_pdbqt}"
    assert os.path.exists(probe_pdbqt), f"Missing {probe_pdbqt}"
    assert os.path.exists(crystal_pdb), f"Missing {crystal_pdb}"

    seeds = [42, 101, 2024]
    results = []

    print(f"{'Random Seed':<12} | {'Vina Affinity (kcal/mol)':<26} | {'Heavy-Atom RMSD (Å)':<22} | {'Gate 2 Decision (<2.0 Å)'}")
    print("-" * 80)

    for seed in seeds:
        affinity, rmsd = run_seed_docking(receptor_pdbqt, probe_pdbqt, crystal_pdb, seed=seed, exhaustiveness=16)
        decision = "PASS" if rmsd < 2.0 else "FAIL"
        print(f"{seed:<12} | {affinity:<26.3f} | {rmsd:<22.3f} | {decision}")
        results.append({
            "seed": seed,
            "affinity": affinity,
            "rmsd": rmsd,
            "decision": decision
        })

    # Summary report
    rmsds = [r["rmsd"] for r in results]
    affinities = [r["affinity"] for r in results]
    all_pass = all(r["decision"] == "PASS" for r in results)

    mean_rmsd = np.mean(rmsds)
    std_rmsd = np.std(rmsds)
    mean_aff = np.mean(affinities)

    report_lines = [
        "=" * 80,
        "DprE1 GATE 2 MULTI-SEED DOCKING ROBUSTNESS REPORT",
        "=" * 80,
        "Receptor:               PDB 4P8K (Chain A + rigid FAD, 2.49 Å)",
        "Reference Ligand:       Ty38c (PDB ID: 38C / CT325, 27 heavy atoms)",
        "Grid Center:            (17.07, -20.26, 1.49) Å",
        "Box Size:               (22.0, 22.0, 22.0) Å",
        "Search Exhaustiveness:  16 (per PROTOCOL.md DEV-09)",
        f"Seeds Tested:           {seeds}",
        "-" * 80,
        f"{'Random Seed':<12} | {'Affinity (kcal/mol)':<22} | {'RMSD (Å)':<16} | {'Gate 2 Status'}",
        "-" * 80,
    ]

    for r in results:
        report_lines.append(f"{r['seed']:<12} | {r['affinity']:<22.3f} | {r['rmsd']:<16.3f} | {r['decision']}")

    report_lines.extend([
        "-" * 80,
        f"Mean RMSD:              {mean_rmsd:.3f} ± {std_rmsd:.3f} Å",
        f"Mean Binding Affinity:  {mean_aff:.3f} kcal/mol",
        f"Gate 2 Threshold:       RMSD < 2.000 Å across all seeds",
        f"Robustness Outcome:     {'PASS - FULLY ROBUST' if all_pass else 'FAIL'}",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines) + "\n"
    out_file = "reports/docking_robustness_seeds.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write(report_text)

    print("-" * 80)
    print(f"[SUMMARY] Mean RMSD = {mean_rmsd:.3f} ± {std_rmsd:.3f} Å | Gate 2 Decision: {'PASS' if all_pass else 'FAIL'}")
    print(f"[SUCCESS] Multi-seed report saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
