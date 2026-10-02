#!/usr/bin/env python3
"""
src/get_pocket_center.py

Downloads reference PDB 4P8K if missing, extracts atomic coordinates for
co-crystallized ligand 38C (CT325) in Chain A (the reference active site),
computes exact geometric centroid and coordinate ranges, and writes
the verified AutoDock Vina configuration file docking/configs/vina_4P8K.txt.
"""

import os
import sys
import urllib.request
import numpy as np


def ensure_receptor_pdb(pdb_id="4P8K", dest_dir="docking/receptors"):
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    if not os.path.exists(dest_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[INFO] Fetching {pdb_id}.pdb from RCSB ({url})...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[INFO] Saved receptor to {dest_path}")
    return dest_path


def parse_ligand_coords(pdb_path, res_name="38C", chain="A"):
    coords = []
    atom_records = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")):
                line_res = line[17:20].strip()
                line_chain = line[21]
                if line_res == res_name:
                    if chain is None or line_chain == chain:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atom_name = line[12:16].strip()
                        element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                        coords.append([x, y, z])
                        atom_records.append({
                            "atom_name": atom_name,
                            "element": element,
                            "chain": line_chain,
                            "coord": np.array([x, y, z])
                        })
    return np.array(coords), atom_records


def main():
    print("=" * 75)
    print("PRECISE POCKET CENTROID CALCULATION: PDB 4P8K / Ligand 38C (CT325)")
    print("=" * 75)

    pdb_path = ensure_receptor_pdb("4P8K", "docking/receptors")
    
    # Extract coordinates for Chain A (reference monomer)
    coords_a, atoms_a = parse_ligand_coords(pdb_path, res_name="38C", chain="A")
    coords_b, atoms_b = parse_ligand_coords(pdb_path, res_name="38C", chain="B")
    
    if len(coords_a) == 0:
        print("[ERROR] No atoms found for residue 38C in Chain A!", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(coords_a)} atoms for ligand 38C (CT325) in Chain A.")
    if len(coords_b) > 0:
        print(f"Note: Homodimer asymmetric unit also contains {len(coords_b)} atoms in Chain B.")

    # Geometric centroid calculation: C = (1/N) * sum(r_i)
    centroid_a = np.mean(coords_a, axis=0)
    cx, cy, cz = centroid_a[0], centroid_a[1], centroid_a[2]

    # Coordinate ranges and spans
    min_x, max_x = np.min(coords_a[:, 0]), np.max(coords_a[:, 0])
    min_y, max_y = np.min(coords_a[:, 1]), np.max(coords_a[:, 1])
    min_z, max_z = np.min(coords_a[:, 2]), np.max(coords_a[:, 2])

    print("\n--- Chain A Coordinate Bounds & Geometric Centroid ---")
    print(f"  X span: [{min_x:.3f}, {max_x:.3f}] (delta: {max_x - min_x:.3f} Å)")
    print(f"  Y span: [{min_y:.3f}, {max_y:.3f}] (delta: {max_y - min_y:.3f} Å)")
    print(f"  Z span: [{min_z:.3f}, {max_z:.3f}] (delta: {max_z - min_z:.3f} Å)")
    print(f"\n  Exact Centroid (x, y, z): ({cx:.4f}, {cy:.4f}, {cz:.4f}) Å")
    print(f"  Rounded Centroid (2 d.p.): ({cx:.2f}, {cy:.2f}, {cz:.2f}) Å")

    # Generate Vina configuration file
    config_dir = "docking/configs"
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "vina_4P8K.txt")

    config_content = f"""# AutoDock Vina Configuration: DprE1 Non-Covalent Reference (PDB 4P8K)
# Target: Mycobacterium tuberculosis DprE1 (Rv3790 / UniProt P9WJF1)
# Pocket: Centered on co-crystallized ligand 38C (CT325) in Chain A adjacent to FAD & Cys387

center_x = {cx:.2f}
center_y = {cy:.2f}
center_z = {cz:.2f}

size_x = 22.0
size_y = 22.0
size_z = 22.0

exhaustiveness = 24
num_modes = 9
energy_range = 3
"""

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"\n[SUCCESS] Written verified Vina config to: {config_path}")
    print("-" * 75)
    print(config_content.strip())
    print("=" * 75)


if __name__ == "__main__":
    main()
