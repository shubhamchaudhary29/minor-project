#!/usr/bin/env python3
"""
src/features.py

Molecular featurization and chemotype annotation suite for DprE1:
  1. Chemotype identification:
     - Hydantoins (is_hydantoin): SMARTS 'O=C1NC(=O)NC1'
     - Covalent Nitro-Aromatic Warheads (nitro_aromatic_warhead): SMARTS 'c[N+](=O)[O-]' and 'c[N]=O'
  2. Structural & Topological Representations:
     - ECFP4 (Morgan radius 2): 2048-bit binary vectors & 2048-dim count vectors
     - MACCS Keys: 166-bit structural keys
     - RDKit 2D Physicochemical Descriptors: Filtered for zero-variance and collinearity (r > 0.95)
  3. Similarity Metrics:
     - 1-NN Tanimoto similarity to training fold actives
     - Maximum Tanimoto similarity to training fold (applicability domain)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Canonical SMARTS patterns
HYDANTOIN_SMARTS = "O=C1NC(=O)NC1"
NITRO_AROMATIC_SMARTS = "c[N+](=O)[O-]"
NITROSO_AROMATIC_SMARTS = "c[N]=O"
BTZ_CORE_SMARTS = "c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2"

_PAT_HYDANTOIN = Chem.MolFromSmarts(HYDANTOIN_SMARTS)
_PAT_NITRO = Chem.MolFromSmarts(NITRO_AROMATIC_SMARTS)
_PAT_NITROSO = Chem.MolFromSmarts(NITROSO_AROMATIC_SMARTS)
_PAT_BTZ = Chem.MolFromSmarts(BTZ_CORE_SMARTS)


def is_hydantoin(smiles: str) -> bool:
    """Returns True if the molecule contains the hydantoin imidazolidinedione core."""
    if not smiles or pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False
    return mol.HasSubstructMatch(_PAT_HYDANTOIN)


def is_nitro_aromatic_warhead(smiles: str) -> bool:
    """Returns True if the molecule possesses an activated aromatic nitro or nitroso warhead."""
    if not smiles or pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return False
    return (
        mol.HasSubstructMatch(_PAT_NITRO)
        or mol.HasSubstructMatch(_PAT_NITROSO)
        or mol.HasSubstructMatch(_PAT_BTZ)
    )


def compute_morgan_fingerprints(smiles_list, radius: int = 2, n_bits: int = 2048, as_counts: bool = False) -> np.ndarray:
    """
    Computes Morgan circular fingerprints (ECFP4 equivalent) as a 2D numpy array.
    If as_counts is True, returns integer feature count vectors.
    Otherwise, returns binary bit vectors.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            raise ValueError(f"Could not parse SMILES: {s}")
        if as_counts:
            fp = gen.GetCountFingerprint(m)
            arr = np.zeros(n_bits, dtype=np.float32)
            for bit_id, val in fp.GetNonzeroElements().items():
                arr[bit_id] = val
            fps.append(arr)
        else:
            fp = gen.GetFingerprint(m)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.vstack(fps)


def compute_maccs_keys(smiles_list) -> np.ndarray:
    """
    Computes standard 166-bit MACCS structural keys as a 2D numpy array.
    (Note: RDKit BitVect length is 167 with bit 0 unused; sliced [:, 1:] to return exact 166 bits).
    """
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            raise ValueError(f"Could not parse SMILES: {s}")
        maccs_bv = MACCSkeys.GenMACCSKeys(m)
        arr = np.zeros(len(maccs_bv), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs_bv, arr)
        fps.append(arr[1:])  # Standard 166 bits
    return np.vstack(fps)


def compute_rdkit_2d_descriptors(smiles_list, filter_collinear: bool = True, corr_threshold: float = 0.95):
    """
    Computes 2D physicochemical molecular descriptors using RDKit.
    Filters out zero-variance features and collinear features (Pearson r > corr_threshold).
    Returns (descriptor_matrix, descriptor_names).
    """
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            raise ValueError(f"Could not parse SMILES: {s}")
        mols.append(m)

    raw_vals = [calc.CalcDescriptors(m) for m in mols]
    df_desc = pd.DataFrame(raw_vals, columns=desc_names)

    # Handle infs and NaNs with median imputation
    df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
    for col in df_desc.columns:
        if df_desc[col].isna().any():
            med = df_desc[col].median()
            df_desc[col] = df_desc[col].fillna(med if not np.isnan(med) else 0.0)

    # 1. Filter zero or near-zero variance
    var = df_desc.var()
    non_zero = var[var > 1e-6].index.tolist()
    df_filtered = df_desc[non_zero].copy()

    # 2. Filter collinear features
    if filter_collinear:
        corr_matrix = df_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        df_filtered = df_filtered.drop(columns=to_drop)

    final_names = df_filtered.columns.tolist()
    final_arr = df_filtered.values.astype(np.float32)
    return final_arr, final_names


def compute_rdkit_bit_fingerprints(smiles_list, radius: int = 2, n_bits: int = 2048):
    """Returns a list of native RDKit ExplicitBitVect objects for fast Tanimoto similarity."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            raise ValueError(f"Could not parse SMILES: {s}")
        fps.append(gen.GetFingerprint(m))
    return fps


def compute_max_tanimoto_to_actives(test_fps, train_fps, train_labels):
    """
    Computes 1-Nearest Neighbor Tanimoto similarity from each test compound
    to all active compounds in the training fold.
    """
    active_indices = [i for i, lbl in enumerate(train_labels) if lbl in [1, "Active", "active"]]
    if len(active_indices) == 0:
        return np.zeros(len(test_fps), dtype=float)

    active_fps = [train_fps[i] for i in active_indices]
    sims = []
    for tfp in test_fps:
        bulk_sims = DataStructs.BulkTanimotoSimilarity(tfp, active_fps)
        sims.append(max(bulk_sims) if len(bulk_sims) > 0 else 0.0)
    return np.array(sims, dtype=float)


def compute_max_tanimoto_to_train(test_fps, train_fps):
    """
    Computes maximum Tanimoto similarity from each test compound to ALL compounds in the training fold
    (Applicability domain analysis).
    """
    sims = []
    for tfp in test_fps:
        bulk_sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        sims.append(max(bulk_sims) if len(bulk_sims) > 0 else 0.0)
    return np.array(sims, dtype=float)


def extract_and_save_all_features(smiles_list, out_dir="data/interim/features"):
    """
    Precomputes and stores all feature representations into out_dir as NumPy arrays and metadata.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Precomputing feature matrices for {len(smiles_list)} compounds...")

    # 1. ECFP4 Bits & Counts
    ecfp4_bits = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048, as_counts=False)
    ecfp4_counts = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048, as_counts=True)
    np.save(os.path.join(out_dir, "ecfp4_bits.npy"), ecfp4_bits)
    np.save(os.path.join(out_dir, "ecfp4_counts.npy"), ecfp4_counts)
    print(f"  - ECFP4 Bits saved:   {ecfp4_bits.shape} -> {os.path.join(out_dir, 'ecfp4_bits.npy')}")
    print(f"  - ECFP4 Counts saved: {ecfp4_counts.shape} -> {os.path.join(out_dir, 'ecfp4_counts.npy')}")

    # 2. MACCS Keys
    maccs_keys = compute_maccs_keys(smiles_list)
    np.save(os.path.join(out_dir, "maccs.npy"), maccs_keys)
    print(f"  - MACCS Keys saved:   {maccs_keys.shape} -> {os.path.join(out_dir, 'maccs.npy')}")

    # 3. RDKit 2D Physicochemical Descriptors
    rdkit_2d, desc_names = compute_rdkit_2d_descriptors(smiles_list, filter_collinear=True, corr_threshold=0.95)
    np.save(os.path.join(out_dir, "rdkit_2d.npy"), rdkit_2d)
    with open(os.path.join(out_dir, "rdkit_2d_names.json"), "w") as f:
        json.dump(desc_names, f, indent=2)
    print(f"  - RDKit 2D saved:     {rdkit_2d.shape} ({len(desc_names)} descriptors) -> {os.path.join(out_dir, 'rdkit_2d.npy')}")

    return {
        "ecfp4_bits": ecfp4_bits,
        "ecfp4_counts": ecfp4_counts,
        "maccs": maccs_keys,
        "rdkit_2d": rdkit_2d,
        "rdkit_2d_names": desc_names
    }


def main():
    print("=" * 80)
    print("FEATURE ENGINEERING SUITE: DprE1 Benchmark (Phase 3)")
    print("=" * 80)

    dataset_path = "data/processed/stratified_cluster_folds.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "data/processed/gate1_compounds.csv"
    df = pd.read_csv(dataset_path)
    labeled = df[df["label"].isin(["Active", "Inactive"])].copy().reset_index(drop=True)
    print(f"[INFO] Loaded {len(labeled)} labeled benchmark compounds from: {dataset_path}")

    features_dict = extract_and_save_all_features(labeled["canonical_smiles"].tolist(), out_dir="data/interim/features")
    print("\n[SUCCESS] All feature matrices generated and persisted successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
