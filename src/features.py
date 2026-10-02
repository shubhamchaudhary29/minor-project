#!/usr/bin/env python3
"""
src/features.py

Molecular featurization and chemotype annotation suite for DprE1:
  1. Chemotype identification:
     - Hydantoins (is_hydantoin): SMARTS 'O=C1NC(=O)NC1'
     - Covalent Nitro-Aromatic Warheads (nitro_aromatic_warhead): SMARTS 'c[N+](=O)[O-]' and 'c[N]=O'
  2. Fingerprint featurization:
     - Morgan circular fingerprints (radius 2, 2048-bit bit vectors or count vectors)
     - Pairwise and 1-NN Tanimoto similarity computations.
"""

import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs


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
    if as_counts:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(str(s).strip())
            if m is None:
                raise ValueError(f"Could not parse SMILES: {s}")
            fp = gen.GetCountFingerprint(m)
            # Convert SparseIntVect to dense numpy array
            arr = np.zeros(n_bits, dtype=np.float32)
            for bit_id, val in fp.GetNonzeroElements().items():
                arr[bit_id] = val
            fps.append(arr)
        return np.vstack(fps)
    else:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(str(s).strip())
            if m is None:
                raise ValueError(f"Could not parse SMILES: {s}")
            fp = gen.GetFingerprint(m)
            arr = np.zeros(n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        return np.vstack(fps)


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
        # Fallback if no actives in training fold
        return np.zeros(len(test_fps), dtype=float)

    active_fps = [train_fps[i] for i in active_indices]
    sims = []
    for tfp in test_fps:
        bulk_sims = DataStructs.BulkTanimotoSimilarity(tfp, active_fps)
        sims.append(max(bulk_sims) if len(bulk_sims) > 0 else 0.0)
    return np.array(sims, dtype=float)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING FEATURES & SUBSTRUCTURE CLASSIFICATION")
    print("=" * 80)

    test_smiles = [
        ("CC(=O)CN1C(=O)NC(C)(c2ccc(S(N)(=O)=O)cc2)C1=O", True, False), # Hydantoin
        ("O=c1nc(N2CCC3(CC2)OCCO3)sc2ccccc12", False, False),           # Non-hydantoin, non-covalent
        ("O=C1N(Cc2ccccc2)CS(=O)(=O)c2cc([N+](=O)[O-])ccc21", False, True) # Nitro-aromatic
    ]

    for smi, exp_hyd, exp_cov in test_smiles:
        hyd = is_hydantoin(smi)
        cov = is_nitro_aromatic_warhead(smi)
        print(f"SMILES: {smi[:40]}... -> Hydantoin={hyd} (exp {exp_hyd}) | Covalent={cov} (exp {exp_cov})")
        assert hyd == exp_hyd, f"Hydantoin mismatch for {smi}"
        assert cov == exp_cov, f"Covalent mismatch for {smi}"

    print("[SUCCESS] All feature assertions passed successfully!")
    print("=" * 80)
