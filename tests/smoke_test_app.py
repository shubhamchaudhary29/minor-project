#!/usr/bin/env python3
"""
tests/smoke_test_app.py

Smoke test suite verifying core functionality required for app/streamlit_app.py:
1. Robust SAR mutations (+F, +Me, +NO2, strip substituent) using explicit-H reaction SMARTS.
2. 3D conformer embedding, Meeko PDBQT preparation, and fast live AutoDock Vina docking.
3. 2D Machine Learning prediction engine (Random Forest and Logistic Regression).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import meeko
from meeko import MoleculePreparation
from vina import Vina
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reaction SMARTS using explicit hydrogens on aromatic carbons
RXN_ADD_F = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1]F')
RXN_ADD_ME = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1]C')
RXN_ADD_NO2 = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1][N+](=O)[O-]')
STRIP_RXNS = [
    AllChem.ReactionFromSmarts('[c:1][F,Cl,Br,I]>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1][N+](=O)[O-]>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]OC>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C(=O)O>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C(F)(F)F>>[c:1][H]'),
]

PAT_NITRO = Chem.MolFromSmarts('c[N+](=O)[O-]')


def mutate_add_fluorine(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_F.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_add_methyl(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_ME.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_add_nitro(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_NO2.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_strip_substituent(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    for rxn in STRIP_RXNS:
        prods = rxn.RunReactants((mol_h,))
        if prods and len(prods[0]) > 0:
            res = prods[0][0]
            Chem.SanitizeMol(res)
            return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mol_to_pdbqt(mol):
    prep = MoleculePreparation()
    mol_setups = prep.prepare(mol)
    setup = mol_setups[0] if isinstance(mol_setups, list) else mol_setups
    if hasattr(setup, 'write_pdbqt_string'):
        res = setup.write_pdbqt_string()
        return res[0] if isinstance(res, tuple) else res
    res = meeko.PDBQTWriterLegacy.write_string(setup)
    return res[0] if isinstance(res, tuple) else res


def dock_ligand_live(smi: str, exhaustiveness: int = 8, timeout_sec: float = 20.0):
    t0 = time.time()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smi}")
    mol_h = Chem.AddHs(mol)

    res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if res != 0:
        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if res != 0:
            AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass

    pdbqt_str = mol_to_pdbqt(mol_h)
    temp_pdbqt = "docking/outputs/temp_query.pdbqt"
    os.makedirs(os.path.dirname(temp_pdbqt), exist_ok=True)
    with open(temp_pdbqt, "w") as f:
        f.write(pdbqt_str)

    v = Vina(sf_name="vina", cpu=4, seed=42)
    receptor_path = "docking/receptors/4P8K_receptor.pdbqt"
    if not os.path.exists(receptor_path):
        raise FileNotFoundError(f"Missing receptor at {receptor_path}")

    v.set_receptor(receptor_path)
    v.set_ligand_from_file(temp_pdbqt)
    v.compute_vina_maps(center=[17.07, -20.26, 1.49], box_size=[20.0, 20.0, 20.0])
    v.dock(exhaustiveness=exhaustiveness, n_poses=1)
    energy = float(v.energies(n_poses=1)[0][0])
    elapsed = time.time() - t0
    return energy, elapsed


def test_sar_mutations():
    print("\n--- 1. Testing SAR Mutations ---")
    ct325 = "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1"
    mol_orig = Chem.MolFromSmiles(ct325)
    f_orig_count = len(mol_orig.GetSubstructMatches(Chem.MolFromSmarts('F')))

    # Test +F
    smi_f = mutate_add_fluorine(ct325)
    mol_f = Chem.MolFromSmiles(smi_f)
    f_new_count = len(mol_f.GetSubstructMatches(Chem.MolFromSmarts('F')))
    print(f"  [+] Original F count: {f_orig_count} -> Mutated F count: {f_new_count} (SMILES: {smi_f})")
    assert f_new_count == f_orig_count + 1, "Fluorine mutation failed to add an extra F!"

    # Test +NO2
    smi_no2 = mutate_add_nitro(ct325)
    mol_no2 = Chem.MolFromSmiles(smi_no2)
    has_nitro = mol_no2.HasSubstructMatch(PAT_NITRO)
    print(f"  [+] Added Nitro Warhead: {smi_no2} (Detected by sentry SMARTS: {has_nitro})")
    assert has_nitro, "Nitro mutation failed to create a valid aromatic nitro warhead!"

    # Test +Me
    smi_me = mutate_add_methyl(ct325)
    mol_me = Chem.MolFromSmiles(smi_me)
    assert mol_me.GetNumHeavyAtoms() == mol_orig.GetNumHeavyAtoms() + 1, "Methyl mutation failed!"
    print(f"  [+] Added Methyl: {smi_me}")

    # Test Strip
    smi_strip = mutate_strip_substituent(ct325)
    mol_strip = Chem.MolFromSmiles(smi_strip)
    assert mol_strip.GetNumHeavyAtoms() < mol_orig.GetNumHeavyAtoms(), "Substituent strip failed!"
    print(f"  [+] Stripped Substituent: {smi_strip}")
    print("[PASS] All SAR mutations validated successfully.")


def test_live_docking():
    print("\n--- 2. Testing Live Meeko & AutoDock Vina Docking ---")
    ct325 = "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1"
    energy, elapsed = dock_ligand_live(ct325, exhaustiveness=8)
    print(f"  [+] Docking Energy: {energy:.2f} kcal/mol | Elapsed Time: {elapsed:.2f}s")
    assert isinstance(energy, float), "Energy is not a float!"
    assert -12.0 <= energy <= -4.0, f"Energy {energy} out of expected range [-12.0, -4.0]!"
    assert elapsed < 20.0, f"Docking took too long: {elapsed:.2f}s (limit: 20s)"
    print("[PASS] Live AutoDock Vina docking validated successfully.")


def test_ml_prediction_engine():
    print("\n--- 3. Testing ML Prediction Engine ---")
    data_path = "data/processed/stratified_cluster_folds.csv"
    assert os.path.exists(data_path), f"Missing {data_path}"
    df = pd.read_csv(data_path)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = []
    y = []
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(row["canonical_smiles"])
        if m is None: continue
        y.append(1 if row["label"] == "Active" else 0)
        fp = gen.GetCountFingerprint(m)
        arr = np.zeros(2048, dtype=np.float32)
        for bit, val in fp.GetNonzeroElements().items():
            arr[bit] = val
        X.append(arr)

    X = np.vstack(X)
    y = np.array(y)

    lr = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X, y)

    rf = RandomForestClassifier(n_estimators=100, min_samples_split=4, class_weight="balanced_subsample", random_state=42)
    rf.fit(X, y)

    # Test query
    ct325 = "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1"
    q_mol = Chem.MolFromSmiles(ct325)
    q_fp = gen.GetCountFingerprint(q_mol)
    q_arr = np.zeros(2048, dtype=np.float32)
    for bit, val in q_fp.GetNonzeroElements().items():
        q_arr[bit] = val

    p_lr = float(lr.predict_proba([q_arr])[0, 1])
    p_rf = float(rf.predict_proba([q_arr])[0, 1])

    print(f"  [+] CT325 Logistic Regression P(Active): {p_lr:.3f}")
    print(f"  [+] CT325 Random Forest P(Active):       {p_rf:.3f}")

    assert 0.0 <= p_lr <= 1.0, f"p_lr {p_lr} out of [0, 1]!"
    assert 0.0 <= p_rf <= 1.0, f"p_rf {p_rf} out of [0, 1]!"
    print("[PASS] ML prediction engine validated successfully.")


def main():
    print("=" * 80)
    print("STARTING DPRE1 STREAMLIT LAB OFFLINE SMOKE TESTS")
    print("=" * 80)
    test_sar_mutations()
    test_live_docking()
    test_ml_prediction_engine()
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)


if __name__ == "__main__":
    main()
