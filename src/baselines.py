#!/usr/bin/env python3
"""
src/baselines.py

Baseline Suite for DprE1 Benchmark (Phase 3):
Implements non-learning and heuristic baselines:
  1. Baseline 0 (Random Floor): Dummy prior predicting training positive class frequency (P ≈ 0.725).
  2. Baseline 1 (Chemotype Confounder): HydantoinDetectorClassifier predicting P=0.843 for hydantoins,
     and P=0.608 for non-hydantoins, quantifying the single-series reporting bias.
  3. Baseline 2 (1-NN Tanimoto Similarity): Nearest-neighbor structural retrieval predicting the maximum
     Tanimoto similarity to known active molecules in the training split.
"""

import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import is_hydantoin, compute_max_tanimoto_to_actives


class RandomFloorClassifier:
    """Baseline 0: Dummy classifier predicting empirical majority class prior from training set."""
    def __init__(self, prior=None):
        self.prior = prior

    def fit(self, X, y):
        self.prior = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = self.prior if self.prior is not None else 0.7255
        n = len(X)
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0 - p
        probs[:, 1] = p
        return probs

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class HydantoinDetectorClassifier:
    """Baseline 1: Heuristic classifier predicting activity solely from hydantoin substructure."""
    def __init__(self, p_hyd=0.8431, p_non_hyd=0.6078):
        self.p_hyd = p_hyd
        self.p_non_hyd = p_non_hyd

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, is_hyd_array):
        """
        Accepts boolean array of is_hydantoin indicators or list of SMILES.
        Returns 2D probabilities [P(Inactive), P(Active)].
        """
        if isinstance(is_hyd_array[0], (bool, np.bool_)):
            flags = np.array(is_hyd_array, dtype=bool)
        else:
            flags = np.array([is_hydantoin(s) for s in is_hyd_array], dtype=bool)

        p_act = np.where(flags, self.p_hyd, self.p_non_hyd)
        probs = np.zeros((len(flags), 2), dtype=float)
        probs[:, 0] = 1.0 - p_act
        probs[:, 1] = p_act
        return probs

    def predict(self, is_hyd_array):
        probs = self.predict_proba(is_hyd_array)[:, 1]
        return (probs >= 0.5).astype(int)


class Tanimoto1NNClassifier:
    """Baseline 2: 1-Nearest Neighbor Tanimoto similarity to training fold actives."""
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        self.train_active_fps = []

    def fit(self, train_fps_or_smiles, y):
        y_arr = np.array(y)
        active_indices = np.where(y_arr == 1)[0]
        if len(active_indices) == 0:
            self.train_active_fps = []
            return self

        if isinstance(train_fps_or_smiles[0], str):
            fps = [self.gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_fps_or_smiles]
        else:
            fps = train_fps_or_smiles

        self.train_active_fps = [fps[i] for i in active_indices]
        return self

    def predict_proba(self, test_fps_or_smiles):
        if isinstance(test_fps_or_smiles[0], str):
            test_fps = [self.gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in test_fps_or_smiles]
        else:
            test_fps = test_fps_or_smiles

        n = len(test_fps)
        if len(self.train_active_fps) == 0:
            sims = np.zeros(n, dtype=float)
        else:
            sims = np.zeros(n, dtype=float)
            for i, tfp in enumerate(test_fps):
                bulk = DataStructs.BulkTanimotoSimilarity(tfp, self.train_active_fps)
                sims[i] = max(bulk) if len(bulk) > 0 else 0.0

        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 1.0 - sims
        probs[:, 1] = sims
        return probs

    def predict(self, test_fps_or_smiles, threshold=0.5):
        probs = self.predict_proba(test_fps_or_smiles)[:, 1]
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING BASELINE SUITE IMPLEMENTATION")
    print("=" * 80)

    # Unit tests
    b0 = RandomFloorClassifier().fit(np.zeros((10, 2)), np.array([1]*7 + [0]*3))
    p0 = b0.predict_proba(np.zeros((5, 2)))[:, 1]
    print(f"Baseline 0 predicted prior: {p0[0]:.4f} (Expected: 0.7000)")
    assert np.isclose(p0[0], 0.7)

    b1 = HydantoinDetectorClassifier()
    p1 = b1.predict_proba([True, False])[:, 1]
    print(f"Baseline 1 predicted probabilities: Hyd={p1[0]:.4f}, Non-Hyd={p1[1]:.4f}")
    assert np.isclose(p1[0], 0.8431)
    assert np.isclose(p1[1], 0.6078)

    print("[SUCCESS] Baselines verified successfully!")
    print("=" * 80)
