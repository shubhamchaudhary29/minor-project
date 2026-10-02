#!/usr/bin/env python3
"""
src/models.py

Machine Learning Model Suite for DprE1 Benchmark (Phase 3):
Defines model architectures, hyperparameter calibration routines, and leak-free pipelines:
  1. Logistic Regression:
     - L2 penalty, tuned C in [0.01, 0.1, 1.0, 10.0] via inner 3-fold CV strictly on train folds.
  2. Random Forest:
     - 300 trees, min_samples_split=4, class_weight='balanced_subsample'.
  3. LightGBM:
     - Shallow boosted trees (max_depth=4), learning_rate=0.05, n_estimators=100.
  4. Descriptor Pipelines:
     - StandardScaler fitted strictly inside training folds to prevent data leakage.
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TunedLogisticRegression:
    """Logistic Regression with inner-CV tuning of regularization parameter C."""
    def __init__(self, c_grid=(0.01, 0.1, 1.0, 10.0), random_state=42):
        self.c_grid = c_grid
        self.random_state = random_state
        self.best_c_ = 1.0
        self.model_ = None

    def fit(self, X, y):
        # 3-Fold Stratified Inner CV to select C
        param_grid = {"C": list(self.c_grid)}
        base_lr = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state
        )
        grid = GridSearchCV(
            base_lr,
            param_grid=param_grid,
            scoring="average_precision",
            cv=3,
            n_jobs=1
        )
        grid.fit(X, y)
        self.best_c_ = grid.best_params_["C"]
        self.model_ = grid.best_estimator_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class ScaledModelPipeline:
    """Pipelines a StandardScaler with a model, strictly fitting the scaler on training data."""
    def __init__(self, base_model):
        self.base_model = base_model
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.base_model.fit(X_scaled, y)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.base_model.predict_proba(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.base_model.predict(X_scaled)


def get_model(model_name: str, feature_type: str = "ecfp4_counts", seed: int = 42):
    """
    Factory function returning the instantiated model or pipeline for the given feature type.
    """
    if model_name == "Logistic Regression":
        lr = TunedLogisticRegression(c_grid=(0.01, 0.1, 1.0, 10.0), random_state=seed)
        if feature_type == "rdkit_2d":
            return ScaledModelPipeline(lr)
        return lr

    elif model_name == "Random Forest":
        rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_split=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1
        )
        if feature_type == "rdkit_2d":
            return ScaledModelPipeline(rf)
        return rf

    elif model_name == "LightGBM":
        lgb = LGBMClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=100,
            class_weight="balanced",
            random_state=seed,
            verbose=-1
        )
        if feature_type == "rdkit_2d":
            return ScaledModelPipeline(lgb)
        return lgb

    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MACHINE LEARNING MODEL SUITE & CALIBRATION")
    print("=" * 80)

    # Smoke test on synthetic data
    np.random.seed(42)
    X_syn = np.random.randn(50, 100)
    y_syn = np.random.choice([0, 1], size=50, p=[0.3, 0.7])

    for m_name in ["Logistic Regression", "Random Forest", "LightGBM"]:
        model = get_model(m_name, feature_type="rdkit_2d", seed=42)
        model.fit(X_syn, y_syn)
        probs = model.predict_proba(X_syn)
        assert probs.shape == (50, 2), f"Expected shape (50, 2), got {probs.shape}"
        print(f"[{m_name:<20}] Smoke test passed. Prob shape: {probs.shape}")

    print("[SUCCESS] All model factories verified successfully!")
    print("=" * 80)
