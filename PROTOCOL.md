# Experimental Benchmark Protocol: DprE1 Structure-Based Docking vs. Ligand-Based Machine Learning

**Project Title**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation*  
**Protocol Version**: 1.1.0-frozen (Phase 1.5 Remediation Release)  
**Date**: October 2, 2026  
**Auditor**: Senior Computational Chemist & Research Software Engineer  

---

## Protocol Deviation & Remediation Log (v1.1)

| Deviation ID | Component | Description of Change & Scientific Rationale |
| :--- | :--- | :--- |
| **DEV-01** | Target ID Correction | Purged legacy misannotated target identifier (human DNA polymerase eta / *POLH*). Locked verified target identifier **`CHEMBL3804751`** (canonical *Mycobacterium tuberculosis* H37Rv DprE1, UniProt **`P9WJF1`** / `Rv3790`). |
| **DEV-02** | Statistical Significance | Forbidden fold-level Wilcoxon signed-rank tests due to cross-validation fold dependence. Mandated **1,000-iteration compound-level and cluster-level bootstrap resampling** with 95% bias-corrected and accelerated (BCa) confidence intervals. |
| **DEV-03** | Gate 1 Reformulation | Replaced arbitrary raw scaffold threshold with **Chemotype Balance & Cluster Analysis**. Identified single-paper hydantoin chemotype dominance (34 compounds, 33.3% of labeled set). Strictly retained separation between Tier A (biochemical $IC_{50}$) and Tier B (whole-cell MIC); forbidden data pooling. |
| **DEV-04** | Covalent Docking Partition | Formally partitioned benchmark into: (1) **Non-Covalent Evaluation Set** ($N=93$ labeled molecules: 67 actives, 26 inactives) scored quantitatively by AutoDock Vina, and (2) **Covalent Nitro-Aromatic Set** ($N=9$ labeled molecules: 7 actives, 2 inactives) analyzed exclusively for pre-reaction pocket proximity. |
| **DEV-05** | Receptor & Grid Coordinates | Programmatically computed exact active-site centroid from co-crystallized ligand **`38C (CT325)`** in PDB `4P8K` Chain A: `(17.07, -20.26, 1.49)` Å with bounding box $22.0 \times 22.0 \times 22.0$ Å. |

---

## 1. Executive Summary & Core Objective

The primary objective of this benchmark is to rigorously determine whether structure-based molecular docking (AutoDock Vina) provides genuine predictive value over 2D ligand-based machine learning (LB-ML)—either as an independent scoring engine or as an orthogonal feature in hybrid models—for prioritizing non-covalent decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1 / Rv3790) inhibitors under out-of-distribution, scaffold-split evaluation.

### Core Benchmark Axioms
1. **Strict Tier Separation (No Data Pooling)**: Tier A (biochemical enzyme inhibition $IC_{50}$) and Tier B (whole-cell phenotypic *M. tuberculosis* MIC) measure fundamentally distinct biophysical phenomena. Whole-cell MIC is confounded by cell-wall permeability, efflux pump liability (e.g., MmpL5), and intracellular metabolism. Tier A and Tier B data must never be pooled to artificially inflate dataset size.
2. **Measured Negatives First**: The primary benchmark evaluates models against experimentally measured biochemical inactives ($pIC_{50} < 5.0$). Property-matched in silico decoys (DeepCoy / DUD-E criteria) are evaluated strictly as a secondary experiment (Experiment B).
3. **Rigorous Statistical Comparison**: Differences in PR-AUC and ROC-AUC are evaluated using 1,000-iteration compound-level and cluster-level bootstrapping, generating 95% BCa confidence intervals.

---

## 2. Biological Ground Truths & Target Specification

### 2.1 Target Identifiers
- **Organism**: *Mycobacterium tuberculosis* H37Rv
- **Target Name**: Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1)
- **Gene Symbol**: `Rv3790`
- **UniProt Accession**: **`P9WJF1`** (461 aa). *(Note: UniProt `P9WGI1` is `sigA`/Rv2703 and must never be referenced for DprE1).*
- **ChEMBL Target ID**: **`CHEMBL3804751`** (Single Protein, 306 verified biochemical records).

### 2.2 Catalytic Mechanism & Essential Cofactor
- **FAD Cofactor**: Flavin Adenine Dinucleotide (FAD) is an integral catalytic cofactor non-covalently bound in the active site. FAD participates directly in the two-step epimerization of DPR to DPX and forms the floor of the ligand-binding pocket.
- **Cofactor Rule**: Under no circumstances should FAD be deleted or stripped during receptor grid generation or docking preparation. FAD coordinates must remain present and parameterized with partial charges at physiological pH 7.4.

### 2.3 Mechanistic Divergence & Ligand Classification
- **Covalent Suicide-Inhibitors**: Nitro-aromatic compounds (e.g., BTZ043, PBTZ169 / macozinone) undergo FAD-catalyzed reduction of the aromatic nitro group to a nitroso intermediate, followed by nucleophilic attack from the thiol of Cys387, forming a stable covalent semimercaptal adduct. Standard rigid docking scoring functions (Vina) cannot model covalent bond enthalpy; these molecules are audited separately.
- **Non-Covalent Inhibitors**: Quinoxalines, pyrroles, 1,4-azaindoles, and hydantoins bind reversibly within the pocket adjacent to FAD without adducting Cys387.
- **Reference Receptor Complex**: **PDB `4P8K`** (Chain A, 2.25 Å resolution).
- **Reference Co-crystal Ligand**: Chemical component code **`38C`**, named **`CT325`** in medicinal chemistry literature. Documented as **`38C (CT325)`**.

---

## 3. Curated Benchmark Dataset (Phase 1.5 Ground Truth)

Audited directly from the complete verified biochemical corpus in ChEMBL (`CHEMBL3804751`, $N=306$ records):

```
Total ChEMBL3804751 Records: 306
       │
       ▼
Tier A Biochemical Assays (IC50 / Ki): 159 records
       │
       ▼
RDKit Sanitization & Salt Stripping (Largest Organic Fragment): 147 unique compounds
       │
       ├── Active (pIC50 >= 6.0, <= 1 uM):        74 compounds
       ├── Inactive (pIC50 < 5.0, > 10 uM):       28 compounds
       └── Gray Zone (5.0 <= pIC50 < 6.0):        45 compounds (Held out)
```

### Primary Benchmark Labeled Set ($N=102$)
- **Active Class**: 74 compounds ($pIC_{50} \ge 6.0$, $\le 1\,\mu\text{M}$)
- **Inactive Class**: 28 compounds ($pIC_{50} < 5.0$, $> 10\,\mu\text{M}$)
- **Gray Zone**: 45 compounds ($5.0 \le pIC_{50} < 6.0$, held out from primary classification)

### Docking Substructure Breakdown
- **Non-Covalent Evaluation Set**: **93 molecules** (67 Actives, 26 Inactives) $\rightarrow$ *Primary docking benchmark sample size*.
- **Covalent Nitro-Aromatic Set**: **9 molecules** (7 Actives, 2 Inactives) $\rightarrow$ *Pre-reaction proximity evaluation track*.

---

## 4. Receptor Preparation & Verified Grid Configuration

### 4.1 Receptor Coordinates & Pocket Centroid
- **Receptor Structure**: PDB `4P8K` (Chain A). Water molecules and crystallographic buffer ions stripped; non-covalent cofactor `FAD` retained.
- **Centroid Calculation**: Programmatically calculated as the geometric mean of all 27 heavy atoms of ligand `38C (CT325)` in Chain A:
  $$\mathbf{C} = \frac{1}{N}\sum_{i=1}^N \mathbf{r}_i = (17.07, -20.26, 1.49)\,\text{Å}$$
- **Proximity to Key Catalytic Residues**:
  - Distance to Cys387 S$\gamma$ ($12.56, -17.59, -0.83$ Å): $\approx 5.7$ Å.
  - Distance to FAD N5 ($18.98, -16.60, 2.05$ Å): $\approx 4.2$ Å.

### 4.2 AutoDock Vina Configuration (`docking/configs/vina_4P8K.txt`)
```text
center_x = 17.07
center_y = -20.26
center_z = 1.49

size_x = 22.0
size_y = 22.0
size_z = 22.0

exhaustiveness = 24
num_modes = 9
energy_range = 3
```

---

## 5. Splitting Strategies & Cross-Validation Architecture

### 5.1 The Hydantoin Series Challenge
A single dominant chemotype, the hydantoin series (`O=C(CN1C(=O)NC(c2ccccc2)C1=O)c1ccccc1`, Gao et al., 2018), accounts for **34 out of 102 labeled compounds (33.3%)**.
Naive Bemis-Murcko GroupKFold forces all 34 compounds into a single fold, leaving the remaining 4 folds with only ~17 compounds each, causing massive variance and fold instability.

### 5.2 Dual Evaluation Splitting Scheme
To ensure rigorous and reproducible benchmarking, models are evaluated under two complementary schemes:

1. **Scheme 1: Cluster-Aware Balanced 5-Fold Partitioning**:
   - Compute circular fingerprints: Morgan ECFP4 equivalent (`radius=2`, `nBits=2048`).
   - Cluster using Butina algorithm with Tanimoto distance threshold $0.55$.
   - Assign clusters to 5 folds using a greedy size-balancing algorithm to ensure equal fold sizes (~20 compounds/fold) while strictly preventing intra-cluster train-test leakage.
   - Replicate across 5 fixed random seeds: `[42, 123, 456, 789, 1011]`.

2. **Scheme 2: Leave-Hydantoin-Out (LHO) Chemotype Transfer**:
   - **Track A**: Train on Non-Hydantoins ($N=51$) $\rightarrow$ Test on Hydantoins ($N=51$).
   - **Track B**: Train on Hydantoins ($N=51$) $\rightarrow$ Test on Non-Hydantoins ($N=51$).
   - Explicitly evaluates out-of-distribution generalization to a completely held-out chemotype.

---

## 6. Models, Featurization & Evaluation Metrics

### 6.1 Model Zoo
- **Baselines**:
  - Majority Class Random Floor: PR-AUC $= P / (P + N)$
  - 1-Nearest Neighbor (1-NN) Tanimoto Similarity: Maximum similarity to training actives.
- **Ligand-Based Machine Learning (LB-ML)**:
  - Logistic Regression ($L_2$ penalty, balanced class weighting)
  - Random Forest (100 estimators, balanced class weighting)
  - LightGBM / XGBoost (max depth 6, learning rate 0.05)
- **Structure-Based Modeling**:
  - AutoDock Vina Standalone ($\Delta G_{\text{bind}}$ in kcal/mol)
- **Hybrid Modeling**:
  - LB-ML feature vectors + AutoDock Vina predicted binding free energy.

### 6.2 Primary Endpoint & Statistical Testing
- **Primary Metric**: Precision-Recall Area Under Curve (**PR-AUC**) on out-of-cluster test sets.
- **Secondary Metrics**: ROC-AUC, Enrichment Factor at 1% ($EF_{1\%}$), Balanced Accuracy.
- **Significance Testing**: Non-parametric compound-level bootstrap resampling (1,000 bootstrap iterations) reporting 95% BCa confidence intervals:
  $$\Delta \text{PR-AUC} = \text{PR-AUC}_{\text{Hybrid}} - \text{PR-AUC}_{\text{LB-ML}}$$
  If the 95% BCa confidence interval for $\Delta \text{PR-AUC}$ excludes zero, structure-based docking adds statistically significant predictive value.
