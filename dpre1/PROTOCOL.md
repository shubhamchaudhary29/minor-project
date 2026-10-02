# Experimental Benchmark Protocol: DprE1 Structure-Based Docking vs. Ligand-Based Machine Learning

**Project Title**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation*  
**Status**: Frozen Benchmark Protocol (Phase 1 Gate Evaluation Complete)  
**Author**: Computational Chemistry & Cheminformatics Benchmark Team  
**Date**: October 2026  

---

## 1. Executive Summary & Objective

The primary objective of this benchmark is to rigorously determine whether structure-based molecular docking (AutoDock Vina) provides statistically significant predictive gain—either as an independent scoring tool or in combination with 2D ligand-based machine learning (LB-ML)—for prioritizing novel decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1 / Rv3790) inhibitors under realistic scaffold-split cross-validation.

In virtual screening campaigns, ligand-based models often demonstrate deceptively high retrospective performance under random train/test splits due to memorization of homologous chemical scaffolds. When evaluated across out-of-scaffold test sets, performance frequently collapses. Molecular docking is hypothesized to be scaffold-agnostic because it models direct 3D atomic complementarity within the protein binding pocket. However, docking is susceptible to scoring function artifacts, molecular weight bias, and cofactor sensitivity. This benchmark establishes a reproducible, leak-free framework to quantify the true value added by structure-based modeling against strong, calibrated baselines.

---

## 2. Biological Ground Truths & Structural System

### 2.1 Target Overview
- **Organism**: *Mycobacterium tuberculosis* (strain ATCC 25618 / H37Rv).
- **Target Name**: Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1).
- **Gene Identifier**: `Rv3790`.
- **UniProt Accession**: `P9WJF1` (461 amino acids).  
  *(Note on UniProt annotation: UniProt `P9WGI1` is historically cross-referenced in some legacy databases but corresponds to RNA polymerase sigma factor SigA / Rv2703; `P9WJF1` is the verified DprE1 entry).*
- **ChEMBL Target Identifiers**:
  - `CHEMBL5542`: Mandated project audit target.
  - `CHEMBL3804751`: Canonical *M. tuberculosis* DprE1 single-protein record.

### 2.2 Catalytic Mechanism & Essential Cofactor
- **Cofactor Requirement**: Flavin Adenine Dinucleotide (FAD) is tightly bound in the active site and is biochemically indispensable for epimerization of decaprenylphosphoryl-$\beta$-D-ribose (DPR) to decaprenylphosphoryl-$\beta$-D-2'-keto-erythro-pentofuranose (DPX).
- **Cofactor Rule**: Under no circumstances should FAD be deleted or stripped during receptor grid generation or docking preparation. FAD forms direct hydrogen bonds and steric boundaries of the ligand-binding pocket.

### 2.3 Inhibitor Classes & Mechanistic Divergence
1. **Covalent Inhibitors (Nitro-aromatics)**:
   - *Archetypes*: 1,3-Benzothiazin-4-ones (BTZ043, PBTZ169 / Macozinone).
   - *Mechanism*: The nitro group undergoes FAD-mediated reduction via a transient nitroso intermediate, which subsequently undergoes nucleophilic attack by the thiol group of Cys387, forming a stable covalent semimercaptal adduct.
   - *Exclusion Rule*: Covalent nitro-adducts are chemically invalid for standard non-covalent rigid/flexible docking scoring functions (e.g., standard AutoDock Vina). They must be excluded from standard docking benchmarks or explicitly modeled via covalent docking scripts.
2. **Non-Covalent Inhibitors**:
   - *Archetypes*: Quinoxalines/pyrroles (e.g., CT325 in PDB `4P8K`), 1,4-azaindoles (e.g., TBA-7371 in PDB `6HEZ`), 2-aminoquinolines (PDB `5OEL`, `5OEP`).
   - *Receptor Selection*: High-resolution non-covalent co-crystal structures with intact non-reduced FAD:
     - Primary Reference: **PDB `4P8K`** (CT325 complex, 2.25–2.49 Å resolution).
     - Secondary Reference: **PDB `6HEZ`** (TBA-7371 complex, 2.00–2.30 Å resolution).
     - Auxiliary Reference: **PDB `4P8L`** (2.02 Å resolution) and **PDB `4P8N`** (1.79 Å resolution).

---

## 3. Phase 1 Data Audit & Gate 1 Decision

### 3.1 Funnel Metrics
The bioactivity records were audited through the Phase 1 Gate 1 Funnel using RDKit structure canonicalization, salt stripping (largest organic fragment), and unit normalization to $pIC_{50} = 9 - \log_{10}(\text{nM})$:

| Metric | Mandated Audit (`CHEMBL5542`) | True DprE1 Audit (`CHEMBL3804751`) | Gate Threshold |
| :--- | :--- | :--- | :--- |
| **Raw Bioactivities** | 21,721 | 306 | — |
| **Tier A (Biochemical $IC_{50}/K_i$)** | 50 | 159 | — |
| **Unique Chemical Structures** | 48 | 147 | — |
| **Active Compounds ($pIC_{50} \ge 6.0$)** | 2 | 74 | — |
| **Inactive Compounds ($pIC_{50} < 5.0$)** | 40 | 28 | — |
| **Gray Zone Compounds ($5.0 \le pIC_{50} < 6.0$)** | 6 | 45 | Excluded |
| **Labeled Outside Gray Zone** | 42 | 102 | — |
| **Total Unique Murcko Scaffolds** | 16 | 45 | $\ge 30$ |
| **Scaffolds with $\ge 3$ Compounds** | **2** | **6** | $\ge 10$ |
| **Gate 1 Verdict** | **FAIL / PIVOT** | **FAIL / PIVOT** | **PASS** requires both |

### 3.2 Formal Pivot Directive
Neither raw biochemical tier alone possesses the required $\ge 10$ Murcko scaffolds containing $\ge 3$ congeneric compounds necessary to train gradient-boosted trees and avoid severe fold collapse under strict scaffold-split cross-validation.

**Actionable Pivot Protocol for Phase 2**:
1. **Tier Combination**: Merge Tier A (biochemical $IC_{50}$) with Tier B (whole-cell phenotypic *M. tuberculosis* MIC assays from `CHEMBL3804751` and verified literature series) after filtering out compounds with known off-target mycobacterial mechanisms.
2. **Decoy / Measured Inactive Augmentation**: Augment active scaffolds with property-matched, uncharged Decoy / Inactive libraries generated via DeepChem / DUD-E criteria (matching molecular weight, calculated $\log P$, hydrogen bond donors/acceptors, and rotatable bonds) to establish robust non-trivial negative classes.
3. **Dual Metric Track**: Report both cross-scaffold classification (PR-AUC) and continuous rank correlation (Spearman's $\rho$ and Kendall's $\tau$) between docking scores and experimental binding affinities.

---

## 4. Benchmark Architecture & Modeling Protocol

### 4.1 Evaluation Endpoints & Metrics
- **Primary Endpoint**: Precision-Recall Area Under Curve (**PR-AUC**) under Bemis-Murcko scaffold-split cross-validation. PR-AUC is chosen over ROC-AUC as the primary metric because virtual screening is inherently class-imbalanced, where the cost of false positives is exceptionally high.
- **Secondary Metrics**:
  - Receiver Operating Characteristic Area Under Curve (**ROC-AUC**).
  - Enrichment Factor at 1% ($EF_{1\%}$):
    $$EF_{1\%} = \frac{\text{Hits in top 1\% of ranked list} / N_{1\%}}{\text{Total active hits} / N_{\text{total}}}$$
  - Balanced Accuracy and Cohen's Kappa ($\kappa$) at optimal decision threshold.
  - Spearman rank correlation ($\rho$) for continuous binding potency predictions.

### 4.2 Baseline Models
All machine learning and docking predictions must be benchmarked against two non-trivial baselines:
1. **Majority Class Random Floor**: Expected PR-AUC equal to the positive class prevalence ($P / (P + N)$) and ROC-AUC of 0.50.
2. **1-Nearest Neighbor (1-NN) Tanimoto Similarity Baseline**: Assigns test query active probability based on the maximum Tanimoto similarity to any training set active using Morgan fingerprints:
   $$\hat{y}_{\text{test}} = \max_{i \in \text{Actives}_{\text{train}}} T_c(\mathbf{f}_{\text{test}}, \mathbf{f}_i)$$
   This explicitly benchmarks whether complex ML or docking captures anything beyond basic 2D memorization.

### 4.3 Primary Machine Learning Models
- **Logistic Regression**: $L_2$-regularized linear baseline with balanced class weighting.
- **Random Forest**: 100 trees, `max_features='sqrt'`, `min_samples_split=3`, balanced class weighting.
- **Gradient Boosted Trees (LightGBM / XGBoost)**: Max depth 6, learning rate 0.05, early stopping based on validation PR-AUC.

### 4.4 Molecular Featurization
1. **Topological / Fingerprint Features**: Morgan Fingerprints (ECFP4 equivalent) generated via RDKit (`radius=2`, `nBits=2048`, `useChirality=True`).
2. **Physicochemical Descriptors**: RDKit 2D descriptor suite (Molecular Weight, SlogP, Topological Polar Surface Area [TPSA], H-Bond Donors [HBD], H-Bond Acceptors [HBA], Rotatable Bond Count, Fraction $sp^3$ carbons).
3. **Docking Affinity Feature**: Standard AutoDock Vina predicted binding free energy ($\Delta G_{\text{bind}}$ in kcal/mol), evaluated as a standalone score and as an orthogonal feature appended to ligand-based vectors.

---

## 5. Structure-Based Docking Specifications

### 5.1 Receptor Preparation & Coordinates
- **Selected Complex**: PDB `4P8K` (chain A, resolution 2.25–2.49 Å).
- **Cofactor**: Residue `FAD` retained in active site with coordinates unmodified. Protonation states calculated at pH 7.4.
- **Solvent & Salts**: Crystallographic waters, sulfates, and buffer molecules stripped.
- **Preparation Tool**: AutoDockFR / Meeko (`mk_prepare_receptor.py`) with Gasteiger / Kollman partial charges assigned.

### 5.2 Search Space Grid Box Definition
- **Center**: Active site cavity defined by the centroid of co-crystallized ligand `38C` in PDB `4P8K`:
  - $X_{\text{center}} \approx 13.8$ Å
  - $Y_{\text{center}} \approx -18.2$ Å
  - $Z_{\text{center}} \approx 28.5$ Å
- **Box Dimensions**: $22.0 \times 22.0 \times 22.0$ Å (encompassing Cys387, Lys418, and the isoalloxazine ring of FAD).
- **Vina Engine Parameters**:
  - `exhaustiveness = 32` (high-accuracy conformational sampling).
  - `num_modes = 9`
  - `energy_range = 3.0` kcal/mol

### 5.3 Ligand Preparation
- Neutral/ionized states at physiological pH 7.4 prepared using RDKit / Meeko (`MoleculePreparation`).
- 3D conformers generated with RDKit ETKDGv3 (`AllChem.EmbedMolecule`).
- Torsion tree and rotatable bonds identified and assigned via Meeko.

---

## 6. Scaffold-Split Cross-Validation Scheme

1. **Scaffold Extraction**: For each sanitized molecule, extract Bemis-Murcko frameworks (`MurckoScaffoldSmiles(includeChirality=False)`).
2. **Scaffold Grouping**: Partition compounds into 5 folds such that:
   $$\text{Scaffolds}(\text{Fold}_i) \cap \text{Scaffolds}(\text{Fold}_j) = \emptyset \quad \forall i \ne j$$
3. **Stratification**: Balance the active-to-inactive ratio across folds using a greedy assignment algorithm.
4. **Reproducibility Seeds**: Benchmark runs are replicated across 5 fixed random seeds:
   ```python
   BENCHMARK_SEEDS = [42, 123, 456, 789, 1011]
   ```
5. **Statistical Significance**: Pairwise comparisons between LB-ML, Docking, and Hybrid (LB-ML + Docking) evaluated using two-sided Wilcoxon signed-rank tests across seed-replicated cross-validation folds.

---

## 7. Protocol Sign-off & Lock
- **Protocol Version**: 1.0.0-frozen
- **Execution Target**: Phase 2 Model Training & Docking Benchmarking
- **Compliance**: Adheres to Best Practices for Machine Learning in Drug Discovery (Bemis-Murcko partitioning, non-trivial baselines, leak-free preprocessing).
