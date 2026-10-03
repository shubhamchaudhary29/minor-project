# Scientific Synthesis Report: DprE1 Virtual Screening Benchmark

**Document**: `reports/BENCHMARK_FINDINGS.md`  
**Project**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation*  
**Author**: Shubham Chaudhary  
**Protocol Version**: `v1.1.2-frozen`  
**Date**: October 3, 2026  
**Repository**: [minor-project](https://github.com/shubhamchaudhary29/minor-project.git)  

---

## 1. Executive Summary & The Core Scientific Answer

### The Central Question
> **Does structure-based docking add predictive value over ligand-based machine learning for prioritizing non-covalent DprE1 inhibitors under scaffold-split evaluation?**

### The Headline Verdict: **No Statistically Detectable Added Value at $N=93$**

Under rigorous, leak-free evaluation on the curated non-covalent biochemical corpus ($N=93$: 67 active, 26 inactive), **structure-based docking does NOT add statistically detectable predictive value over 2D ligand-based machine learning**:

1. **Null Statistical Hypothesis Cannot Be Rejected**:
   - Across all evaluated splits (Cluster 5-Fold CV, Leave-Hydantoin-Out, and the independent Non-Hydantoin test partition), the 1,000-sample paired bootstrap 95% confidence intervals for $\Delta\text{ROC} = \text{ROC}_{\text{Hybrid}} - \text{ROC}_{\text{ML}}$ **cross zero** (e.g., Track B: $[-0.139, +0.113]$; Track C: $[-0.013, +0.138]$; Non-Hydantoins: $[-0.068, +0.198]$).
   - The primary benchmark is underpowered to confirm any independent predictive contribution of AutoDock Vina over ligand-based ML at $\alpha = 0.05$.

2. **The Naive Substructure Baseline Beats Standalone Docking**:
   - The trivial `HydantoinDetectorClassifier` baseline—a 1-rule heuristic checking for the hydantoin core (`O=C1NC(=O)NC1`) without any training—achieves an **ROC-AUC of 0.660** [0.541, 0.758].
   - In comparison, standalone AutoDock Vina achieves an **ROC-AUC of only 0.610** [0.463, 0.746]. A simple 2D substructure lookup outperforms full 3D physics-based molecular docking due to extreme chemotype prevalence bias in public chemical repositories.

3. **Linear Regularity Overcomes Cross-Chemotype Generalization Collapse**:
   - In cross-chemotype transfer (Track C: Leave-Hydantoin-Out), Random Forest suffers an out-of-scaffold generalization collapse to **ROC-AUC = 0.527** [0.402, 0.646] (random floor: 0.500).
   - In contrast, $L_2$-regularized **Logistic Regression** generalizes effectively across chemotype boundaries, achieving **ROC-AUC = 0.703** [0.563, 0.824] and **PR-AUC = 0.828** [0.719, 0.923] on ECFP4 count fingerprints. Linear models with strong regularization retain predictive utility on unseen scaffolds without requiring 3D structural docking scores.

4. **Post-Hoc Size-Bias Explains Molecular Weight Subgroups**:
   - An exploratory post-hoc subgroup analysis indicates that raw Vina docking scores correlate strongly with molecular weight ($r = -0.437$ with affinity / $+0.437$ with binding magnitude, $p = 1.21 \times 10^{-5}$), despite molecular weight having zero correlation with true biological activity ($r = -0.017, p = 0.869$).
   - Consequently, raw docking scores collapse on compounds with $\text{MW} \ge 400$ Da (ROC-AUC = 0.414) due to size-driven false positives, a technical artifact partially remediated post-hoc by Ligand Efficiency ($\text{LE} = -\Delta G / N_{\text{heavy}}$, ROC-AUC = 0.657).

---

## 2. Master Benchmark Performance Matrix

All metrics are evaluated on the non-covalent cohort ($N=93$: 67 active, 26 inactive; or non-hydantoin subset $N=43$: 25 active, 18 inactive). Confidence intervals (95% BCa / percentile) were generated via 1,000 bootstrap iterations on Pooled Out-Of-Fold (OOF) predictions.

| Cohort | Virtual Screening Method | ROC-AUC [95% CI] | PR-AUC [95% CI] | $EF_{10\%}$ | $\Delta \text{ROC}_{\text{Hybrid}-\text{ML}}$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track B: Cluster 5-Fold CV**<br>*(N=93, Prior=0.720)* | Baseline 0 (Random Floor) | 0.500 [0.500, 0.500] | 0.720 [0.634, 0.806] | 0.77x | — |
| | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | Baseline 2 (1-NN Tanimoto) | 0.616 [0.495, 0.742] | 0.820 [0.718, 0.901] | 1.39x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | AutoDock Vina (Ligand Efficiency) | 0.464 [0.312, 0.623] | 0.679 [0.569, 0.814] | 0.93x | — |
| | **Random Forest (ECFP4 Counts)** | **0.700** [0.559, 0.814] | **0.833** [0.723, 0.924] | 0.93x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.677** [0.540, 0.794] | **0.806** [0.694, 0.917] | 0.93x | — |
| | RRF (RF + Vina Hybrid) | 0.697 [0.552, 0.832] | 0.814 [0.701, 0.920] | 1.08x | -0.003 [-0.139, +0.113] |
| | RRF (LR + Vina Hybrid) | 0.691 [0.543, 0.826] | 0.787 [0.684, 0.912] | 0.93x | +0.014 [-0.101, +0.114] |
| | MPR (RF + Vina Hybrid) | 0.726 [0.590, 0.849] | 0.829 [0.726, 0.934] | 1.23x | +0.027 [-0.095, +0.138] |
| | Z-Score Sum (RF + Vina Hybrid) | 0.707 [0.567, 0.834] | 0.826 [0.720, 0.929] | 1.08x | +0.007 [-0.131, +0.119] |
| **Track C: Leave-Hydantoin-Out**<br>*(N=93, Prior=0.720)* | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | Baseline 2 (1-NN Tanimoto) | 0.525 [0.394, 0.657] | 0.745 [0.642, 0.855] | 0.93x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | Random Forest (ECFP4 Counts) | 0.527 [0.402, 0.646] | 0.777 [0.677, 0.871] | 1.23x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.703** [0.563, 0.824] | **0.828** [0.719, 0.923] | 1.08x | — |
| | RRF (RF + Vina Hybrid) | 0.587 [0.443, 0.715] | 0.761 [0.651, 0.885] | 1.08x | +0.061 [-0.013, +0.138] |
| | RRF (LR + Vina Hybrid) | 0.677 [0.527, 0.812] | 0.786 [0.680, 0.911] | 0.93x | -0.026 [-0.115, +0.064] |
| | MPR (RF + Vina Hybrid) | 0.587 [0.443, 0.714] | 0.761 [0.650, 0.885] | 1.08x | +0.060 [-0.015, +0.143] |
| | Z-Score Sum (RF + Vina Hybrid) | 0.584 [0.446, 0.717] | 0.756 [0.653, 0.881] | 1.08x | +0.058 [-0.020, +0.147] |
| **Non-Hydantoin Independent Test**<br>*(N=43, Prior=0.581)* | Baseline 1 (Hydantoin Detector) | 0.500 [0.500, 0.500] | 0.581 [0.419, 0.744] | 0.00x | — |
| | Baseline 2 (1-NN Tanimoto) | 0.587 [0.393, 0.765] | 0.663 [0.478, 0.850] | 1.29x | — |
| | AutoDock Vina (Raw Affinity) | 0.647 [0.470, 0.820] | 0.656 [0.503, 0.881] | 0.86x | — |
| | Random Forest (ECFP4 Counts) | 0.660 [0.491, 0.810] | **0.790** [0.621, 0.919] | 1.72x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.704** [0.524, 0.871] | 0.711 [0.541, 0.932] | 0.86x | — |
| | RRF (RF + Vina Hybrid) | **0.709** [0.543, 0.878] | 0.724 [0.554, 0.929] | 1.29x | +0.049 [-0.068, +0.198] |
| | RRF (LR + Vina Hybrid) | 0.687 [0.509, 0.866] | 0.668 [0.509, 0.906] | 0.86x | -0.018 [-0.147, +0.109] |
| | MPR (RF + Vina Hybrid) | 0.706 [0.539, 0.868] | 0.726 [0.557, 0.923] | 1.29x | +0.046 [-0.071, +0.194] |

---

## 3. Key Mechanistic Findings & Methodological Nuances

### 3.1 The Scaffold Inversion: Tree Ensembles Collapse, Linear Models Generalize
- In Random 5-Fold CV (Track A), Random Forest achieves an artificially elevated **ROC-AUC = 0.799** and **PR-AUC = 0.923** due to analogue leakage across structurally similar congeners.
- In Cluster 5-Fold CV (Track B), RF drops to **0.700**, and in Leave-Hydantoin-Out (Track C), RF collapses completely to **0.527**, unable to map non-hydantoin decision trees onto hydantoin features.
- In contrast, regularized **Logistic Regression** maintains strong generalization (**ROC-AUC = 0.703** in Track C, **0.704** in Non-Hydantoins). By learning smooth, additive feature weights across conserved pharmacophores rather than rigid hierarchical splits, linear models generalize across distinct chemotype families far better than unpruned tree ensembles.

### 3.2 Exploratory Subgroup Analysis: Molecular Weight Size-Bias
- **Empirical Correlation**:
  - Pearson correlation between Molecular Weight and raw Vina affinity: **$r = -0.437$ ($p = 1.21 \times 10^{-5}$)**.
  - Point-biserial correlation between true Bioactivity Label and Molecular Weight: **$r = -0.017$ ($p = 0.869$)**.
- **Post-Hoc Stratification**:
  * **Low MW (< 400 Da, $N=35$)**: AutoDock Vina achieves an exploratory **ROC-AUC = 0.780** [PR-AUC = 0.884], outperforming Random Forest (0.640).
  * **High MW ($\ge 400$ Da, $N=58$)**: AutoDock Vina collapses to **ROC-AUC = 0.414** due to size-driven false positives (e.g., `CHEMBL4760909`: inactive, MW 482.6 Da, assigned top score $-10.64$ kcal/mol).
  * **Ligand Efficiency Correction**: Normalizing by heavy atom count ($\text{LE} = -\Delta G / N_{\text{heavy}}$) restores high-MW performance to **ROC-AUC = 0.657**, demonstrating that empirical scoring functions require size correction when screening diverse molecular weight libraries.

### 3.3 Quadrant Discordance Case Studies
- **Docking Rescues ML (False Negatives of ML)**:
  * `CHEMBL4858566` & `CHEMBL4848357` (Active non-hydantoins): Novel pyrrole/quinoxaline derivatives assigned low ML probabilities ($P \approx 0.54$, bottom 16th percentile) in LHO. Vina recognized shape complementarity with the CT325 pocket ($-9.14$ kcal/mol, top 26th percentile), rescuing them into the hit pool.
- **ML Rescues Docking (False Negatives of Rigid Docking)**:
  * `CHEMBL3262462` & `CHEMBL5197700` (Active rigid scaffolds): Penalized by rigid-pocket Vina docking ($-8.25$ kcal/mol, bottom 12th percentile) due to steric clashes with Lys418 / Tyr314 sidechains. 2D ML recognized active ECFP4 subgraphs ($P = 0.677$, 94th percentile), preventing their false rejection.

---

## 4. Methodological Defense & Viva Voce Critique Points

When defending this benchmark before academic examiners, the following core methodological foundations must be highlighted:

1. **Strict Assay Tier Separation (No Data Pooling)**:
   - Biochemical enzyme inhibition ($IC_{50}$) and whole-cell phenotypic MIC measure fundamentally distinct biophysical phenomena. Whole-cell MIC is heavily confounded by mycobacterial cell-wall permeability (mycolic acid barrier) and active efflux (e.g., MmpL5). Combining Tier A and Tier B data corrupts molecular scoring functions.
2. **Preservation of Catalytic FAD Cofactor**:
   - Flavin Adenine Dinucleotide (FAD) participates directly in the epimerization mechanism and forms the active site floor. Parameterizing FAD in `4P8K_receptor.pdbqt` prevents the unphysical binding cavities produced when cofactors are stripped.
3. **Gate 2 Redocking Pose Validation**:
   - Crystallographic redocking of co-crystal ligand `38C (CT325)` in PDB `4P8K` reproduced the experimental pose with **RMSD = 1.282 Å**, decisively passing the Gate 2 threshold (< 2.0 Å).
4. **Honest Reporting of Null / Underpowered Findings**:
   - Rather than selectively reporting positive numbers, we rigorously show that at $N=93$ non-covalent inhibitors, hybrid rank fusion does not provide statistically significant improvements over ML alone. Demonstrating the limits of virtual screening modalities under small-data regimes is gold-standard scientific research.

---

## 5. Artifact & Repository Index

- **Master Dataset**: [`data/processed/master_predictions_phase5.csv`](file:///home/gigachad/minor_project/data/processed/master_predictions_phase5.csv) ($N=93$ aligned compounds with ML probabilities, Vina scores, and hybrid ranks).
- **Master Metrics**: [`reports/phase5_benchmark_master.csv`](file:///home/gigachad/minor_project/reports/phase5_benchmark_master.csv) (1,000-sample bootstrap CIs).
- **Ablation Tables**: [`reports/ablation_results.csv`](file:///home/gigachad/minor_project/reports/ablation_results.csv) (MW and domain stratification).
- **Discordance Cases**: [`reports/discordance_analysis.csv`](file:///home/gigachad/minor_project/reports/discordance_analysis.csv) (Detailed failure mode analysis).
- **Publication Figures (300 DPI)**:
  * [`reports/figures/roc_pr_master_curves.png`](file:///home/gigachad/minor_project/reports/figures/roc_pr_master_curves.png)
  * [`reports/figures/scaffold_inversion_barplot.png`](file:///home/gigachad/minor_project/reports/figures/scaffold_inversion_barplot.png)
  * [`reports/figures/mw_ablation_effect.png`](file:///home/gigachad/minor_project/reports/figures/mw_ablation_effect.png)
