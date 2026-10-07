# DprE1 Benchmark Project: Structure-Based Docking vs. Ligand-Based Machine Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03.1-green.svg)](https://www.rdkit.org/)
[![AutoDock Vina](https://img.shields.io/badge/AutoDock_Vina-1.2.5-red.svg)](https://vina.scripps.edu/)
[![ChEMBL](https://img.shields.io/badge/ChEMBL-ChEMBL__37-orange.svg)](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL3804751/)
[![License: CC BY-SA 3.0](https://img.shields.io/badge/Data_License-CC_BY--SA_3.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/3.0/)
[![Status: Phase 7 Complete | Viva Ready](https://img.shields.io/badge/Benchmark_Status-Phase_7_Complete_%7C_Viva_Ready-brightgreen.svg)]()

**Author**: Shubham Chaudhary  
**Department**: University School of Automation and Robotics (USAR), GGSIPU, Delhi  
**Degree**: Bachelor of Technology, Minor Project (Academic Year 2025–2026)  
**Repository**: [shubhamchaudhary29/minor-project](https://github.com/shubhamchaudhary29/minor-project.git)  

> **Research Question**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation.*

---

## 1. Project Overview

Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (**DprE1** / **Rv3790**, UniProt **`P9WJF1`**) is an essential flavoenzyme in *Mycobacterium tuberculosis* responsible for catalyzing the epimerization of decaprenylphosphoryl-$\beta$-D-ribofuranose (DPR) to decaprenylphosphoryl-$\beta$-D-2'-ketoribofuranose (DPX), a critical precursor in cell-wall arabinogalactan biosynthesis.

While molecular docking and ligand-based machine learning (LB-ML) are ubiquitous in computational drug discovery, published virtual screening benchmarks suffer from four systemic vulnerabilities:
1. **Analogue Leakage**: Evaluating ML models via random train/test splits rather than scaffold- or series-disjoint partitions, creating heavily inflated performance metrics.
2. **Chemotype Imbalance**: Failing to audit dominant chemical series in public repositories (e.g., hydantoins comprising 33.3% of active DprE1 inhibitors from a single publication).
3. **Covalent Scoring Confounding**: Applying non-covalent thermodynamic scoring functions (e.g., AutoDock Vina) to covalent suicide-inhibitors (nitrobenzothiazinones like BTZ043 and PBTZ169).
4. **Assay Data Pooling**: Inappropriately pooling isolated biochemical enzyme inhibition ($IC_{50}$) with phenotypic whole-cell MIC assays governed by cell-wall penetration and efflux pumps.

This benchmark establishes a fully audited, leak-free, reproducible evaluation pipeline comparing **Structure-Based Molecular Docking (AutoDock Vina)** against **Ligand-Based Machine Learning (Random Forest, Logistic Regression)** under **Stratified Cluster GroupKFold (5-fold CV)** and **Leave-Hydantoin-Out (LHO)** cross-chemotype transfer.

---

## 2. Benchmark Findings & Headline Performance

All metrics are evaluated on the strictly audited non-covalent cohort ($N=93$: 67 active, 26 inactive; or non-hydantoin subset $N=43$: 25 active, 18 inactive). Confidence intervals (95% BCa / percentile) were determined via 1,000 bootstrap iterations on Pooled Out-Of-Fold (OOF) predictions.

### Key Scientific Insights
1. **No Statistically Detectable Value Added**: All 1,000-sample paired bootstrap 95% confidence intervals for $\Delta\text{ROC} = \text{ROC}_{\text{Hybrid}} - \text{ROC}_{\text{ML}}$ cross zero. Under small-data biochemical constraints ($N=93$), docking does not provide statistically significant improvements over ML at $\alpha = 0.05$.
2. **Naive Heuristic Outperforms Standalone Docking**: A trivial, unparameterized substructure rule (`HydantoinDetectorClassifier`, ROC-AUC = 0.660) outperforms standalone AutoDock Vina (ROC-AUC = 0.610), exposing the heavy impact of chemotype prevalence in public chemical repositories.
3. **Linear Regularity Overcomes Scaffold Inversion**: In cross-chemotype transfer (Leave-Hydantoin-Out), Random Forest collapses to ROC-AUC = 0.527. However, $L_2$-regularized Logistic Regression generalizes robustly across chemical families (**ROC-AUC = 0.703** [0.563, 0.824], PR-AUC = 0.828).
4. **Vina Molecular Weight Size-Bias**: An exploratory post-hoc subgroup analysis reveals a strong empirical correlation between Vina binding affinity and molecular weight ($r = -0.437, p = 1.21 \times 10^{-5}$), causing raw Vina scores to collapse on high-MW compounds ($\text{MW} \ge 400$ Da, ROC-AUC = 0.414). Normalizing by heavy atom count ($\text{LE} = -\Delta G / N_{\text{heavy}}$) rescues high-MW docking performance to ROC-AUC = 0.657.

### Summary Evaluation Matrix

| Generalization Cohort | Model / Screening Strategy | ROC-AUC [95% CI] | PR-AUC [95% CI] | $EF_{10\%}$ | $\Delta \text{ROC}$ vs ML |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track B: Cluster 5-Fold CV**<br>*(N=93, Prior=0.720)* | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | **Random Forest (ECFP4 Counts)** | **0.700** [0.559, 0.814] | **0.833** [0.723, 0.924] | 0.93x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.677** [0.540, 0.794] | **0.806** [0.694, 0.917] | 0.93x | — |
| | RRF (RF + Vina Hybrid) | 0.697 [0.552, 0.832] | 0.814 [0.701, 0.920] | 1.08x | -0.003 [-0.139, +0.113] |
| **Track C: Leave-Hydantoin-Out**<br>*(N=93, Prior=0.720)* | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | Random Forest (ECFP4 Counts) | 0.527 [0.402, 0.646] | 0.777 [0.677, 0.871] | 1.23x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.703** [0.563, 0.824] | **0.828** [0.719, 0.923] | 1.08x | — |
| | RRF (RF + Vina Hybrid) | 0.587 [0.443, 0.715] | 0.761 [0.651, 0.885] | 1.08x | +0.061 [-0.013, +0.138] |
| **Non-Hydantoin Independent Test**<br>*(N=43, Prior=0.581)* | Baseline 1 (Hydantoin Detector) | 0.500 [0.500, 0.500] | 0.581 [0.419, 0.744] | 0.00x | — |
| | AutoDock Vina (Raw Affinity) | 0.647 [0.470, 0.820] | 0.656 [0.503, 0.881] | 0.86x | — |
| | Random Forest (ECFP4 Counts) | 0.660 [0.491, 0.810] | **0.790** [0.621, 0.919] | 1.72x | — |
| | **Logistic Regression (ECFP4 Counts)** | **0.704** [0.524, 0.871] | 0.711 [0.541, 0.932] | 0.86x | — |
| | **RRF (RF + Vina Hybrid)** | **0.709** [0.543, 0.878] | 0.724 [0.554, 0.929] | 1.29x | +0.049 [-0.068, +0.198] |

---

## 3. Repository Architecture

```text
dpre1/
├── app/
│   └── streamlit_app.py                 # Interactive Streamlit screening & triage application
├── data/
│   ├── raw/
│   │   ├── chembl_dpre1_raw.csv         # 306 verified records for CHEMBL3804751
│   │   └── pdb_audit.csv                # 38 cataloged crystal structures
│   ├── interim/
│   │   └── gate1_compounds.csv          # Sanitized and curated compound set
│   └── processed/
│       ├── gate1_compounds.csv          # Curated compound set with is_covalent flag
│       ├── stratified_cluster_folds.csv # 102 labeled compounds (5 folds, >=3 inactives/fold)
│       └── master_predictions_phase5.csv# 93 non-covalent molecules with aligned ML & Vina scores
├── docking/
│   ├── configs/
│   │   └── vina_4P8K.txt                # Verified Vina grid configuration
│   ├── ligands/                         # Prepared ligand conformers & crystal reference 38C
│   ├── outputs/                         # Batch Vina docking outputs
│   └── receptors/
│       ├── 4P8K.pdb                     # Cleaned receptor with intact catalytic FAD
│       └── 4P8K_receptor.pdbqt          # OpenBabel-parameterized receptor PDBQT
├── notebooks/
│   └── exploration.ipynb                # Interactive exploratory analysis
├── reports/
│   ├── figures/                         # High-resolution 300 DPI benchmark figures
│   │   ├── roc_pr_master_curves.png     # Dual-panel ROC & PR master curves
│   │   ├── scaffold_inversion_barplot.png # Scaffold inversion barplot
│   │   ├── mw_ablation_effect.png       # Molecular weight size-bias scatter & performance flip
│   │   ├── applicability_domain_decay.png
│   │   ├── top_ecfp4_substructures.png
│   │   └── vina_score_vs_mw.png
│   ├── DprE1_Minor_Project_Report.md    # Comprehensive academic thesis dissertation (10 chapters)
│   ├── BENCHMARK_FINDINGS.md            # Frozen scientific synthesis & viva defense report
│   ├── phase5_benchmark_master.csv      # Master evaluation table with 1,000 bootstrap CIs
│   ├── ablation_results.csv             # MW and domain stratification results
│   ├── discordance_analysis.csv         # Compound-level quadrant discordance case studies
│   ├── discordance_interactions.csv     # Residue-level protein-ligand contact analysis (PDB 4P8K)
│   ├── interaction_analysis_summary.txt # Biophysical contact summary across discordance cohorts
│   ├── gate2_redocking.txt              # Gate 2 crystallographic validation (RMSD = 1.282 Å)
│   ├── covalent_audit.csv               # Mechanism breakdown (covalent vs non-covalent)
│   ├── hydantoin_cluster_analysis.json  # Cluster dominance and partition report
│   └── DprE1_Literature_Audit_verified.csv # 26 CrossRef-verified DOIs
├── src/
│   ├── analyze_interactions.py          # Residue contact analysis engine (Chain A + FAD 501)
│   ├── integrate.py                     # Multi-modal fusion engine (RRF, MPR, Z-score)
│   ├── evaluate_integration.py          # Master benchmark evaluation & 1,000 bootstrap CIs
│   ├── ablation_analysis.py             # MW stratification & quadrant discordance analysis
│   ├── plot_figures.py                  # Publication figure generation suite
│   ├── dock.py                          # Receptor prep, Gate 2 redocking, & batch Vina
│   ├── models.py                        # ML classifiers (Logistic Regression, Random Forest)
│   ├── features.py                      # Fingerprint & descriptor calculation engine
│   ├── baselines.py                     # Hydantoin detector, 1-NN Tanimoto, Random Floor
│   ├── controls.py                      # Y-randomization & applicability domain controls
│   ├── splits.py                        # Stratified cluster fold partitioning
│   └── gate1_funnel.py                  # Gate 1 bioactivity curation funnel
├── DATA_CARD.md                         # Data provenance & bias notes
├── PROTOCOL.md                          # Frozen Protocol v1.1.2 with Deviation Log
└── README.md                            # Project documentation
```

---

## 4. Environment Installation & Activation

The environment runs on standard x86_64 Linux CPUs with zero GPU requirements:

```bash
# Create the environment using micromamba or conda
micromamba create -y -n dpre1 -f environment.yml

# Activate the environment
micromamba activate dpre1
```

---

## 5. Execution Pipeline

Execute the end-to-end benchmark workflow from data curation to hybrid evaluation and web deployment:

```bash
# 1. Gate 1 bioactivity curation funnel (ChEMBL3804751)
python src/gate1_funnel.py

# 2. Stratified cluster fold generation (5 folds, >=3 inactives/fold)
python src/splits.py

# 3. Phase 3 Ligand ML training and out-of-fold cross-validation
python src/evaluate_ml.py

# 4. Phase 4 Receptor preparation & Gate 2 redocking validation (RMSD = 1.282 Å)
python src/dock.py --step prep
python src/dock.py --step redock
python src/dock.py --step batch

# 5. Phase 5 Multi-modal integration & Rank Fusion (RRF, MPR, Z-Score)
python src/integrate.py
python src/evaluate_integration.py
python src/ablation_analysis.py
python src/plot_figures.py

# 6. Launch Phase 6 Interactive Streamlit Web Application
streamlit run app/streamlit_app.py
```

---

## 6. Phase 6 Streamlit Application

The repository includes a production-ready single-page application in `app/streamlit_app.py`:
- **Real-Time SMILES Validation**: Instant RDKit parsing, structure rendering, and Lipinski physicochemical profiling.
- **Covalent Warhead Sentry**: Automated SMARTS detection of aromatic nitro/nitroso warheads with an alert regarding physical invalidity in non-covalent docking.
- **Live ML Inference**: Real-time $P(\text{Active})$ predictions from calibrated Logistic Regression and Random Forest models.
- **Applicability Domain Monitor**: Live Tanimoto distance computation against the ChEMBL3804751 training actives with out-of-domain warning (< 0.40).
- **Docking Data Vault**: Immediate retrieval of precomputed AutoDock Vina affinity, Ligand Efficiency, and percentile ranks for benchmark compounds.

---

## 7. Citation & Data Provenance

Bioactivity data retrieved from **ChEMBL 37** (`CHEMBL3804751`, canonical *M. tuberculosis* H37Rv DprE1, UniProt **`P9WJF1`**). Structure coordinates from **RCSB PDB `4P8K`** (Chain A at 2.49 Å resolution with catalytic FAD). All data and code are distributed under the [CC BY-SA 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/).
