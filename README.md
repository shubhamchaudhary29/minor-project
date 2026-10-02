# DprE1 Benchmark Project: Structure-Based Docking vs. Ligand-Based Machine Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03.1-green.svg)](https://www.rdkit.org/)
[![ChEMBL](https://img.shields.io/badge/ChEMBL-ChEMBL__37-orange.svg)](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL3804751/)
[![License: CC BY-SA 3.0](https://img.shields.io/badge/Data_License-CC_BY--SA_3.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/3.0/)
[![Status: Phase 1.5 Remediated](https://img.shields.io/badge/Benchmark_Status-Phase_1.5_Remediated-brightgreen.svg)]()

> **Research Question**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation.*

---

## 1. Project Overview

Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (**DprE1** / **Rv3790**, UniProt **`P9WJF1`**) is a clinically validated flavoenzyme essential for cell-wall arabinogalactan biosynthesis in *Mycobacterium tuberculosis*.

While molecular docking and ligand-based machine learning (LB-ML) are widely applied to DprE1 virtual screening, existing studies suffer from pervasive methodological vulnerabilities:
1. **Scaffold Leakage**: Evaluating ML models via random train/test splits rather than out-of-scaffold partitions, disguising generalizability failures.
2. **Chemotype Imbalance**: Failing to account for dominant chemical series in public repositories (e.g., the hydantoin series comprising 33.3% of active DprE1 inhibitors).
3. **Covalent Scoring Confounding**: Applying non-covalent docking tools (e.g., AutoDock Vina) to covalent suicide-inhibitors (nitrobenzothiazinones like BTZ043 and PBTZ169).
4. **Data Pooling Confounding**: Inappropriately combining isolated enzyme inhibition ($IC_{50}$) with whole-cell phenotypic MIC assays.

This benchmark establishes a reproducible evaluation pipeline comparing **Structure-Based Docking (AutoDock Vina)** against **Ligand-Based Machine Learning (Random Forest, LightGBM, Logistic Regression)** under **cluster-aware balanced splits** and explicit **covalent vs. non-covalent partitioning**.

---

## 2. Repository Architecture

```text
minor_project/
├── app/                                 # Interactive exploration application
├── data/
│   ├── raw/
│   │   ├── chembl_dpre1_raw.csv         # 306 verified records for CHEMBL3804751
│   │   └── pdb_audit.csv                # 38 cataloged crystal structures
│   ├── interim/
│   │   └── gate1_compounds.csv          # Sanitized and curated compound set
│   └── processed/
│       └── gate1_compounds.csv          # Curated compound set with is_covalent flag
├── docking/
│   ├── configs/
│   │   └── vina_4P8K.txt                # Programmatically verified Vina grid configuration
│   ├── ligands/                         # Prepared ligand conformers
│   ├── outputs/                         # Docking poses and log files
│   └── receptors/
│       └── 4P8K.pdb                     # Cleaned receptor with intact catalytic FAD
├── notebooks/                           # Analysis and plotting notebooks
├── reports/
│   ├── figures/                         # High-resolution benchmark figures
│   ├── DprE1_Literature_Audit.csv       # 22 curated literature entries
│   ├── DprE1_Literature_Audit_verified.csv # 22 live CrossRef-verified DOIs (HTTP 200)
│   ├── chembl_target_audit.txt          # Target verification report for P9WJF1
│   ├── covalent_audit.csv               # Mechanism breakdown (covalent vs non-covalent)
│   ├── hydantoin_cluster_analysis.json  # Cluster dominance and partition report
│   ├── balanced_folds.csv               # Cluster-aware 5-fold assignments
│   └── gate1_funnel_summary.json        # Funnel metrics summary
├── src/
│   ├── check_targets.py                 # Exhaustive ChEMBL target auditor
│   ├── get_pocket_center.py             # Geometric centroid calculator for ligand 38C (CT325)
│   ├── audit_covalent.py                # Covalent warhead classifier
│   ├── analyze_hydantoins.py            # Hydantoin dominance & Butina fold partitioner
│   ├── verify_dois.py                   # Live CrossRef DOI verification engine
│   ├── audit_chembl.py                  # Live ChEMBL bioactivity extractor
│   ├── audit_pdb.py                     # RCSB PDB structural auditor
│   └── gate1_funnel.py                  # Gate 1 curation and funnel engine
├── DATA_CARD.md                         # Data provenance, ChEMBL_37 release, and bias notes
├── environment.yml                      # Pinned Conda environment (Python 3.10)
├── PROTOCOL.md                          # Frozen Protocol v1.1 with Deviation Log
└── README.md                            # Project documentation
```

---

## 3. Environment Installation & Activation

The environment is CPU-compatible with zero GPU dependencies. It pins Python 3.10, RDKit, scikit-learn, AutoDock Vina, Meeko, ProLIF, OpenBabel, pandas, and Streamlit.

```bash
# Create the environment using micromamba or conda
micromamba create -y -n dpre1 -f environment.yml

# Activate the environment
micromamba activate dpre1
```

---

## 4. Phase 1.5 Remediation & Execution Pipeline

Execute the remediation and verification suite:

```bash
# 1. Verify unambiguous target mapping to P9WJF1 / CHEMBL3804751
python src/check_targets.py

# 2. Compute exact pocket centroid from ligand 38C (CT325) in PDB 4P8K
python src/get_pocket_center.py

# 3. Classify covalent vs non-covalent molecules
python src/audit_covalent.py

# 4. Resolve hydantoin dominance with Butina cluster-aware partitioning
python src/analyze_hydantoins.py

# 5. Verify all 22 literature DOIs live against CrossRef
python src/verify_dois.py

# 6. Execute Gate 1 curation funnel
python src/gate1_funnel.py
```

---

## 5. Phase 1.5 Audit & Ground Truth Summary

### Gate 1 Funnel Metrics (Target: `CHEMBL3804751`)
- **Total Raw Bioactivities**: 306 records
- **Tier A Biochemical ($IC_{50}/K_i$)**: 159 records
- **Sanitized Unique Structures**: 147 compounds
- **Active ($pIC_{50} \ge 6.0$, $\le 1\,\mu\text{M}$)**: 74 compounds
- **Inactive ($pIC_{50} < 5.0$, $> 10\,\mu\text{M}$)**: 28 compounds
- **Gray Zone ($5.0 \le pIC_{50} < 6.0$)**: 45 compounds (held out)
- **Primary Labeled Benchmark Set**: **102 compounds**

### Covalent vs. Non-Covalent Docking Sample Size
- **Non-Covalent Evaluation Set**: **93 molecules** (67 Active, 26 Inactive) $\rightarrow$ **True primary docking sample size**.
- **Covalent Subset**: **9 molecules** (7 Active, 2 Inactive) $\rightarrow$ Evaluated in separate pre-reaction proximity track.

### Pocket Centroid Coordinates (PDB `4P8K`, Ligand `38C (CT325)`)
- **Center**: $X = 17.07$, $Y = -20.26$, $Z = 1.49$ Å
- **Box Size**: $22.0 \times 22.0 \times 22.0$ Å (`docking/configs/vina_4P8K.txt`)
- **Catalytic Proximity**: $5.7$ Å from Cys387 S$\gamma$, $4.2$ Å from FAD N5.

### Literature Audit Verification
- **Total Publications**: 22 papers across 3 buckets.
- **CrossRef Verification**: **22/22 DOIs verified with HTTP 200 OK** (`reports/DprE1_Literature_Audit_verified.csv`).
