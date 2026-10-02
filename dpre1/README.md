# DprE1 Benchmark Project: Structure-Based Docking vs. Ligand-Based Machine Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![RDKit](https://img.shields.io/badge/RDKit-2026.03.1-green.svg)](https://www.rdkit.org/)
[![License: CC BY-SA 3.0](https://img.shields.io/badge/Data_License-CC_BY--SA_3.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/3.0/)
[![Status: Phase 1 Complete](https://img.shields.io/badge/Benchmark_Status-Phase_1_Gate_Evaluated-orange.svg)]()

> **Research Question**: *Does Structure-Based Docking Add Predictive Value Over Ligand-Based Machine Learning for Prioritizing DprE1 Inhibitors? A Reproducible Benchmark Under Scaffold-Split Evaluation.*

---

## 1. Project Overview

Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (**DprE1** / **Rv3790**) is a clinically validated flavoenzyme essential for cell wall arabinogalactan biosynthesis in *Mycobacterium tuberculosis*. While numerous computational studies apply molecular docking or ligand-based QSAR to DprE1, existing literature suffers from two pervasive methodological biases:
1. **Scaffold Leakage**: Evaluating machine learning models via random train/test splits rather than out-of-scaffold splits, concealing catastrophic generalization failures.
2. **Scoring Confounding**: Relying on uncalibrated docking scores without controlling for molecular weight bias or comparing against trivial 2D nearest-neighbor baselines.

This benchmark project establishes an open-source, mathematically leak-free evaluation pipeline to compare **Structure-Based Docking (AutoDock Vina)** against **Ligand-Based Machine Learning (Random Forest, LightGBM, Logistic Regression)** under rigorous **Bemis-Murcko scaffold-split cross-validation**.

---

## 2. Repository Architecture

```text
dpre1/
├── app/                           # Interactive visualization & Streamlit exploration app
├── data/
│   ├── raw/
│   │   ├── chembl_dpre1_raw.csv         # Live extracted bioactivity records for CHEMBL5542 (21,721 records)
│   │   ├── chembl_dpre1_rv3790_raw.csv  # Live extracted records for M. tuberculosis DprE1 (306 records)
│   │   └── pdb_audit.csv                # RCSB PDB structural catalog (38 crystal structures)
│   ├── interim/                         # Filtered intermediate subsets
│   └── processed/
│       ├── gate1_compounds.csv          # Canonicalized, curated compound benchmark set
│       └── gate1_compounds_chembl5542.csv
├── docking/
│   ├── configs/                         # Vina search box and exhaustiveness configurations
│   ├── ligands/                         # Prepared ligand PDBQT files
│   ├── outputs/                         # Docking poses and log files
│   └── receptors/                       # Prepared receptor PDBQT files (PDB 4P8K with intact FAD)
├── notebooks/                     # Exploratory analysis and figure generation notebooks
├── reports/
│   ├── figures/                         # High-resolution benchmark figures
│   ├── DprE1_Literature_Audit.csv       # 22 curated papers across 3 methodological buckets
│   └── gate1_funnel_summary.json        # Quantitative Gate 1 funnel metrics
├── src/
│   ├── audit_chembl.py                  # Live ChEMBL bioactivity extractor & statistics
│   ├── audit_pdb.py                     # RCSB PDB API structural auditor
│   └── gate1_funnel.py                  # Gate 1 RDKit curation, Murcko scaffold & funnel engine
├── DATA_CARD.md                   # Full data provenance, ChEMBL version, and bias documentation
├── environment.yml                # Conda/micromamba environment specification (Python 3.10)
├── PROTOCOL.md                    # Frozen experimental protocol & evaluation metrics
└── README.md                      # Project documentation and quickstart guide
```

---

## 3. Environment Installation & Activation

The project environment is CPU-compatible with zero GPU dependencies. It pins Python 3.10, RDKit, scikit-learn, AutoDock Vina, Meeko, ProLIF, OpenBabel, pandas, and Streamlit.

### Setup via Micromamba / Conda
```bash
# Clone repository and enter project root
cd minor_project/dpre1

# Create the pinned environment
micromamba create -y -n dpre1 -f environment.yml

# Activate the environment
micromamba activate dpre1
```

---

## 4. Replication Instructions (Phase 1)

Execute the end-to-end data audit and Gate 1 evaluation pipeline with live public API queries:

### 1. PDB Structural Catalog Audit
Queries the RCSB PDB Search and Data APIs for all DprE1 crystal structures, cataloging resolution, release dates, co-crystallized ligands, FAD presence, and covalent Cys387 status:
```bash
python src/audit_pdb.py
```
*Output*: Saved to `data/raw/pdb_audit.csv`.

### 2. Live ChEMBL Bioactivity Extraction
Extracts all bioactivity records for the mandated target `CHEMBL5542` and canonical DprE1 target `CHEMBL3804751`, parsing Tier A (biochemical $IC_{50}/K_i$) and Tier B (phenotypic MIC) data:
```bash
python src/audit_chembl.py
```
*Output*: Saved to `data/raw/chembl_dpre1_raw.csv` and `data/raw/chembl_dpre1_rv3790_raw.csv`.

### 3. Gate 1 Funnel Computation
Sanitizes chemical structures via RDKit, strips salts, standardizes units to nM, computes $pIC_{50} = 9 - \log_{10}(\text{nM})$, handles relational operators (`=`, `>`, `<`), generates Bemis-Murcko scaffolds, and evaluates the scaffold diversity threshold:
```bash
python src/gate1_funnel.py
```
*Output*: Formatted terminal funnel table, `data/processed/gate1_compounds.csv`, and `reports/gate1_funnel_summary.json`.

---

## 5. Phase 1 Audit & Gate 1 Results Summary

### Quantitative Funnel Table

| Stage / Metric | Mandated (`CHEMBL5542`) | Canonical DprE1 (`CHEMBL3804751`) | Threshold / Criterion |
| :--- | :--- | :--- | :--- |
| **Raw Bioactivities Extracted** | 21,721 | 306 | Live query |
| **Tier A Biochemical Assays** | 50 | 159 | $IC_{50}$ or $K_i$ |
| **Sanitized Unique Structures** | 48 | 147 | RDKit canonicalized |
| **Active Compounds ($pIC_{50} \ge 6.0$)** | 2 | 74 | $\le 1\,\mu\text{M}$ |
| **Inactive Compounds ($pIC_{50} < 5.0$)** | 40 | 28 | $> 10\,\mu\text{M}$ |
| **Gray Zone Compounds ($5.0 \le pIC_{50} < 6.0$)** | 6 | 45 | Excluded |
| **Labeled Compounds (Outside Gray)** | 42 | 102 | Usable benchmark set |
| **Total Unique Murcko Scaffolds** | 16 | 45 | Requirement: $\ge 30$ |
| **Scaffolds with $\ge 3$ Compounds** | **2** | **6** | Requirement: $\ge 10$ |
| **GATE 1 VERDICT** | **FAIL / PIVOT** | **FAIL / PIVOT** | Pivot to Combined Tiers |

### Key Biological Findings
- **Cofactor Integrity**: 33 of 38 cataloged PDB structures retain the essential FAD cofactor. PDB **`4P8K`** (CT325 complex, 2.25–2.49 Å) was selected as the frozen reference non-covalent receptor.
- **Covalent Nitro-Inhibitor Rule**: Covalent adducts (e.g., BTZ043 in `4FF6`, PBTZ169 in `4NCR`) involve FAD-mediated nitroreduction followed by nucleophilic attack on Cys387; they are excluded from standard rigid Vina runs.
- **Target Discrepancy Clarified**: In ChEMBL, `CHEMBL5542` corresponds to human DNA Polymerase Eta, while `CHEMBL3804751` is the true *M. tuberculosis* DprE1 target. Both were audited and evaluated through Gate 1. In both cases, Gate 1 triggered **FAIL / PIVOT** due to an insufficient count of scaffolds containing $\ge 3$ congeneric compounds.

---

## 6. Phase 2 Roadmap & Next Steps

Following the **FAIL / PIVOT** verdict at Gate 1, Phase 2 implements the protocol-defined pivot:
1. **Tier Merging**: Combine Tier A biochemical assays with Tier B whole-cell phenotypic assays (*M. tuberculosis* H37Rv MIC) to reach sufficient scaffold breadth.
2. **Decoy Library Generation**: Generate property-matched decoys (matching MW, SlogP, HBD, HBA, rotatable bonds) to overcome severe publication bias (deficit of true inactives).
3. **Scaffold-Split Benchmark**: Execute 5-fold scaffold-split cross-validation across 5 random seeds (`[42, 123, 456, 789, 1011]`), evaluating PR-AUC, ROC-AUC, and $EF_{1\%}$ for LB-ML, Vina Docking, and Hybrid models.
