# Data Card: DprE1 Benchmark Dataset

## 1. Dataset Description & Provenance

### 1.1 Target Specification
- **Target Protein**: Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1 / Rv3790)
- **Organism**: *Mycobacterium tuberculosis* H37Rv
- **UniProt Accession**: **`P9WJF1`** (461 amino acids)
- **ChEMBL Target ID**: **`CHEMBL3804751`** (Single Protein, confirmed sole target for *M. tuberculosis* DprE1)

### 1.2 Data Sources & Database Versions
- **ChEMBL Database**: Queried live via ChEMBL Web Services REST API.
  - **Database Version**: **`ChEMBL_37`**
  - **Release Date**: **2026-05-01**
  - **Status**: Live / Verified
- **RCSB Protein Data Bank (PDB)**: Queried live via RCSB PDB Search API v2 and Data API v1.
  - **Audit Snapshot**: October 2026
  - **Receptor Structure**: PDB **`4P8K`** (Non-covalent complex with ligand `38C (CT325)`, 2.25 Å resolution)

---

## 2. Filtering Funnel & Quantitative Audit Summary

The 306 bioactivity records for `CHEMBL3804751` were processed through the strict Gate 1 Funnel:

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

### Quantitative Metrics Summary

| Pipeline Stage | Compound Count | Percentage of Corpus | Notes |
| :--- | :--- | :--- | :--- |
| **Total Raw ChEMBL Records** | 306 | 100.0% | Complete verified biochemical corpus |
| **Tier A Bioactivities ($IC_{50}/K_i$)** | 159 | 52.0% | Assays with quantitative potency endpoints |
| **Sanitized Unique Structures** | 147 | 48.0% | Canonical SMILES after salt stripping |
| **Active Compounds ($pIC_{50} \ge 6.0$)** | 74 | 24.2% | $\le 1\,\mu\text{M}$ potency |
| **Inactive Compounds ($pIC_{50} < 5.0$)** | 28 | 9.2% | $> 10\,\mu\text{M}$ or relational `>` above $10\,\mu\text{M}$ |
| **Gray Zone Compounds ($5.0 \le pIC_{50} < 6.0$)** | 45 | 14.7% | Held out from primary classification |
| **Primary Labeled Benchmark Set** | **102** | 33.3% | Actives + Inactives |
| **Total Unique Murcko Scaffolds** | 45 | — | Bemis-Murcko frameworks |
| **Scaffolds with $\ge 3$ Compounds** | 6 | — | Governed by single hydantoin series |

---

## 3. Covalent vs. Non-Covalent Substructure Breakdown

Compounds were audited for covalent suicide-inhibitor warheads (nitro-aromatic core `c[N+](=O)[O-]` and 1,3-benzothiazin-4-one core `c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2`):

| Activity Category | Non-Covalent | Covalent | Total | Benchmark Track |
| :--- | :--- | :--- | :--- | :--- |
| **Active ($pIC_{50} \ge 6.0$)** | 67 | 7 | 74 | Labeled Set |
| **Inactive ($pIC_{50} < 5.0$)** | 26 | 2 | 28 | Labeled Set |
| **Primary Labeled Benchmark** | **93** | **9** | **102** | — |
| **Gray Zone ($5.0 \le pIC_{50} < 6.0$)** | 36 | 9 | 45 | Held-out Set |
| **Total Dataset** | **129** | **18** | **147** | Full Curated Corpus |

> [!IMPORTANT]
> **Primary Docking Benchmark Sample Size**: Exactly **93 non-covalent labeled molecules** (67 active, 26 inactive) are eligible for quantitative AutoDock Vina rigid docking. The 9 covalent suicide-inhibitors are evaluated in a separate pre-reaction proximity track.

---

## 4. Structural Receptor Catalog & Pocket Center

- **Selected Structure**: PDB `4P8K` (Chain A, 2.25–2.49 Å resolution)
- **Catalytic Cofactor**: FAD is retained in the active site with partial charges assigned at pH 7.4.
- **Reference Ligand**: `38C (CT325)` in Chain A (27 heavy atoms)
- **Calculated Centroid**:
  $$\mathbf{C} = (17.07, -20.26, 1.49)\,\text{Å}$$
- **Vina Search Box**: $22.0 \times 22.0 \times 22.0$ Å centered on `(17.07, -20.26, 1.49)` Å (`docking/configs/vina_4P8K.txt`).

---

## 5. Known Biases & Handling Strategies

1. **Hydantoin Chemotype Dominance**:
   - A single hydantoin series (`O=C(CN1C(=O)NC(c2ccccc2)C1=O)c1ccccc1`, Gao et al., 2018) comprises 34 of the 102 labeled compounds (33.3%).
   - *Mitigation*: Cluster-aware balanced partitioning (Butina clustering at 0.55 distance threshold) distributes clusters across 5 equal folds. A complementary Leave-Hydantoin-Out (LHO) evaluation explicitly tests out-of-distribution transfer.
2. **Publication Bias (Active Preponderance)**:
   - Experimental inactives ($N=28$) are outnumbered by actives ($N=74$) due to reporting bias in medicinal chemistry literature.
   - *Mitigation*: The primary benchmark evaluates models against these 28 measured inactives. Secondary Experiment B augments with property-matched deep learning decoys (DeepCoy / DUD-E).
3. **Covalent Incompatibility with Rigid Docking**:
   - Nitro-aromatics undergo an FAD-mediated redox cascade forming a covalent bond with Cys387.
   - *Mitigation*: Strict exclusion of covalent warheads from standard Vina scoring; separate proximity analysis.

---

## 6. Licensing & Data Use

- **ChEMBL Data**: Licensed under [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).
- **RCSB PDB Data**: Released under [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
- **Benchmark Code & Derived Artifacts**: Released under the Apache 2.0 License.
