# Data Card: DprE1 Benchmark Dataset

## 1. Dataset Description & Provenance

### 1.1 Dataset Identification
- **Name**: DprE1 Inhibitor Virtual Screening Benchmark Dataset (Phase 1 Audit)
- **Target Protein**: Decaprenylphosphoryl-$\beta$-D-ribose $2'$-epimerase (DprE1 / Rv3790)
- **Organism**: *Mycobacterium tuberculosis* H37Rv
- **Data Sources**:
  - **ChEMBL Database**: Queried live via ChEMBL Web Services (REST API v2.0) and `chembl-webresource-client` (v0.10.9). Targets audited:
    - Primary Mandated Target: `CHEMBL5542`
    - Reference DprE1 Target: `CHEMBL3804751`
  - **RCSB Protein Data Bank (PDB)**: Queried live via RCSB PDB Search API v2 and Data API v1. Target accessions: UniProt `P9WJF1` / `P9WGI1` and text queries "DprE1", "Rv3790".
- **Audit Date**: October 2, 2026
- **Database Versions**: ChEMBL 34 (REST API 2026 snapshot); RCSB PDB (October 2026 release).

---

## 2. Filtering Funnel & Quantitative Audit Summary

The raw bioactivity records were processed through the strict Phase 1 Gate 1 Funnel:

```
[Raw Records Extracted]
       │
       ▼
[Assay Tier Separation] ────► Tier B (Phenotypic whole-cell MIC)
       │
       ▼
[Tier A: Biochemical Binding (IC50 / Ki)]
       │
       ▼
[RDKit Sanitization & Salt Stripping (Largest Organic Fragment)]
       │
       ▼
[Unit Standardization (nM) & pIC50 Calculation: 9 - log10(nM)]
       │
       ▼
[Activity Categorization]
 ├── Active: pIC50 >= 6.0 (<= 1 uM)
 ├── Inactive: pIC50 < 5.0 (> 10 uM or relation '>' >= 10 uM)
 └── Gray Zone: 5.0 <= pIC50 < 6.0 (Held out)
       │
       ▼
[Bemis-Murcko Scaffold Extraction (No Chirality)]
       │
       ▼
[Gate 1 Evaluation: Unique Scaffolds >= 30 AND Scaffolds with >= 3 Compounds >= 10]
```

### Quantitative Metrics Table

| Pipeline Stage | Mandated Audit (`CHEMBL5542`) | True DprE1 Audit (`CHEMBL3804751`) |
| :--- | :--- | :--- |
| **Total Raw Bioactivity Records** | 21,721 | 306 |
| **Unique Canonical SMILES (Raw)** | 19,091 | 195 |
| **Assay Type B (Biochemical)** | 61 | 306 |
| **Assay Type F (Whole-cell Phenotypic)** | 21,660 | 0 |
| **Tier A Bioactivities ($IC_{50} / K_i$)** | 50 | 159 |
| **Sanitized Unique Structures (Tier A)** | 48 | 147 |
| **Active Compounds ($pIC_{50} \ge 6.0$)** | 2 | 74 |
| **Inactive Compounds ($pIC_{50} < 5.0$)** | 40 | 28 |
| **Gray Zone Compounds ($5.0 \le pIC_{50} < 6.0$)**| 6 | 45 |
| **Total Labeled Compounds (Outside Gray)** | 42 | 102 |
| **Total Unique Murcko Scaffolds** | 16 | 45 |
| **Scaffolds with $\ge 3$ Compounds** | **2** | **6** |
| **Gate 1 Evaluation Verdict** | **FAIL / PIVOT** | **FAIL / PIVOT** |

---

## 3. Structural Catalog & PDB Provenance

A total of 38 crystal structures were cataloged via the RCSB PDB Data API (`data/raw/pdb_audit.csv`):
- **Resolution Range**: 1.128 Å to 3.000 Å (Median: 2.30 Å).
- **FAD Cofactor Presence**:
  - Present: 33 structures (including all primary DprE1 co-complexes).
  - Absent: 5 structures (apo/solvent-perturbed entries or off-target control `6HES`).
- **Mechanism Breakdown**:
  - Covalent adducts at Cys387: 9 structures (e.g., `4FF6` with BTZ reduced adduct, `4NCR` with PBTZ169 bound adduct).
  - Non-covalent complexes: 28 structures (e.g., `4P8K`, `4P8L`, `4P8N`, `6HEZ`, `5OEL`, `6HF0`).
  - Apo structure: 1 structure (`4AUT`).

### Reference Benchmark Structures
- **Primary Docking Receptor**: PDB **`4P8K`** (Chain A, 2.25–2.49 Å resolution). Co-crystallized with non-covalent pyrrole/quinoxaline inhibitor `38C`. Active-site FAD retained.
- **Secondary Receptor**: PDB **`6HEZ`** (2.00–2.30 Å resolution). Co-crystallized with 1,4-azaindole `0SK` (TBA-7371). Active-site FAD retained.

---

## 4. Known Data Biases & Limitations

1. **Publication Bias (Low Measured Inactives)**:
   - In standard biochemical assays from ChEMBL (`CHEMBL3804751`), 72.5% of evaluated compounds are active ($pIC_{50} \ge 6.0$), with only 28 confirmed inactives.
   - Medicinal chemistry literature systematically fails to publish inactive analogs tested during lead optimization, creating an artificially high active prevalence.
2. **Covalent Nitro-Adduct Confounding**:
   - The most potent known inhibitors of DprE1 (BTZ043, PBTZ169) rely on FAD-mediated reduction to form a covalent adduct with Cys387.
   - Standard molecular docking engines (AutoDock Vina) compute non-covalent van der Waals, electrostatic, and desolvation energies; they are unable to score the free energy of covalent bond formation without customized covalent docking workflows.
3. **Assay Format Discrepancies**:
   - Biochemical enzyme assays measure isolated recombinant DprE1 inhibition using artificial electron acceptors (e.g., resazurin or FPR fluorescence).
   - Whole-cell phenotypic assays (Tier B, MIC) require penetration across the highly lipophilic, mycolic-acid-rich mycobacterial cell envelope and are subject to efflux pumps (e.g., Rv3723/MmpL5).
4. **Target Annotation Ambiguity**:
   - In ChEMBL, `CHEMBL5542` corresponds to human DNA Polymerase Eta (`POLH`), while `CHEMBL3804751` represents *M. tuberculosis* DprE1. Both datasets have been audited and frozen side-by-side to guarantee scientific integrity and complete reproducibility.

---

## 5. Licensing & Data Use

- **ChEMBL Data**: Licensed under the [Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/).
- **RCSB PDB Data**: Released under [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
- **Benchmark Code & Derived Artifacts**: Released for open scientific research under the Apache 2.0 License.
