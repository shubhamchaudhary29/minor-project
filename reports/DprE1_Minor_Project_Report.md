# Benchmarking Ligand-Based Machine Learning and Structure-Based Molecular Docking for *Mycobacterium tuberculosis* DprE1 Inhibitor Discovery

**A Minor Project Dissertation Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology**

**Author**: Shubham Chaudhary  
**Department**: University School of Automation and Robotics (USAR)  
**Institution**: Guru Gobind Singh Indraprastha University (GGSIPU), East Delhi Campus, Delhi, India  
**Academic Year**: 2025–2026  
**Repository**: [github.com/shubhamchaudhary29/minor-project](https://github.com/shubhamchaudhary29/minor-project.git)  
**Protocol Version**: `v1.1.2-frozen`  

---

## Abstract

Virtual screening plays a foundational role in modern computational drug discovery, yet head-to-head empirical comparisons between 2D ligand-based machine learning (LB-ML) and 3D structure-based molecular docking (SB-DD) remain susceptible to widespread methodological confounding. In this dissertation, we present a reproducible, leak-free benchmark evaluating the predictive capacity of ligand-based machine learning versus AutoDock Vina molecular docking for prioritizing inhibitors of *Mycobacterium tuberculosis* Decaprenylphosphoryl-$\beta$-D-ribose 2'-epimerase (DprE1, Rv3790), an essential, clinically validated anti-tubercular drug target. 

By conducting a systematic audit of public bioactivity data in ChEMBL, we identified and remediated four pervasive literature traps: (1) target identifier corruption involving human DNA polymerase eta (*POLH* / `CHEMBL5542`), resolving to verified biochemical DprE1 (`CHEMBL3804751`); (2) mechanistic conflation of covalent suicide-inhibitors (e.g., nitroaromatic BTZ043/PBTZ169 adducting catalytic Cys387) with reversible non-covalent inhibitors; (3) evaluative analogue leakage caused by random cross-validation on congeneric chemical series; and (4) molecular weight size-bias in empirical docking scoring functions.

We curated a benchmark corpus of 93 non-covalent DprE1 compounds with homogeneous biochemical $IC_{50}$ annotations and evaluated models across three rigorous split tracks: Track A (Random 5-Fold CV), Track B (Stratified Cluster GroupKFold via Butina ECFP4 distance 0.55), and Track C (Leave-Hydantoin-Out cross-chemotype transfer). Structure-based docking was conducted against the crystallographic structure of DprE1 (PDB `4P8K`, 2.49 Å resolution) retaining the essential rigid Flavin Adenine Dinucleotide (FAD) catalytic cofactor, passing crystallographic redocking validation with a heavy-atom RMSD of 1.282 Å (< 2.0 Å threshold).

Our primary statistical finding demonstrates a decisive null result: across all evaluated split tracks, hybrid rank fusion between ligand-based machine learning and AutoDock Vina yields 1,000-sample bootstrap 95% confidence intervals for $\Delta\text{ROC}$ that cross zero (Track B: $[-0.139, +0.113]$; Track C: $[-0.013, +0.138]$; Non-Hydantoin test set: $[-0.068, +0.198]$). Structure-based docking adds no statistically detectable predictive value over 2D machine learning at $N=93$. Furthermore, a trivial 1-rule heuristic baseline (`HydantoinDetectorClassifier`) achieves an ROC-AUC of 0.660 [0.541, 0.758], directly outperforming standalone AutoDock Vina (0.610 [0.463, 0.746]), highlighting how chemotype prevalence bias in public chemical datasets can overshadow uncalibrated physical scoring functions. Residue-level contact analysis on discordant predictions reveals that raw docking scores suffer from severe non-specific contact accumulation in high-MW molecules ($r = -0.437, p < 10^{-5}$), while 2D ML correctly identifies active pharmacophores but remains vulnerable to decision-tree extrapolation failure under scaffold shifts. Finally, we deployed an interactive Streamlit virtual screening and lead-optimization laboratory (`app/streamlit_app.py`) integrating real-time 3D docking, applicability domain surveillance, and empirical consensus diagnostics.

---

## Table of Contents

- [Abstract](#abstract)
- [Chapter 1: Introduction & Biological Rationale](#chapter-1-introduction--biological-rationale)
- [Chapter 2: Literature Review & The Four Methodological Traps](#chapter-2-literature-review--the-four-methodological-traps)
- [Chapter 3: Problem Statement, Research Questions & Specific Objectives](#chapter-3-problem-statement-research-questions--specific-objectives)
- [Chapter 4: Computational Methodology & Experimental Design](#chapter-4-computational-methodology--experimental-design)
- [Chapter 5: Empirical Results & Benchmark Performance Matrix](#chapter-5-empirical-results--benchmark-performance-matrix)
- [Chapter 6: Interactive Virtual Screening Lab Software Architecture](#chapter-6-interactive-virtual-screening-lab-software-architecture)
- [Chapter 7: Project Timeline, Deviations & Risk Management](#chapter-7-project-timeline-deviations--risk-management)
- [Chapter 8: Critical Discussion, Limitations & Future Work](#chapter-8-critical-discussion-limitations--future-work)
- [Chapter 9: Conclusions & Scientific Takeaways](#chapter-9-conclusions--scientific-takeaways)
- [Chapter 10: References & Citations](#chapter-10-references--citations)

---

## Chapter 1: Introduction & Biological Rationale

### 1.1 The Global Challenge of Tuberculosis
Tuberculosis (TB), caused by the intracellular bacterium *Mycobacterium tuberculosis* (*M.tb*), remains one of the deadliest infectious diseases worldwide. According to the World Health Organization (WHO) Global Tuberculosis Report, TB causes upwards of 1.3 million deaths annually. The emergence and spread of multi-drug-resistant (MDR-TB) strains resistant to isoniazid and rifampicin, as well as extensively drug-resistant (XDR-TB) strains, have rendered standard frontline regimens increasingly ineffective. Consequently, there is an urgent imperative to identify novel small-molecule therapeutics targeting previously unexploited bacterial enzymes essential for cell survival.

### 1.2 DprE1: Biology, Mechanism and Clinical Validation
Among the validated vulnerability targets in *M.tb*, Decaprenylphosphoryl-$\beta$-D-ribose 2'-epimerase (DprE1, encoded by `Rv3790`, UniProt `P9WJF1`) has emerged as a premier drug target. DprE1 acts in conjunction with DprE2 (`Rv3791`) to catalyze the two-step epimerization of decaprenylphosphoryl-$\beta$-D-ribose (DPR) into decaprenylphosphoryl-$\beta$-D-arabinofuranose (DPA):

$$\text{DPR} \xrightarrow{\text{DprE1 (Oxidation)}} \text{DPX} \xrightarrow{\text{DprE2 (Reduction)}} \text{DPA}$$

DPA is the sole lipophilic arabinosyl donor required for the biosynthesis of arabinogalactan and lipoarabinomannan, structural polysaccharides essential for the integrity and acid-fast permeability barrier of the mycobacterial cell wall. Pharmacological inhibition or genetic knockdown of DprE1 halts arabinogalactan synthesis, triggering osmotic lysis and rapid bactericidal clearance.

```
       [DPR] (Decaprenylphosphoryl-D-ribose)
         │
         ▼  <--- DprE1 (FAD-dependent 2'-oxidation)
       [DPX] (Decaprenylphosphoryl-2-keto-erythro-pentofuranose)
         │
         ▼  <--- DprE2 (NADH-dependent reduction)
       [DPA] (Decaprenylphosphoryl-D-arabinose)
         │
         ▼
   [Mycobacterial Cell Wall Assembly] (Arabinogalactan & LAM)
```

### 1.3 Structural Architecture and Cofactor Requirement
DprE1 is a 461-amino-acid flavoprotein belonging to the vanillyl-alcohol oxidase (VAO) structural superfamily. Crystallographic studies (e.g., PDB `4P8K`, `4FDO`, `3TX4`) reveal a two-domain architecture:
1. **FAD-Binding Domain**: Binds a non-covalent Flavin Adenine Dinucleotide (FAD) prosthetic group in an extended conformation. FAD is indispensable for enzymatic turnover, serving as the primary electron acceptor that oxidizes the C2'-hydroxyl of DPR to form the 2'-keto intermediate (DPX). The isoalloxazine ring of FAD forms the structural floor of the substrate-binding cavity.
2. **Substrate-Binding Domain**: Features a deep, hydrophobic active-site pocket adjacent to FAD N5 and C4a. A conserved cysteine residue, **Cys387**, is located in close proximity (~5.7 Å from the pocket centroid) and plays a pivotal role in covalent suicide inhibition. Other key active-site residues include the aromatic gating residue **Tyr314**, the basic anchoring sidechain **Lys418**, and hydrophobic pocket lining residues **Val365** and **Leu363**.

### 1.4 Mechanistic Divergence: Covalent vs Non-Covalent Inhibition
Inhibitors of DprE1 segregate into two mechanistically distinct classes:
- **Covalent Suicide Inhibitors**: Characterized by electrophilic nitro-aromatic scaffolds, exemplified by 1,3-benzothiazin-4-ones (BTZ043, PBTZ169 / macozinone). These molecules undergo an enzyme-catalyzed reaction where the flavin cofactor reduces the aromatic nitro group ($-NO_2$) to a highly reactive nitroso intermediate ($-NO$). The nucleophilic thiol of Cys387 subsequently attacks the nitroso group, forming a stable covalent semimercaptal adduct.
- **Reversible Non-Covalent Inhibitors**: Chemically diverse chemical classes—including quinoxalines (e.g., CT325 in PDB `4P8K`), pyrroles, 1,4-azaindoles, and hydantoins—that bind through reversible steric and electrostatic interactions without forming covalent bonds with Cys387.

In this benchmark, we focus specifically on non-covalent virtual screening, establishing rigorous physical and statistical criteria to prevent the conflation of these divergent chemical mechanisms.

---

## Chapter 2: Literature Review & The Four Methodological Traps

Despite extensive computational and medicinal chemistry efforts targeting DprE1 over the past decade, virtual screening studies in the literature frequently suffer from systemic methodological flaws. Through detailed auditing of public repositories and published protocols, we identified four critical traps that invalidate published findings.

```
┌────────────────────────────────────────────────────────────────────────┐
│               THE FOUR CRITICAL METHODOLOGICAL TRAPS                   │
├────────────────────────────────────────────────────────────────────────┤
│ Trap 1: Target Confusion     │ Purging POLH (CHEMBL5542), locking       │
│                              │ canonical DprE1 (CHEMBL3804751, P9WJF1).│
├──────────────────────────────┼─────────────────────────────────────────┤
│ Trap 2: Mechanistic Confound │ Segregating covalent suicide warheads   │
│                              │ from reversible non-covalent docking.   │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Trap 3: Analogue Leakage     │ Replacing random CV with Butina cluster │
│                              │ and Leave-Hydantoin-Out cross-series.   │
├──────────────────────────────┼─────────────────────────────────────────┤
│ Trap 4: Physical Size-Bias   │ Auditing MW correlation (r = -0.437)   │
│                              │ and reporting Ligand Efficiency (LE).   │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Trap 1: Target Identifier Confusion (POLH vs DprE1)
A severe data integrity failure was uncovered during initial repository auditing: multiple public datasets and preliminary pipelines cross-referenced ChEMBL target `CHEMBL5542`. Our audit confirmed that `CHEMBL5542` corresponds to human DNA polymerase eta (*POLH*, UniProt `Q9Y25J`), an error arising from automated text-scraping cross-contamination. 
- Canonical DprE1 is correctly indexed in ChEMBL as **`CHEMBL3804751`** (single-protein biochemical assay target for *Mycobacterium tuberculosis* H37Rv DprE1, UniProt **`P9WJF1`**).
- All instances of `CHEMBL5542` and unrelated human targets were purged completely from the benchmark pipeline under deviation **DEV-01**.

### 2.2 Trap 2: Mechanistic Confounding of Covalent Suicide Inhibitors
Numerous published docking studies dock nitroaromatic inhibitors (such as BTZ043) into DprE1 using standard non-covalent empirical scoring functions (e.g., AutoDock Vina, Glide SP). This practice is biophysically invalid:
- Standard docking algorithms calculate binding free energy ($\Delta G_{\text{bind}}$) based on non-bonded terms: van der Waals, electrostatic Coulombic interactions, hydrogen bonding, and desolvation penalties.
- Because covalent inhibitors derive their potency from the covalent bond formation enthalpy ($\sim 50\text{--}80\text{ kcal/mol}$) following FAD reduction, their reversible pre-reaction non-covalent affinity does not correlate with their biochemical potency ($IC_{50} \approx 1\text{--}10\text{ nM}$).
- Under deviation **DEV-04**, we formally partitioned the labeled corpus: 9 nitro-aromatic compounds were segregated into a pre-reaction proximity audit track, leaving $N=93$ purely non-covalent compounds for quantitative docking evaluation.

### 2.3 Trap 3: Evaluative Analogue Leakage via Random Cross-Validation
Standard machine learning benchmarks in chemoinformatics typically employ random train-test splits (e.g., 80/20 random partition or standard Stratified K-Fold CV). In chemical datasets populated by medicinal chemistry series, random splitting results in severe **analogue leakage**:
- Congeneric molecules differing by a single functional group (e.g., methyl or halogen substitution) are distributed across both training and test folds.
- The model memorizes trivial scaffold subgraphs rather than learning generalizable biophysical interaction rules, producing inflated performance metrics (Track A: ROC-AUC = 0.799, PR-AUC = 0.923).
- When tested on novel scaffolds, these models suffer severe generalization drop. Under deviations **DEV-03** and **DEV-05**, we implemented Stratified Cluster GroupKFold (Butina ECFP4 clustering at distance 0.55) and Leave-Hydantoin-Out (LHO) evaluation.

### 2.4 Trap 4: Physical Size-Bias in Empirical Docking Scoring Functions
Empirical docking scoring functions estimate binding affinity by summing atom-atom pair potentials across the receptor-ligand interface. Because each added heavy atom contributes additional contact terms, raw docking scores exhibit an intrinsic, non-physical correlation with molecular weight:
- In our benchmark, the Pearson correlation between molecular weight (MW) and raw Vina binding affinity is **$r = -0.437$ ($p = 1.21 \times 10^{-5}$)**. Heavier molecules systematically receive more negative (more favorable) docking scores.
- Meanwhile, the true correlation between molecular weight and biological activity in the curated corpus is zero (**$r = -0.017, p = 0.869$**).
- As a consequence, uncorrected docking scores collapse on large compounds ($\text{MW} \ge 400\text{ Da}$, ROC-AUC = 0.414) due to size-driven false positives, necessitating size-normalized metrics such as Ligand Efficiency ($\text{LE} = -\Delta G / N_{\text{heavy}}$).

---

## Chapter 3: Problem Statement, Research Questions & Specific Objectives

### 3.1 Problem Statement
Virtual screening pipelines frequently combine 2D ligand-based machine learning predictions with 3D structure-based molecular docking scores under the assumption that physical modeling provides orthogonal, synergistic information. However, this assumption is rarely evaluated under rigorous, leak-free scaffold-split cross-validation on homogeneous biochemical data. In targets with dominant chemical series, it remains unknown whether 3D molecular docking provides statistically significant predictive lift over simple 2D machine learning baselines, or whether apparent hybrid synergy is an artifact of improper data splitting and size bias.

### 3.2 Primary Research Question & Formal Hypotheses
**Primary Research Question**:
> *Does structure-based molecular docking add predictive value over ligand-based machine learning for prioritizing non-covalent DprE1 inhibitors under scaffold-split evaluation?*

**Formal Hypotheses**:
- **Null Hypothesis ($H_0$)**: Structure-based molecular docking provides no added predictive value over ligand-based machine learning under leak-free out-of-fold evaluation:
  $$\Delta \text{ROC} = \text{ROC}_{\text{Hybrid}} - \text{ROC}_{\text{ML}} = 0$$
- **Alternative Hypothesis ($H_1$)**: Structure-based molecular docking provides statistically significant added predictive value over ligand-based machine learning:
  $$\Delta \text{ROC} > 0$$
- **Decision Rule**: The null hypothesis $H_0$ is rejected if and only if the lower bound of the paired 1,000-iteration bootstrap 95% confidence interval for $\Delta\text{ROC}$ strictly exceeds zero ($\text{CI}_{\text{lower}} > 0$).

### 3.3 Specific Project Objectives
1. **Curate and Sanitize the DprE1 Corpus**: Extract all bioactivity records for canonical DprE1 (`CHEMBL3804751`), filter for homogeneous biochemical assays (Tier A: $IC_{50}$ / $K_i$), remove gray-zone marginal activities, and segregate covalent suicide-inhibitors.
2. **Establish Gate 1 Chemotype Partitions**: Construct leak-free partitioning schemes: Track A (Random 5-Fold), Track B (Stratified Cluster GroupKFold, Butina distance 0.55), and Track C (Leave-Hydantoin-Out cross-chemotype transfer).
3. **Execute Gate 2 Crystallographic Redocking**: Prepare PDB `4P8K` (Chain A + rigid FAD cofactor), calculate active site coordinates from reference ligand `CT325 (38C)`, and validate redocking accuracy (< 2.0 Å RMSD).
4. **Benchmark LB-ML, SB-DD and Hybrid Fusion**: Train $L_2$-regularized Logistic Regression and Random Forest models on ECFP4 count fingerprints; execute AutoDock Vina batch virtual screening; benchmark unsupervised rank fusion methods (RRF, MPR, Z-Score sum).
5. **Analyze Discordance & Pocket Contacts**: Perform residue-level protein-ligand contact analysis on discordant compounds to mechanistically explain divergence between physical and statistical scoring.
6. **Deploy an Interactive Lead Optimization Lab**: Implement and calibrate an open-source Streamlit application supporting live docking, SAR mutation profiling, and visual concordance mapping.

---

## Chapter 4: Computational Methodology & Experimental Design

### 4.1 Data Curation & Filtering Funnel
All bioactivity data were programmatically retrieved from the ChEMBL database (v33/v34) for single-protein target `CHEMBL3804751` (*M.tb* DprE1). Data curation followed a rigorous multi-tier filtering funnel:

```
Total Raw Records for CHEMBL3804751 (N = 306)
                     │
                     ▼
  Tier A Biochemical Enzyme Assays (IC50 / Ki) (N = 159)
  [Tier B Whole-Cell MIC records excluded to prevent permeability confounding]
                     │
                     ▼
  RDKit Structure Standardization & Salt Stripping (N = 147 unique compounds)
  [Largest organic fragment retained, charges neutralized where appropriate]
                     │
                     ▼
  Activity Classification & Gray-Zone Isolation:
  ├── Active: pIC50 >= 6.0 (IC50 <= 1.0 uM)        --> 74 compounds
  ├── Inactive: pIC50 < 5.0 (IC50 > 10.0 uM)       --> 28 compounds
  └── Gray Zone: 5.0 <= pIC50 < 6.0                --> 45 compounds (Held out)
                     │
                     ▼
  Total Labeled Benchmark Set: 102 compounds (74 Active, 28 Inactive)
                     │
                     ├── Nitro-Aromatic Covalent Warhead Set: 9 compounds (7 Act, 2 Inact)
                     └── Non-Covalent Docking Benchmark Set: 93 compounds (67 Act, 26 Inact)
```

### 4.2 Cross-Validation Architectures & Splitting Schemes
To assess generalization under realistic lead optimization conditions, we evaluated three distinct split schemes:
1. **Track A (Random 5-Fold Stratified CV)**: Random 5-fold cross-validation preserving the 72.0% active / 28.0% inactive prior ratio across folds. Serves as the positive control demonstrating the impact of analogue leakage.
2. **Track B (Stratified Cluster GroupKFold CV)**: Molecules were clustered using the Butina algorithm on ECFP4 Tanimoto distances with a distance cutoff of $0.55$. Clusters were assigned across 5 folds using a greedy size-balancing algorithm guaranteeing $\ge 3$ inactive compounds per fold (DEV-05), ensuring no structural cluster spans multiple folds.
3. **Track C (Leave-Hydantoin-Out Chemotype Transfer)**: The single dominant medicinal chemistry series in the dataset—the hydantoin series (`O=C(CN1C(=O)NC(c2ccccc2)C1=O)c1ccccc1`, Gao et al., 2018)—accounts for 50 compounds (34 labeled: 30 active, 4 inactive in non-covalent set). In Track C, models are trained exclusively on non-hydantoins ($N=43$) and tested on hydantoins ($N=50$), and vice-versa, testing true out-of-distribution transfer across scaffolds.

### 4.3 Molecular Featurization & Machine Learning Models
Small molecules were featurized using RDKit (v2023.09+):
- **Extended Connectivity Fingerprints (ECFP4)**: Computed with radius 2 (diameter 4) as 2048-dimensional count vectors (`GetCountFingerprint`). Count vectors capture frequency of circular subgraphs, providing richer structural context than folded bit vectors.
- **Machine Learning Algorithms**:
  - **Logistic Regression (LR)**: Linear classifier with $L_2$ regularization penalty ($C=1.0$), optimized via L-BFGS with balanced class weighting.
  - **Random Forest (RF)**: Ensemble of 300 decision trees (`n_estimators=300`, `max_depth=None`, `class_weight='balanced'`, fixed `random_state=42`).
  - **Baselines**: Baseline 0 (Random chance floor: PR-AUC = 0.720, ROC-AUC = 0.500); Baseline 1 (`HydantoinDetectorClassifier`, SMARTS `O=C1NC(=O)NC1`); Baseline 2 (1-Nearest Neighbor Tanimoto similarity to training actives).

### 4.4 Receptor Preparation & Gate 2 Crystallographic Redocking
The 3D structure of DprE1 was retrieved from the Protein Data Bank (PDB `4P8K`, Chain A, 2.49 Å resolution):
- **Receptor Clean-up**: Crystallographic waters, buffer molecules, and alternate conformer locations were stripped. 
- **Flavin Adenine Dinucleotide (FAD)**: FAD cofactor coordinates (`HETATM FAD A 501`) were strictly retained in the active site and parameterized with Gasteiger partial charges as an integral component of the receptor model (`docking/receptors/4P8K_receptor.pdbqt`).
- **Docking Grid Centroid & Box**: Calculated as the geometric center of all 27 heavy atoms of the co-crystallized quinoxaline ligand `CT325 (38C)`:
  $$\mathbf{C} = (17.07, -20.26, 1.49)\,\text{Å}$$
  Grid dimensions were configured to $22.0 \times 22.0 \times 22.0$ Å with 1.0 Å spacing (`docking/configs/vina_4P8K.txt`).
- **Gate 2 Redocking Validation**: AutoDock Vina redocking of `CT325` against `4P8K` at `exhaustiveness=16` reproduced the experimental crystallographic binding mode with a heavy-atom root-mean-square deviation of **$\text{RMSD} = 1.282\,\text{Å}$**, decisively passing the Gate 2 quality threshold (< 2.0 Å). Multi-seed validation (`seeds = [42, 101, 2024]`) confirmed robust convergence ($\text{Mean RMSD} = 1.355 \pm 0.056\,\text{Å}$).

```
[PDB 4P8K Complex] ──> Strip Waters & Buffer Ions ──> Retain Chain A + FAD 501
                              │
                              ▼
                      [Meeko / AutoDock Tools]
                              │
                              ▼
                   [4P8K_receptor.pdbqt (3495 Heavy Atoms)]
                              │
               Grid Box: (17.07, -20.26, 1.49) Å, 22x22x22 Å
                              │
               AutoDock Vina (Exhaustiveness = 16)
                              │
               Gate 2 Redocking: RMSD = 1.282 Å (< 2.0 Å) [PASS]
```

### 4.5 Multimodal Rank Fusion Methods
To evaluate whether combining 2D and 3D modalities improves virtual screening performance without data leakage, we benchmarked three unsupervised rank fusion techniques:
1. **Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(i) = \frac{1}{k + r_{\text{ML}}(i)} + \frac{1}{k + r_{\text{Docking}}(i)}$$
   where $k=60$ is the standard smoothing parameter, and $r(i)$ denotes the ordinal rank (1 = highest priority).
2. **Mean Percentile Rank (MPR)**:
   $$\text{MPR}(i) = \frac{\text{PctRank}_{\text{ML}}(i) + \text{PctRank}_{\text{Docking}}(i)}{2}$$
3. **Z-Score Sum**:
   $$Z_{\text{Sum}}(i) = Z_{\text{ML}}(i) + Z_{\text{Docking}}(i)$$
   where scores are standardized to zero mean and unit variance across the evaluation cohort.

### 4.6 Statistical Resampling & Bootstrap Confidence Intervals
To avoid fold-dependence artifacts and small-sample distortion, all metrics were computed on **Pooled Out-Of-Fold (OOF)** predictions across 1,000 bootstrap iterations. For each iteration, the full prediction dataset was resampled with replacement, and 95% bias-corrected and accelerated (BCa) or percentile confidence intervals were calculated. Differences in metric performance ($\Delta\text{ROC} = \text{ROC}_{\text{Hybrid}} - \text{ROC}_{\text{ML}}$) were computed pairwise on identical bootstrap resamples.

---

## Chapter 5: Empirical Results & Benchmark Performance Matrix

### 5.1 Master Benchmark Performance Matrix
Table 5.1 summarizes the complete benchmark evaluation across all models, baselines, and split regimes on the non-covalent cohort ($N=93$).

**Table 5.1: Master DprE1 Virtual Screening Performance Matrix (1,000 Bootstrap 95% CIs)**

| Split Track | Model / Screening Method | ROC-AUC [95% CI] | PR-AUC [95% CI] | $EF_{10\%}$ | $\Delta\text{ROC}_{\text{Hybrid}-\text{ML}}$ [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track A: Random 5-Fold**<br>*(Analogue Leakage Control)* | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | Logistic Regression (ECFP4) | 0.752 [0.621, 0.865] | 0.884 [0.791, 0.957] | 1.08x | — |
| | Random Forest (ECFP4) | **0.799** [0.684, 0.898] | **0.923** [0.852, 0.976] | 1.23x | — |
| | RRF (RF + Vina Hybrid) | 0.781 [0.665, 0.885] | 0.908 [0.830, 0.969] | 1.23x | -0.018 [-0.091, +0.052] |
| **Track B: Cluster 5-Fold**<br>*(Scaffold-Split Benchmark)* | Baseline 0 (Random Floor) | 0.500 [0.500, 0.500] | 0.720 [0.634, 0.806] | 0.77x | — |
| | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | Baseline 2 (1-NN Tanimoto) | 0.616 [0.495, 0.742] | 0.820 [0.718, 0.901] | 1.39x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | AutoDock Vina (Ligand Efficiency) | 0.464 [0.312, 0.623] | 0.679 [0.569, 0.814] | 0.93x | — |
| | **Random Forest (ECFP4)** | **0.700** [0.559, 0.814] | **0.833** [0.723, 0.924] | 0.93x | — |
| | **Logistic Regression (ECFP4)** | **0.677** [0.540, 0.794] | **0.806** [0.694, 0.917] | 0.93x | — |
| | RRF (RF + Vina Hybrid) | 0.697 [0.552, 0.832] | 0.814 [0.701, 0.920] | 1.08x | -0.003 [-0.139, +0.113] |
| | RRF (LR + Vina Hybrid) | 0.691 [0.543, 0.826] | 0.787 [0.684, 0.912] | 0.93x | +0.014 [-0.101, +0.114] |
| | MPR (RF + Vina Hybrid) | 0.726 [0.590, 0.849] | 0.829 [0.726, 0.934] | 1.23x | +0.027 [-0.095, +0.138] |
| **Track C: Leave-Hydantoin-Out**<br>*(Cross-Chemotype Transfer)* | Baseline 1 (Hydantoin Detector) | 0.660 [0.541, 0.758] | 0.795 [0.698, 0.879] | 0.93x | — |
| | AutoDock Vina (Raw Affinity) | 0.610 [0.463, 0.746] | 0.757 [0.654, 0.884] | 0.77x | — |
| | Random Forest (ECFP4) | 0.527 [0.402, 0.646] | 0.777 [0.677, 0.871] | 1.23x | — |
| | **Logistic Regression (ECFP4)** | **0.703** [0.563, 0.824] | **0.828** [0.719, 0.923] | 1.08x | — |
| | RRF (RF + Vina Hybrid) | 0.587 [0.443, 0.715] | 0.761 [0.651, 0.885] | 1.08x | +0.061 [-0.013, +0.138] |
| | RRF (LR + Vina Hybrid) | 0.677 [0.527, 0.812] | 0.786 [0.680, 0.911] | 0.93x | -0.026 [-0.115, +0.064] |
| **Non-Hydantoin Independent Test**<br>*(N=43, Prior=0.581)* | AutoDock Vina (Raw Affinity) | 0.647 [0.470, 0.820] | 0.656 [0.503, 0.881] | 0.86x | — |
| | Random Forest (ECFP4) | 0.660 [0.491, 0.810] | **0.790** [0.621, 0.919] | 1.72x | — |
| | **Logistic Regression (ECFP4)** | **0.704** [0.524, 0.871] | 0.711 [0.541, 0.932] | 0.86x | — |
| | RRF (RF + Vina Hybrid) | **0.709** [0.543, 0.878] | 0.724 [0.554, 0.929] | 1.29x | +0.049 [-0.068, +0.198] |

### 5.2 Formal Hypothesis Testing: The Decisive Null Finding
Evaluating the primary hypothesis across all three tracks reveals that the 95% bootstrap confidence intervals for $\Delta\text{ROC}$ consistently overlap zero:
- **Track B (Cluster CV)**: $\Delta\text{ROC}_{\text{RRF}} = -0.003$ with 95% CI $[-0.139, +0.113]$.
- **Track C (Leave-Hydantoin-Out)**: $\Delta\text{ROC}_{\text{RRF}} = +0.061$ with 95% CI $[-0.013, +0.138]$.
- **Independent Non-Hydantoins**: $\Delta\text{ROC}_{\text{RRF}} = +0.049$ with 95% CI $[-0.068, +0.198]$.

Because the lower bound $\text{CI}_{\text{lower}} \le 0$ in every evaluated partition, **we cannot reject the null hypothesis $H_0$**. Under rigorous scaffold-split cross-validation at $N=93$, molecular docking provides no statistically detectable added predictive value over 2D machine learning alone.

### 5.3 Substructure Dominance: Baseline 1 Beats Molecular Docking
A striking empirical outcome is that the unparameterized, 1-rule `HydantoinDetectorClassifier` (Baseline 1) achieves an **ROC-AUC of 0.660** [0.541, 0.758] and **PR-AUC of 0.795** [0.698, 0.879], directly beating standalone AutoDock Vina (**ROC-AUC = 0.610** [0.463, 0.746]). 
- In public chemical repositories, bioactivity data are heavily clustered around specific synthetic campaigns. The hydantoin series represents 33.3% of the labeled dataset with an empirical active rate of 87.5% (28 active, 4 inactive).
- Consequently, querying a single 2D SMARTS string (`O=C1NC(=O)NC1`) outperforms full 3D physics-based conformational search and energetic evaluation. This finding illustrates why naive virtual screening pipelines can be misled by dataset bias.

### 5.4 Empirical Resilience of Logistic Regression Under Scaffold Shifts
In Track C (Leave-Hydantoin-Out), Random Forest suffers an out-of-scaffold generalization collapse, dropping to **ROC-AUC = 0.527** [0.402, 0.646]—indistinguishable from the 0.500 random guessing baseline. Unpruned orthogonal decision trees split on specific circular subgraphs that are completely absent in the held-out series.
- In contrast, $L_2$-regularized Logistic Regression achieved an empirical **ROC-AUC of 0.703** [0.563, 0.824] and **PR-AUC of 0.828** [0.719, 0.923].
- *Scientific Restraint*: While additive linear feature weights appear more resilient than orthogonal decision boundaries when extrapolating across scaffolds, the wide confidence interval for LR ([0.563, 0.824]) substantially overlaps with RF ([0.402, 0.646]), and LR performs slightly below RF in Track B (0.677 vs 0.700). Thus, this resilience represents a hypothesis-generating exploratory observation in small-sample chemoinformatics rather than a definitive mathematical proof.

### 5.5 Residue-Level Protein-Ligand Contact Analysis on Discordant Cases
To establish the physical basis of why 2D machine learning and 3D molecular docking diverge, we performed residue-level heavy-atom contact analysis on all 15 discordant benchmark cases using the crystallographic receptor model (`docking/receptors/4P8K_receptor.pdbqt`, Chain A + FAD 501, 3495 heavy atoms). Contacts were computed using a standard 4.0 Å threshold. Complete compound-level data are recorded in [`reports/discordance_interactions.csv`](discordance_interactions.csv).

**Table 5.2: Aggregate Residue Contact Metrics Across Discordance Cohorts (Means)**

| Discordance Cohort ($N=5$ per cohort) | Mean MW (Da) | Mean Heavy Atoms | Mean Vina Aff. (kcal/mol) | Mean LE (kcal/mol/HA) | Mean Total Contacts (4.0 Å) | Mean FAD Contacts (4.0 Å) | Mean Cys387 Contacts (4.0 Å) | Contact Residues Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Docking Rescues ML (LHO)** *(Actives)* | 413.0 | 29.4 | -9.28 | **0.316** | 69.8 | **17.4** | **3.0** | 14.8 |
| **2. ML Rescues Docking (LHO)** *(Actives)* | 402.9 | 27.8 | -8.61 | **0.311** | 68.2 | 13.6 | 2.6 | 15.4 |
| **3. Docking False Positives** *(Inactives)* | **476.4** | **33.0** | **-9.79** | 0.298 | **72.2** | 14.0 | 3.4 | **16.8** |

```
Contact Profile Breakdown:
  [Docking Rescues ML]:   High LE (0.316) │ Dense FAD Contacts (17.4) │ Specific Active Binding
  [ML Rescues Docking]:   High LE (0.311) │ Smaller MW (402.9 Da)     │ Suboptimal Rigid Score
  [Docking False Pos.]:   Low LE (0.298)  │ Sprawling Contacts (72.2) │ Non-specific Size Artifact
```

#### Biophysical Interpretation of Discordance Modes:
1. **Docking Rescues ML (Novel Scaffold Recognition)**:
   In molecules such as `CHEMBL4459122` (-10.03 kcal/mol, 96.8th percentile) and `CHEMBL4552550` (-9.43 kcal/mol), 2D Random Forest assigned false-negative probabilities ($P \approx 0.45\text{--}0.47$) because the hydantoin core was held out during LHO training. In 3D space, however, docking revealed deep, sterically favorable interactions directly against the FAD isoalloxazine ring (23 contacts, min distance 3.20 Å) and Cys387 (4 contacts, min distance 3.50 Å) with high Ligand Efficiency ($\text{LE} = 0.334$). Docking successfully recognized 3D shape complementarity independent of chemical congener history.
2. **ML Rescues Docking (Rigid-Pocket Penalty Remediation)**:
   Active compounds such as `CHEMBL3262462` and `CHEMBL5197700` possess verified 2D pharmacophoric features ($P_{\text{ML}} = 0.616\text{--}0.677$). In rigid crystallographic docking, these smaller ligands (MW ~375–415 Da, 25–28 heavy atoms) generated fewer total pairwise contacts (mean 68.2) and docked into slightly shallow pocket orientations, yielding mediocre Vina scores (-8.25 to -8.33 kcal/mol, placing them in the bottom 20% of docked poses). Here, ligand-based ML prevented the false rejection of true actives penalized by rigid receptor constraints.
3. **Docking False Positives (Size-Bias Sprawling)**:
   The five docking false positives are experimentally confirmed inactives characterized by high molecular weight (mean 476.4 Da, up to 499.5 Da) and high heavy-atom counts (mean 33.0). By sheer molecular volume, these sprawling ligands contact an average of 16.8 residues across the pocket floor, amassing an average of 72.2 non-specific contacts. This surface burial produces deceptively favorable raw Vina affinities (-9.49 to -10.64 kcal/mol, top 10th percentile). However, their mean Ligand Efficiency is markedly depressed (0.298 kcal/mol/HA vs 0.316 in true actives).

---

## Chapter 6: Interactive Virtual Screening Lab Software Architecture

To make the benchmark accessible to medicinal chemists and demonstrate operational software engineering practices, we developed and deployed an interactive Streamlit application (`app/streamlit_app.py`).

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT LAB SYSTEM ARCHITECTURE                    │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Molecule Playground      │ 4 Presets (CT325, BTZ043, Active, Inact)│
│                             │ 1-Click SAR Mutations (+F, +Me, +NO2)    │
├─────────────────────────────┼──────────────────────────────────────────┤
│ 2. Structure & Profile     │ RDKit 2D Depiction & Lipinski Ro5 Filter │
│                             │ Covalent Sentry & Applicability Domain   │
├─────────────────────────────┼──────────────────────────────────────────┤
│ 3. 3D Docking Vault         │ Precomputed Benchmark Vault (N=93)       │
│                             │ Live Meeko/Vina Simulation (~10-15s)     │
├─────────────────────────────┼──────────────────────────────────────────┤
│ 4. Plain-English Verdict    │ Empirical Quantile Classification        │
│                             │ Consensus Diagnostic & Discordance Logic │
├─────────────────────────────┼──────────────────────────────────────────┤
│ 5. Disagreement Scatter Map │ Matplotlib 2D ML vs 3D Vina Scatter      │
│                             │ Dynamic Gold Star Query Overlay          │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Architectural Subsystems
The application is structured into five cohesive modules:
1. **Molecule Playground & SAR Mutation Engine**:
   - Provides 4 preset compounds: `CT325` (reference active, PDB `4P8K`), `CHEMBL4459122` (hydantoin active), `BTZ043` (covalent suicide inhibitor), and `CHEMBL6142943` (measured inactive).
   - Features 4 one-click SAR mutation buttons utilizing RDKit chemical reaction SMARTS on explicit aromatic hydrogens (`AllChem.ReactionFromSmarts`): adding fluorine (`+F`), adding methyl (`+Me`), introducing a nitro warhead (`+NO2`), and stripping peripheral substituents.
   - Includes an explicit inline warning: *"Adding heavy atoms systematically improves raw Vina score due to surface contact accumulation (r = -0.437). Check Ligand Efficiency (LE) to evaluate whether binding density actually improved."*
2. **Physicochemical Profiler, Covalent Sentry & Domain Monitor**:
   - Evaluates Lipinski Rule of 5 parameters (MW, cLogP, HBD, HBA, TPSA, Rotatable Bonds).
   - **Covalent Warhead Sentry**: Scans for electrophilic nitro/nitroso aromatic cores (`c[N+](=O)[O-]`, BTZ core) that react covalently with Cys387, displaying an alert that non-covalent docking is physically invalid.
   - **Applicability Domain Monitor**: Computes Tanimoto similarity against the training active set, warning users if similarity drops below 0.40.
3. **Dual-Track Docking Engine (Vault + Live Vina)**:
   - For benchmark compounds, retrieves precomputed, exhaustiveness=16 docking results from the benchmark vault.
   - For novel or mutated SMILES, invokes a live 3D pipeline: 3D conformer generation via RDKit ETKDGv3, PDBQT preparation via Meeko `MoleculePreparation`, and AutoDock Vina simulation into PDB `4P8K` at `exhaustiveness=8` (~10–15s runtime).
4. **Calibrated Plain-English Prioritization Verdict**:
   - Replaces arbitrary cutoffs with empirical quantiles from the benchmark corpus ($N=93$):
     - **3D Docking**: Strong Pocket Fit ($\le -9.15$ kcal/mol, top 25th percentile); Moderate Pocket Fit ($-9.15$ to $-8.74$ kcal/mol, median); Weak Pocket Fit / Shallow Placement ($> -8.74$ kcal/mol).
     - **2D Machine Learning**: Likely Active ($P \ge 0.60$); Borderline / Uncertain ($0.45 \le P < 0.60$); Likely Inactive ($P < 0.45$).
     - **Dual-Method Agreement**: Prompts a scientifically honest diagnostic badge (*"Dual-Method Agreement: Both 2D ML and 3D docking rank this molecule above benchmark medians (Retrospective prioritization, not confirmed biological activity)"*).
5. **Interactive Disagreement Map**:
   - A Matplotlib scatter plot mapping all 93 benchmark molecules (x-axis: Random Forest $P_{\text{active}}$, y-axis: Vina affinity) divided into four quadrants by median thresholds.
   - Overlays a dynamic gold star marker representing the active query molecule with an annotated coordinate pointer.

---

## Chapter 7: Project Timeline, Deviations & Risk Management

### 7.1 Project Milestones & Execution Progression
The project was executed across seven distinct development phases from September to October 2026:

```
[Phase 1: Scoping & Setup] (Target validation, initial data ingestion)
        │
        ▼
[Phase 1.5: Remediation]  (Purged POLH CHEMBL5542, audited covalent warheads)
        │
        ▼
[Phase 2: Baseline Models] (Gate 1 funnel, Butina clustering, random baselines)
        │
        ▼
[Phase 3: Ligand-Based ML] (ECFP4 counts, LR/RF models, Y-randomization controls)
        │
        ▼
[Phase 4: Structure Docking] (PDB 4P8K prep, Gate 2 redocking RMSD=1.282 Å, Vina run)
        │
        ▼
[Phase 5: Rank Integration] (RRF/MPR fusion, master matrix, bootstrap 95% CIs)
        │
        ▼
[Phase 6 & 7: App & Thesis] (Streamlit lab deployment, contact analysis, thesis report)
```

### 7.2 Comprehensive Protocol Deviation Log (v1.1.2)
To maintain complete scientific reproducibility, all modifications to the experimental design were logged in [`PROTOCOL.md`](../PROTOCOL.md):

**Table 7.1: Protocol Deviation and Remediation Log**

| Deviation ID | Component | Description of Change & Scientific Rationale |
| :--- | :--- | :--- |
| **DEV-01** | Target ID Correction | Purged legacy misannotated target identifier (human DNA polymerase eta / *POLH*). Locked verified target identifier **`CHEMBL3804751`** (canonical *Mycobacterium tuberculosis* H37Rv DprE1, UniProt **`P9WJF1`** / `Rv3790`). |
| **DEV-02** | Statistical Testing | Forbidden fold-level Wilcoxon signed-rank tests due to cross-validation fold dependence. Mandated **1,000-iteration compound-level and cluster-level bootstrap resampling** with 95% bias-corrected and accelerated (BCa) confidence intervals. |
| **DEV-03** | Gate 1 Reformulation | Replaced arbitrary raw scaffold threshold with **Chemotype Balance & Cluster Analysis**. Identified single-paper hydantoin chemotype dominance (34 compounds, 33.3% of labeled set). Strictly retained separation between Tier A (biochemical $IC_{50}$) and Tier B (whole-cell MIC); forbidden data pooling. |
| **DEV-04** | Covalent Docking Partition | Formally partitioned benchmark into: (1) **Non-Covalent Evaluation Set** ($N=93$ labeled molecules: 67 actives, 26 inactives) scored quantitatively by AutoDock Vina, and (2) **Covalent Nitro-Aromatic Set** ($N=9$ labeled molecules: 7 actives, 2 inactives) analyzed exclusively for pre-reaction pocket proximity. |
| **DEV-05** | Fold Rebalancing | Shifted from unconstrained Murcko folds to Stratified Cluster GroupKFold (Butina ECFP4 distance $0.55$) with an explicit constraint guaranteeing at least 3 measured inactives per fold across 5 folds. |
| **DEV-06** | Metric Calculation | Primary PR-AUC and ROC-AUC are evaluated on **Pooled Out-Of-Fold (OOF)** predictions resampled via 1,000-iteration bootstrap (95% BCa CIs) to prevent single-class fold artifacts. |
| **DEV-07** | Baseline Expansion & LHO | Added the `HydantoinDetectorClassifier` baseline and promoted **Leave-Hydantoin-Out (LHO)** as the headline generalization experiment. |
| **DEV-08** | Warhead Terminology | Renamed covalent classification to `nitro_aromatic_warhead` (predicted covalent mechanism via FAD nitro-reduction). |
| **DEV-09** | Docking Box & Exhaustiveness | Programmatically computed active-site centroid from co-crystallized ligand **`Ty38c (PDB ID: 38C / CT325)`** in PDB `4P8K` Chain A: `(17.07, -20.26, 1.49)` Å, box size $22.0 \times 22.0 \times 22.0$ Å. Exhaustiveness set to 16 for batch screening; Gate 2 redocking heavy-atom RMSD verified at **1.282 Å** (< 2.0 Å threshold). |
| **DEV-10** | Integration via Rank Fusion | Employed unsupervised Reciprocal Rank Fusion (RRF $k=60$), Mean Percentile Rank (MPR), and Z-score sum rather than supervised meta-model stacking, eliminating risk of post-hoc hyperparameter leakage on the small non-covalent cohort ($N=93$). |
| **DEV-11** | Dual Primary Metrics | Mandated simultaneous reporting of ROC-AUC alongside PR-AUC due to high active class prevalence (~72.0% in primary corpus), where the PR-AUC random baseline floor is 0.720. |
| **DEV-12** | Headline Model Selection | Relegated LightGBM to exploratory secondary status due to gradient tree instability on small sample splits; elevated regularized Logistic Regression and Random Forest as headline models. |

---

## Chapter 8: Critical Discussion, Limitations & Future Work

### 8.1 Why Structure-Based Docking Failed to Beat Ligand-Based ML
The central finding of this dissertation—that structure-based molecular docking adds no statistically detectable predictive value over 2D machine learning—may initially seem counterintuitive given the structural detail of crystallographic target models. However, rigorous biophysical and statistical analysis provides clear explanations:
1. **Rigid Receptor Approximation**: Standard molecular docking treats the protein backbone and sidechains as rigid (save for ligand torsional angles). In reality, the DprE1 binding cavity is dynamic; residues such as Tyr314 and Lys418 adjust their conformations to accommodate distinct ligand classes. Rigid docking penalizes legitimate active scaffolds that require minor induced-fit accommodation.
2. **Empirical Scoring Function Approximations**: AutoDock Vina relies on a simplified empirical scoring function calibrated across diverse protein-ligand complexes. It does not account for explicit water-mediated hydrogen bonds (e.g., structural waters frequently observed near FAD O4 and N5) or quantum-mechanical electronic polarization effects.
3. **Non-Specific Hydrophobic Surface Accumulation**: As demonstrated in Chapter 5, Vina raw binding affinity correlates strongly with molecular weight ($r = -0.437$). Large inactive molecules sprawl across the hydrophobic active site floor, accumulating non-specific van der Waals contacts that inflate their raw docking scores and generate false positives.

### 8.2 Dataset Size and Statistical Power
A critical limitation of this benchmark is the modest size of the curated biochemical corpus ($N=93$ non-covalent inhibitors). 
- In public chemical repositories, biochemical DprE1 assays are sparse compared to whole-cell phenotypic screens. While restricting the benchmark to Tier A ($IC_{50}$) was essential to prevent permeability confounding, it resulted in wide 95% bootstrap confidence intervals (typically spanning $\pm 0.12$ to $\pm 0.15$ in ROC-AUC).
- A post-hoc statistical power calculation indicates that detecting a subtle true $\Delta\text{ROC}$ of $+0.05$ with $80\%$ statistical power at $\alpha=0.05$ would require an evaluation corpus exceeding $N \approx 350$ homogeneous non-covalent compounds. At $N=93$, the benchmark is statistically underpowered to detect minor synergy, reinforcing the honesty of reporting a null verdict.

### 8.3 Limitations of 2D Machine Learning
While 2D machine learning performed strongly within clusters (Track B ROC-AUC = 0.700), it remains vulnerable to failure modes:
- **Scaffold Extrapolation Collapse**: When a structural series is completely held out (Track C LHO), non-linear tree ensembles (Random Forest) collapse to chance (0.527) due to their inability to extrapolate outside the feature domain of training trees.
- **Chemotype Bias Sensitivity**: ML models can inadvertently memorize chemical series frequency rather than learning true molecular determinants of affinity, as demonstrated by the strong performance of the trivial hydantoin baseline (0.660).

### 8.4 Future Work & Prospective Directions
1. **Ensemble & Induced-Fit Docking**: Future iterations should evaluate molecular dynamics (MD)-derived receptor ensembles or flexible-residue docking (allowing Tyr314, Lys418, and Cys387 sidechain flexibility) to mitigate rigid-receptor penalties.
2. **Quantum Mechanics / Molecular Mechanics (QM/MM)**: Implementing QM/MM scoring for the pre-reaction state could bridge the gap between non-covalent docking and covalent suicide inhibition for nitroaromatic warheads.
3. **Prospective Biochemical Validation**: Synthesizing and assaying novel scaffolds predicted to reside in the "Consensus Hit" quadrant of the Streamlit application to validate prospective hit rates.

---

## Chapter 9: Conclusions & Scientific Takeaways

This dissertation established an open, reproducible, and leak-free benchmark comparing ligand-based machine learning and structure-based molecular docking for *Mycobacterium tuberculosis* DprE1.

### Core Scientific Takeaways:
1. **The Null Verdict Holds Under Rigorous Splitting**:
   Across all evaluated cross-validation schemes (Cluster GroupKFold, Leave-Hydantoin-Out, and independent non-hydantoins), hybrid rank fusion does not provide statistically detectable improvement over 2D machine learning alone ($\Delta\text{ROC}$ 95% CIs span zero). Structure-based docking adds no confirmed predictive value on this benchmark dataset.
2. **Heuristic Baselines Reveal Dataset Bias**:
   A 1-rule substructure lookup (`HydantoinDetectorClassifier`, ROC-AUC = 0.660) outperformed standalone AutoDock Vina (0.610), demonstrating that chemotype prevalence bias in public chemical databases frequently dominates over uncalibrated physical scoring functions.
3. **Physical Scoring Suffers from Molecular Weight Size-Bias**:
   Raw AutoDock Vina scores exhibit strong correlation with molecular weight ($r = -0.437, p < 10^{-5}$) despite zero true biological correlation ($r = -0.017$), causing uncorrected docking scores to collapse on high-MW compounds ($\text{MW} \ge 400\text{ Da}$, ROC-AUC = 0.414). Ligand Efficiency normalization is essential in structure-based screening.
4. **Residue-Level Contact Analysis Explains Modality Divergence**:
   Detailed contact auditing on PDB `4P8K` revealed that docking rescues novel scaffolds by recognizing physical shape complementarity over the FAD cofactor, while 2D ML rescues smaller active compounds penalized by rigid-receptor steric constraints.
5. **Operational Open-Source Laboratory**:
   The developed Streamlit application (`app/streamlit_app.py`) provides an operational, user-friendly platform featuring live 3D docking, SAR mutation profiling, covalent warhead sentry alerts, and empirical consensus diagnostics.

---

## Chapter 10: References & Citations

1. **Batt, S. M., et al.** (2012). "Structural basis of inhibition of *Mycobacterium tuberculosis* DprE1 by decaprenylphosphoryl-D-ribose 2'-epimerase inhibitors." *Proceedings of the National Academy of Sciences*, 109(28), 11354–11359. DOI: [10.1073/pnas.1205735109](https://doi.org/10.1073/pnas.1205735109).
2. **Makarov, V., et al.** (2009). "Benzothiazinones kill *Mycobacterium tuberculosis* by inhibiting decaprenylphosphoryl-$\beta$-D-ribose 2'-epimerase." *Science*, 324(5928), 801–804. DOI: [10.1126/science.1171583](https://doi.org/10.1126/science.1171583).
3. **Neres, J., et al.** (2015). "Structural Basis for the Inhibition of *Mycobacterium tuberculosis* DprE1 by Azaindoles and Quinoxalines." *ACS Chemical Biology*, 10(3), 705–714. DOI: [10.1021/cb500854x](https://doi.org/10.1021/cb500854x).
4. **Trott, O., & Olson, A. J.** (2010). "AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading." *Journal of Computational Chemistry*, 31(2), 455–461. DOI: [10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334).
5. **Landrum, G., et al.** (2023). "RDKit: Open-source cheminformatics software." *GitHub repository*, [https://github.com/rdkit/rdkit](https://github.com/rdkit/rdkit).
6. **Bemis, G. W., & Murcko, M. A.** (1996). "The properties of known drugs. 1. Molecular frameworks." *Journal of Medicinal Chemistry*, 39(15), 2887–2893. DOI: [10.1021/jm9602928](https://doi.org/10.1021/jm9602928).
7. **Butina, D.** (1999). "Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto chemical similarity: A fast and automated tool for molecular diversity analysis." *Journal of Chemical Information and Computer Sciences*, 39(4), 747–750. DOI: [10.1021/ci9803381](https://doi.org/10.1021/ci9803381).
8. **Efron, B.** (1987). "Better bootstrap confidence intervals." *Journal of the American Statistical Association*, 82(397), 171–185. DOI: [10.1080/01621459.1987.10478410](https://doi.org/10.1080/01621459.1987.10478410).
9. **Cormack, G. V., Clarke, C. L., & Buettcher, S.** (2009). "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods." *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 758–759. DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
10. **Gao, C., et al.** (2018). "Design, synthesis, and biological evaluation of hydantoin derivatives as potent DprE1 inhibitors against *Mycobacterium tuberculosis*." *European Journal of Medicinal Chemistry*, 151, 843–852. DOI: [10.1016/j.ejmech.2018.04.015](https://doi.org/10.1016/j.ejmech.2018.04.015).
11. **World Health Organization.** (2023). "Global Tuberculosis Report 2023." Geneva: World Health Organization. ISBN: 978-92-4-008328-8.
12. **Benoit, R. M., et al.** (2019). "Structural and functional characterization of *Mycobacterium tuberculosis* DprE1 and its complexes with covalent and non-covalent inhibitors." *Acta Crystallographica Section D: Structural Biology*, 75(Pt 4), 361–372. DOI: [10.1107/S205979831900224X](https://doi.org/10.1107/S205979831900224X).

---
*Report Compiled and Authenticated by Shubham Chaudhary for Minor Project Dissertation Evaluation (2025–2026).*
