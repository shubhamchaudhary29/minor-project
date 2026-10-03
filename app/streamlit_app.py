#!/usr/bin/env python3
"""
app/streamlit_app.py

Phase 6: DprE1 Virtual Screening & Predictive Triage Platform
Author: Shubham Chaudhary

An interactive Streamlit application demonstrating:
1. Real-time SMILES validation and 2D chemical structure rendering via RDKit.
2. Physicochemical property profiling (MW, LogP, HBD, HBA, TPSA, RotB) and Lipinski filter.
3. Automated sentry for covalent nitro-aromatic warheads (targeting Cys387 via FAD reduction).
4. Machine Learning inferences:
   - For benchmark compounds: displays precomputed, leak-free Out-Of-Fold (OOF) cross-validation predictions.
   - For novel compounds: displays live predictions from calibrated models fitted on the full dataset.
5. Applicability domain monitoring (Tanimoto similarity to ChEMBL3804751 training actives).
6. Precomputed AutoDock Vina docking score, true percentile rank, and Ligand Efficiency retrieval.
7. Executive benchmark synthesis report and model comparison matrix.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator, DataStructs

# Set page config
st.set_page_config(
    page_title="DprE1 Screening & Triage Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .header-box h1 {
        color: white;
        font-size: 28px;
        margin-bottom: 5px;
    }
    .header-box p {
        color: #e0e8f9;
        font-size: 14px;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# Canonical SMARTS patterns
HYDANTOIN_SMARTS = "O=C1NC(=O)NC1"
NITRO_AROMATIC_SMARTS = "c[N+](=O)[O-]"
NITROSO_AROMATIC_SMARTS = "c[N]=O"
BTZ_CORE_SMARTS = "c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2"

_PAT_HYDANTOIN = Chem.MolFromSmarts(HYDANTOIN_SMARTS)
_PAT_NITRO = Chem.MolFromSmarts(NITRO_AROMATIC_SMARTS)
_PAT_NITROSO = Chem.MolFromSmarts(NITROSO_AROMATIC_SMARTS)
_PAT_BTZ = Chem.MolFromSmarts(BTZ_CORE_SMARTS)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")


@st.cache_resource
def load_and_train_models():
    """
    Loads curated biochemical training corpus (stratified_cluster_folds.csv)
    and fits calibrated Logistic Regression and Random Forest models.
    Also caches training actives for applicability domain lookup.
    """
    csv_file = os.path.join(DATA_PATH, "stratified_cluster_folds.csv")
    if not os.path.exists(csv_file):
        st.error(f"Missing training corpus at {csv_file}")
        return None, None, None, None, None

    df = pd.read_csv(csv_file)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    X_counts = []
    X_bits = []
    y = []
    actives_info = []

    for _, row in df.iterrows():
        s = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        is_act = 1 if row["label"] == "Active" else 0
        y.append(is_act)

        # Count vector
        fp_count = gen.GetCountFingerprint(mol)
        arr_cnt = np.zeros(2048, dtype=np.float32)
        for bit_id, val in fp_count.GetNonzeroElements().items():
            arr_cnt[bit_id] = val
        X_counts.append(arr_cnt)

        # Bit vector for Tanimoto
        fp_bit = gen.GetFingerprint(mol)
        X_bits.append(fp_bit)

        if is_act == 1:
            actives_info.append({
                "chembl_id": row["molecule_chembl_ids"],
                "smiles": s,
                "fp": fp_bit,
                "is_hydantoin": row["is_hydantoin"]
            })

    X_counts = np.vstack(X_counts)
    y = np.array(y)

    # 1. Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_counts, y)

    # 2. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, min_samples_split=4, class_weight="balanced_subsample", random_state=42)
    rf.fit(X_counts, y)

    # Load master predictions for docking lookup
    master_csv = os.path.join(DATA_PATH, "master_predictions_phase5.csv")
    df_master = pd.read_csv(master_csv) if os.path.exists(master_csv) else pd.DataFrame()

    return lr, rf, gen, actives_info, df_master


def check_covalent_warhead(mol):
    """Detects if molecule has an activated aromatic nitro or nitroso warhead."""
    has_nitro = mol.HasSubstructMatch(_PAT_NITRO)
    has_nitroso = mol.HasSubstructMatch(_PAT_NITROSO)
    has_btz = mol.HasSubstructMatch(_PAT_BTZ)
    return has_nitro or has_nitroso or has_btz


def compute_ad_similarity(query_mol, gen, actives_info):
    """Computes max Tanimoto similarity to known training actives."""
    q_fp = gen.GetFingerprint(query_mol)
    best_sim = 0.0
    best_match = None

    for act in actives_info:
        sim = DataStructs.TanimotoSimilarity(q_fp, act["fp"])
        if sim > best_sim:
            best_sim = sim
            best_match = act

    return best_sim, best_match


def main():
    # Top Disclaimer Banner
    st.warning("⚠️ **Academic Research Prototype**: In silico predictions do not constitute biological confirmation.")

    # Header Box
    st.markdown("""
    <div class="header-box">
        <h1>🔬 DprE1 Virtual Screening & Predictive Triage Platform</h1>
        <p>Target Flavoenzyme: <i>Mycobacterium tuberculosis</i> H37Rv DprE1 (Rv3790, UniProt P9WJF1, PDB 4P8K) | Author: <b>Shubham Chaudhary</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Load models and data
    lr_model, rf_model, fp_gen, actives_info, df_master = load_and_train_models()
    if lr_model is None:
        st.stop()

    # Preset Demonstrator Library
    PRESETS = {
        "CT325 (38C) — Reference Non-Covalent Active (PDB 4P8K)": "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1",
        "CHEMBL4459122 — Potent Hydantoin Active": "CC1(c2ccc(C#N)cc2)NC(=O)N(CC(=O)c2ccc(F)c(C(F)(F)F)c2)C1=O",
        "CHEMBL5564800 — Nitrobenzothiazinone Covalent Active (BTZ Series)": "CN(CC#CC1CCCCC1)c1nc(=O)c2cc(C(F)(F)F)cc([N+](=O)[O-])c2s1",
        "CHEMBL6142943 — Verified Biochemical Inactive": "C=C(C)C(=O)Nc1cc(C(F)(F)F)cc2c(=O)nc(N3CCC4(CC3)OC[C@H](C)O4)sc12",
        "Custom SMILES Entry": ""
    }

    # Sidebar
    st.sidebar.header("🎯 Compound Input & Triage")
    selected_preset = st.sidebar.selectbox("Choose a Demo Compound or Custom:", list(PRESETS.keys()))

    default_smiles = PRESETS[selected_preset]
    if selected_preset == "Custom SMILES Entry":
        default_smiles = "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1"

    query_smiles = st.sidebar.text_area("Input SMILES String:", value=default_smiles, height=100).strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 Benchmark Protocol v1.1.2
    - **Target**: DprE1 (`CHEMBL3804751`)
    - **Receptor**: PDB `4P8K` (Chain A + FAD)
    - **Docking Box**: `(17.07, -20.26, 1.49)` Å
    - **Exhaustiveness**: 16 (DEV-09)
    - **Gate 2 Pose Validation**: RMSD = 1.282 Å (< 2.0 Å)
    - **Evaluation**: 1,000 Bootstrap CIs on Pooled OOF
    """)

    # Main Area Layout
    col_left, col_right = st.columns([1, 1])

    # Parse SMILES
    mol = Chem.MolFromSmiles(query_smiles) if query_smiles else None

    # Check if entered SMILES matches benchmark master dataset
    matched_row = None
    if mol is not None and not df_master.empty:
        try:
            canon_q = Chem.CanonSmiles(query_smiles)
            for _, r in df_master.iterrows():
                if Chem.CanonSmiles(r["canonical_smiles"]) == canon_q:
                    matched_row = r
                    break
        except Exception:
            pass

    with col_left:
        st.subheader("1. Structure & Physicochemical Profile")
        if mol is None:
            st.error("❌ Invalid SMILES string. Please provide a chemically valid SMILES.")
            st.stop()

        # Render 2D structure
        img = Draw.MolToImage(mol, size=(450, 280))
        st.image(img, caption="2D Chemical Topology (RDKit Depiction)", use_container_width=True)

        # Compute Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        lipinski_pass = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)

        # Metric Grid
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Mol. Weight", f"{mw:.1f} Da")
        mcol2.metric("LogP (cLogP)", f"{logp:.2f}")
        mcol3.metric("TPSA", f"{tpsa:.1f} Å²")

        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("H-Donors (HBD)", hbd)
        mcol5.metric("H-Acceptors (HBA)", hba)
        mcol6.metric("Rotatable Bonds", rotb)

        if lipinski_pass:
            st.success("✅ **Lipinski Rule of 5**: Compliant (Oral Drug-like Space)")
        else:
            st.warning("⚠️ **Lipinski Rule of 5**: Violation detected (Potential bioavailability liability)")

    with col_right:
        st.subheader("2. Target Sentry & Predictive Inference")

        # 1. Covalent Warhead Sentry
        is_covalent = check_covalent_warhead(mol)
        if is_covalent:
            st.error("""
            🚨 **PREDICTED COVALENT SUICIDE INHIBITOR**:
            Detected aromatic nitro/nitroso warhead (`c[N+](=O)[O-]` / BTZ scaffold).
            **Biophysical Mechanism**: Undergoes FAD-mediated reduction to a nitroso species, forming a covalent semimercaptal adduct with **Cys387**.
            **Modeling Consequence**: Non-covalent thermodynamic docking scoring (e.g., AutoDock Vina) is **physically invalid** for this molecule.
            """)
        else:
            st.info("🛡️ **Non-Covalent Ligand**: No electrophilic aromatic nitro warheads detected. Reversible non-covalent binding mode.")

        # 2. Chemotype Detection
        is_hyd = mol.HasSubstructMatch(_PAT_HYDANTOIN)
        if is_hyd:
            st.warning("🏷️ **Hydantoin Core Identified**: Matches `O=C1NC(=O)NC1`. Belongs to dominant literature chemotype cluster (Baseline 1 Score: 0.843).")

        # 3. Applicability Domain
        best_sim, best_match = compute_ad_similarity(mol, fp_gen, actives_info)
        if best_sim < 0.40:
            st.warning(f"⚠️ **OUT-OF-DOMAIN WARNING**: Max Tanimoto similarity to training actives is **{best_sim:.3f}** (< 0.40 threshold). High extrapolation risk.")
        else:
            st.success(f"✅ **IN-DOMAIN**: Max Tanimoto similarity to known active is **{best_sim:.3f}** (Nearest: `{best_match['chembl_id']}`).")

        # 4. Machine Learning Inference Handling
        if matched_row is not None:
            st.info("ℹ️ **Benchmark Molecule**: Showing precomputed, leak-free Out-Of-Fold (OOF) cross-validation predictions.")
            p_lr_cluster = float(matched_row["prob_lr_cluster"])
            p_rf_cluster = float(matched_row["prob_rf_cluster"])
            p_lr_lho = float(matched_row["prob_lr_lho"])
            p_rf_lho = float(matched_row["prob_rf_lho"])

            st.markdown("#### 🤖 Precomputed Out-Of-Fold (OOF) Inferences")
            tab_cluster, tab_lho = st.tabs(["Track B: Cluster 5-Fold CV (Series-Disjoint)", "Track C: Leave-Hydantoin-Out (Cross-Chemotype)"])

            with tab_cluster:
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.metric("Logistic Regression P(Active)", f"{p_lr_cluster:.1%}")
                    st.progress(p_lr_cluster)
                    st.caption("Series-Disjoint OOF | Benchmark: ROC-AUC = 0.677")
                with pcol2:
                    st.metric("Random Forest P(Active)", f"{p_rf_cluster:.1%}")
                    st.progress(p_rf_cluster)
                    st.caption("Series-Disjoint OOF | Benchmark: ROC-AUC = 0.700")

            with tab_lho:
                lcol1, lcol2 = st.columns(2)
                with lcol1:
                    st.metric("Logistic Regression P(Active)", f"{p_lr_lho:.1%}")
                    st.progress(p_lr_lho)
                    st.caption("Cross-Chemotype OOF | Benchmark: ROC-AUC = 0.703")
                with lcol2:
                    st.metric("Random Forest P(Active)", f"{p_rf_lho:.1%}")
                    st.progress(p_rf_lho)
                    st.caption("Cross-Chemotype OOF | Benchmark: ROC-AUC = 0.527")

        else:
            st.warning("⚠️ **Novel Structure**: Showing live predictions from models fitted on the full benchmark dataset.")
            # Compute live count vector
            fp_count = fp_gen.GetCountFingerprint(mol)
            arr_cnt = np.zeros(2048, dtype=np.float32)
            for bit_id, val in fp_count.GetNonzeroElements().items():
                arr_cnt[bit_id] = val

            p_lr = float(lr_model.predict_proba([arr_cnt])[0, 1])
            p_rf = float(rf_model.predict_proba([arr_cnt])[0, 1])

            st.markdown("#### 🤖 Live Ligand-Based Machine Learning Inferences")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.metric("Logistic Regression P(Active)", f"{p_lr:.1%}")
                st.progress(p_lr)
                st.caption("Regularized L2 Linear Model (Benchmark on LHO: ROC-AUC = 0.703)")

            with pcol2:
                st.metric("Random Forest P(Active)", f"{p_rf:.1%}")
                st.progress(p_rf)
                st.caption("300-Tree Subsampled Ensemble (Benchmark on Cluster CV: ROC-AUC = 0.700)")

    # Section 3: Structure-Based Docking Vault
    st.markdown("---")
    st.subheader("3. Structure-Based Docking Vault (AutoDock Vina & Ligand Efficiency)")

    if matched_row is not None:
        st.success(f"🎯 **Benchmark Compound Identified**: `{matched_row['molecule_chembl_ids']}` (Experimental Assay Label: **{matched_row['label']}**)")

        vina_val = float(matched_row['vina_score'].iloc[0] if hasattr(matched_row['vina_score'], 'iloc') else matched_row['vina_score'])
        vina_aff = float(matched_row['vina_affinity'].iloc[0] if hasattr(matched_row['vina_affinity'], 'iloc') else matched_row['vina_affinity'])

        # Calculate genuine percentile rank against benchmark corpus (N=93)
        percentile = float((df_master['vina_score'] < vina_val).mean() * 100.0)
        top_pct = 100.0 - percentile
        vina_rank_str = f"Top {top_pct:.1f}% ({percentile:.1f}th Pct)"

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric("Vina Affinity (ΔG)", f"{vina_aff:.2f} kcal/mol")
        dcol2.metric("Vina Percentile Rank", vina_rank_str)
        dcol3.metric("Ligand Efficiency (LE)", f"{float(matched_row['ligand_efficiency']):.3f} kcal/mol/HA")
        dcol4.metric("RRF Hybrid Rank", f"{float(matched_row['rrf_rf_vina_cluster']):.4f}")

        st.caption(f"Evaluated in PDB `4P8K` (Chain A + rigid FAD, 2.49 Å, exhaustiveness=16). Heavy atoms: {n_heavy} | Cluster Fold: {matched_row['fold']}")
    else:
        st.info("""
        ℹ️ **Precomputed Docking Score Not Available in Vault**:
        This compound was not part of the primary $N=93$ non-covalent benchmark cohort. Live AutoDock Vina simulation is disabled in the web interface to maintain sub-second responsiveness.
        To dock this novel molecule into PDB `4P8K`, run the batch pipeline via:
        `micromamba run -n dpre1 python src/dock.py --step batch`
        """)

    # Section 4: Master Benchmark Results Matrix Tab
    st.markdown("---")
    with st.expander("📊 View Master Benchmark Synthesis Matrix & viva Defense Findings", expanded=False):
        st.markdown(r"""
        ### Empirical Benchmark Performance Summary (1,000 Bootstrap 95% CIs)
        *Evaluated on N=93 Non-Covalent DprE1 Inhibitors (ChEMBL3804751)*
        """)

        summary_table = pd.DataFrame([
            {"Cohort / Split": "Track B (Cluster 5-Fold CV)", "Method": "Random Forest (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.700 [0.559, 0.814]", "PR-AUC [95% CI]": "0.833 [0.723, 0.924]", "EF 10%": "0.93x"},
            {"Cohort / Split": "Track B (Cluster 5-Fold CV)", "Method": "Logistic Regression (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.677 [0.540, 0.794]", "PR-AUC [95% CI]": "0.806 [0.694, 0.917]", "EF 10%": "0.93x"},
            {"Cohort / Split": "Track B (Cluster 5-Fold CV)", "Method": "AutoDock Vina (Raw Affinity)", "ROC-AUC [95% CI]": "0.610 [0.463, 0.746]", "PR-AUC [95% CI]": "0.757 [0.654, 0.884]", "EF 10%": "0.77x"},
            {"Cohort / Split": "Track B (Cluster 5-Fold CV)", "Method": "RRF (RF + Vina Hybrid)", "ROC-AUC [95% CI]": "0.697 [0.552, 0.832]", "PR-AUC [95% CI]": "0.814 [0.701, 0.920]", "EF 10%": "1.08x"},
            {"Cohort / Split": "Track C (Leave-Hydantoin-Out)", "Method": "Baseline 1 (Hydantoin Detector)", "ROC-AUC [95% CI]": "0.660 [0.541, 0.758]", "PR-AUC [95% CI]": "0.795 [0.698, 0.879]", "EF 10%": "0.93x"},
            {"Cohort / Split": "Track C (Leave-Hydantoin-Out)", "Method": "AutoDock Vina (Raw Affinity)", "ROC-AUC [95% CI]": "0.610 [0.463, 0.746]", "PR-AUC [95% CI]": "0.757 [0.654, 0.884]", "EF 10%": "0.77x"},
            {"Cohort / Split": "Track C (Leave-Hydantoin-Out)", "Method": "Random Forest (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.527 [0.402, 0.646]", "PR-AUC [95% CI]": "0.777 [0.677, 0.871]", "EF 10%": "1.23x"},
            {"Cohort / Split": "Track C (Leave-Hydantoin-Out)", "Method": "Logistic Regression (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.703 [0.563, 0.824]", "PR-AUC [95% CI]": "0.828 [0.719, 0.923]", "EF 10%": "1.08x"},
            {"Cohort / Split": "Track C (Leave-Hydantoin-Out)", "Method": "RRF (RF + Vina Hybrid)", "ROC-AUC [95% CI]": "0.587 [0.443, 0.715]", "PR-AUC [95% CI]": "0.761 [0.651, 0.885]", "EF 10%": "1.08x"},
            {"Cohort / Split": "Non-Hydantoin Independent Test (N=43)", "Method": "Logistic Regression (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.704 [0.524, 0.871]", "PR-AUC [95% CI]": "0.711 [0.541, 0.932]", "EF 10%": "0.86x"},
            {"Cohort / Split": "Non-Hydantoin Independent Test (N=43)", "Method": "Random Forest (ECFP4 Counts)", "ROC-AUC [95% CI]": "0.660 [0.491, 0.810]", "PR-AUC [95% CI]": "0.790 [0.621, 0.919]", "EF 10%": "1.72x"},
            {"Cohort / Split": "Non-Hydantoin Independent Test (N=43)", "Method": "AutoDock Vina (Raw Affinity)", "ROC-AUC [95% CI]": "0.647 [0.470, 0.820]", "PR-AUC [95% CI]": "0.656 [0.503, 0.881]", "EF 10%": "0.86x"},
            {"Cohort / Split": "Non-Hydantoin Independent Test (N=43)", "Method": "RRF (RF + Vina Hybrid)", "ROC-AUC [95% CI]": "0.709 [0.543, 0.878]", "PR-AUC [95% CI]": "0.724 [0.554, 0.929]", "EF 10%": "1.29x"},
        ])
        st.dataframe(summary_table, use_container_width=True)

        st.markdown(r"""
        **Core Scientific Takeaways**:
        1. **Null Result**: At $N=93$ non-covalent inhibitors, hybrid rank fusion does not achieve statistically significant predictive gains over ML alone ($\Delta\text{ROC}$ 95% CIs cross zero).
        2. **Heuristic Strength**: A simple hydantoin substructure detector (ROC-AUC = 0.660) beats AutoDock Vina (0.610).
        3. **Observation on Linear Resilience**: Logistic Regression (0.703) showed empirical resilience under scaffold inversion (Leave-Hydantoin-Out), though wide CIs overlap with RF.
        4. **Size-Bias Confounder**: Vina raw affinity correlates with MW ($r = -0.437, p = 1.21 \times 10^{-5}$); Ligand Efficiency correction mitigates high-MW false positives.
        """)

    # Bottom Disclaimer Banner
    st.markdown("---")
    st.warning("⚠️ **Academic Research Prototype**: In silico predictions do not constitute biological confirmation. Benchmark developed by **Shubham Chaudhary**.")


if __name__ == "__main__":
    main()
