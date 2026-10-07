#!/usr/bin/env python3
"""
app/streamlit_app.py

Interactive DprE1 Virtual Screening & Lead Optimization Lab
Author: Shubham Chaudhary

Features:
1. Molecule Playground:
   - 4 Presets: CT325 (reference active), Hydantoin active, BTZ043 (covalent suicide inhibitor), CHEMBL6142943 (inactive).
   - 4 One-Click SAR Mutations: +F, +Me, +NO2 warhead, and substituent stripping.
2. Structure & Physicochemical Profiling:
   - RDKit 2D depiction & Lipinski Rule of 5 filter.
   - Covalent Warhead Sentry (nitro-aromatic FAD reduction mechanism targeting Cys387).
   - Applicability Domain Monitor (Tanimoto similarity to ChEMBL3804751 training actives).
3. Fast Live AutoDock Vina Docking & Precomputed Vault:
   - Instant retrieval for benchmark compounds.
   - Fast live 3D docking (~10-15s, exhaustiveness=8) into PDB 4P8K (Chain A + rigid FAD) for custom/mutated ligands.
4. Plain-English Prioritization Verdict:
   - 2D ML (Likely Active / Uncertain / Likely Inactive).
   - 3D Docking (Strong Fit / Moderate Fit / Weak Fit).
   - Consensus Diagnostic (Consensus Hit, Consensus Rejection, Discordant Rescue, Steric Trap).
5. Interactive Virtual Screening Disagreement Map:
   - Matplotlib scatter plot mapping all 93 benchmark compounds with dynamic gold star overlay for the query molecule.
6. Viva Defense Synthesis Accordion:
   - Honest reporting of benchmark null results (docking ROC-AUC 0.610, hybrid Delta crosses zero, hydantoin baseline 0.660).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdFingerprintGenerator, DataStructs
import meeko
from meeko import MoleculePreparation
from vina import Vina
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="DprE1 Lead Optimization Lab",
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
    .verdict-box {
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RECEPTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "docking", "receptors", "4P8K_receptor.pdbqt")

# Canonical SMARTS
HYDANTOIN_SMARTS = "O=C1NC(=O)NC1"
NITRO_AROMATIC_SMARTS = "c[N+](=O)[O-]"
NITROSO_AROMATIC_SMARTS = "c[N]=O"
BTZ_CORE_SMARTS = "c1c([N+](=O)[O-])cc2c(c1)C(=O)NCS2"

_PAT_HYDANTOIN = Chem.MolFromSmarts(HYDANTOIN_SMARTS)
_PAT_NITRO = Chem.MolFromSmarts(NITRO_AROMATIC_SMARTS)
_PAT_NITROSO = Chem.MolFromSmarts(NITROSO_AROMATIC_SMARTS)
_PAT_BTZ = Chem.MolFromSmarts(BTZ_CORE_SMARTS)

# Reaction SMARTS using explicit hydrogens on aromatic carbons
RXN_ADD_F = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1]F')
RXN_ADD_ME = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1]C')
RXN_ADD_NO2 = AllChem.ReactionFromSmarts('[c:1][H]>>[c:1][N+](=O)[O-]')
STRIP_RXNS = [
    AllChem.ReactionFromSmarts('[c:1][F,Cl,Br,I]>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1][N+](=O)[O-]>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]OC>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C(=O)O>>[c:1][H]'),
    AllChem.ReactionFromSmarts('[c:1]C(F)(F)F>>[c:1][H]'),
]

# 4 Standard Benchmark Presets
PRESETS = {
    "CT325 (Reference Active, PDB 4P8K)": "COc1ccc(CNc2nc3cc(C(F)(F)F)ccc3nc2C(=O)O)cc1",
    "Hydantoin Active (CHEMBL4459122)": "CC1(c2ccc(C#N)cc2)NC(=O)N(CC(=O)c2ccc(F)c(C(F)(F)F)c2)C1=O",
    "BTZ043 (Covalent Suicide Inhibitor)": "O=c1nc(N2CCC3(CC2)OCCO3)sc2c1cc([N+](=O)[O-])cc2C(F)(F)F",
    "CHEMBL6142943 (Measured Inactive)": "C=C(C)C(=O)Nc1cc(C(F)(F)F)cc2c(=O)nc(N3CCC4(CC3)OC[C@H](C)O4)sc12"
}


# --- SAR Mutation Helpers ---
def mutate_add_fluorine(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_F.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_add_methyl(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_ME.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_add_nitro(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    prods = RXN_ADD_NO2.RunReactants((mol_h,))
    if prods and len(prods[0]) > 0:
        res = prods[0][0]
        Chem.SanitizeMol(res)
        return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mutate_strip_substituent(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return smi
    mol_h = Chem.AddHs(mol)
    for rxn in STRIP_RXNS:
        prods = rxn.RunReactants((mol_h,))
        if prods and len(prods[0]) > 0:
            res = prods[0][0]
            Chem.SanitizeMol(res)
            return Chem.MolToSmiles(Chem.RemoveHs(res))
    return smi


def mol_to_pdbqt(mol):
    """Dynamic Meeko PDBQT string writer with backwards compatibility."""
    prep = MoleculePreparation()
    mol_setups = prep.prepare(mol)
    setup = mol_setups[0] if isinstance(mol_setups, list) else mol_setups
    if hasattr(setup, 'write_pdbqt_string'):
        res = setup.write_pdbqt_string()
        return res[0] if isinstance(res, tuple) else res
    res = meeko.PDBQTWriterLegacy.write_string(setup)
    return res[0] if isinstance(res, tuple) else res


def dock_ligand_live(smi: str, exhaustiveness: int = 8):
    """Executes live AutoDock Vina 3D docking into PDB 4P8K (Chain A + rigid FAD)."""
    t0 = time.time()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smi}")
    mol_h = Chem.AddHs(mol)

    res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if res != 0:
        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
        if res != 0:
            AllChem.EmbedMolecule(mol_h, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass

    pdbqt_str = mol_to_pdbqt(mol_h)
    temp_pdbqt = os.path.join(os.path.dirname(__file__), "..", "docking", "outputs", "temp_query.pdbqt")
    os.makedirs(os.path.dirname(temp_pdbqt), exist_ok=True)
    with open(temp_pdbqt, "w") as f:
        f.write(pdbqt_str)

    v = Vina(sf_name="vina", cpu=4, seed=42)
    if not os.path.exists(RECEPTOR_PATH):
        raise FileNotFoundError(f"Missing receptor at {RECEPTOR_PATH}")

    v.set_receptor(RECEPTOR_PATH)
    v.set_ligand_from_file(temp_pdbqt)
    v.compute_vina_maps(center=[17.07, -20.26, 1.49], box_size=[20.0, 20.0, 20.0])
    v.dock(exhaustiveness=exhaustiveness, n_poses=1)
    energy = float(v.energies(n_poses=1)[0][0])
    elapsed = time.time() - t0
    return energy, elapsed


@st.cache_resource
def load_and_train_models():
    """Loads biochemical training corpus and fits calibrated LR & RF models."""
    csv_file = os.path.join(DATA_PATH, "stratified_cluster_folds.csv")
    if not os.path.exists(csv_file):
        st.error(f"Missing training corpus at {csv_file}")
        return None, None, None, None, None

    df = pd.read_csv(csv_file)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    X_counts = []
    y = []
    actives_info = []

    for _, row in df.iterrows():
        s = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
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
        if is_act == 1:
            actives_info.append({
                "chembl_id": row["molecule_chembl_ids"],
                "smiles": s,
                "fp": fp_bit,
                "is_hydantoin": row["is_hydantoin"]
            })

    X_counts = np.vstack(X_counts)
    y = np.array(y)

    lr = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_counts, y)

    rf = RandomForestClassifier(n_estimators=300, min_samples_split=4, class_weight="balanced_subsample", random_state=42)
    rf.fit(X_counts, y)

    master_csv = os.path.join(DATA_PATH, "master_predictions_phase5.csv")
    df_master = pd.read_csv(master_csv) if os.path.exists(master_csv) else pd.DataFrame()

    return lr, rf, gen, actives_info, df_master


def check_covalent_warhead(mol):
    """Detects if molecule has an activated aromatic nitro or nitroso warhead."""
    return (
        mol.HasSubstructMatch(_PAT_NITRO)
        or mol.HasSubstructMatch(_PAT_NITROSO)
        or mol.HasSubstructMatch(_PAT_BTZ)
    )


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
    # Top Academic Banner
    st.warning("⚠️ **Academic Research Prototype**: In silico predictions do not constitute biological confirmation.")

    # Header Box
    st.markdown("""
    <div class="header-box">
        <h1>🔬 DprE1 Virtual Screening & Lead Optimization Lab</h1>
        <p>Target Flavoenzyme: <i>Mycobacterium tuberculosis</i> H37Rv DprE1 (Rv3790, UniProt P9WJF1, PDB 4P8K) | Author: <b>Shubham Chaudhary</b></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "current_smiles" not in st.session_state:
        st.session_state["current_smiles"] = PRESETS["CT325 (Reference Active, PDB 4P8K)"]
    if "docking_cache" not in st.session_state:
        st.session_state["docking_cache"] = {}

    lr_model, rf_model, fp_gen, actives_info, df_master = load_and_train_models()
    if lr_model is None:
        st.stop()

    # --- SIDEBAR: MOLECULE PLAYGROUND ---
    st.sidebar.header("🧪 Molecule Playground")

    def handle_preset_change():
        chosen = st.session_state["preset_select"]
        if chosen in PRESETS:
            st.session_state["current_smiles"] = PRESETS[chosen]

    st.sidebar.selectbox(
        "Choose a Demonstration Molecule:",
        list(PRESETS.keys()),
        key="preset_select",
        on_change=handle_preset_change
    )

    st.sidebar.markdown("#### ⚡ 1-Click SAR Mutations")
    btn_col1, btn_col2 = st.sidebar.columns(2)
    with btn_col1:
        if st.button("➕ Add -F", use_container_width=True):
            st.session_state["current_smiles"] = mutate_add_fluorine(st.session_state["current_smiles"])
            st.rerun()
        if st.button("💣 Add -NO₂", use_container_width=True):
            st.session_state["current_smiles"] = mutate_add_nitro(st.session_state["current_smiles"])
            st.rerun()

    with btn_col2:
        if st.button("➕ Add -Me", use_container_width=True):
            st.session_state["current_smiles"] = mutate_add_methyl(st.session_state["current_smiles"])
            st.rerun()
        if st.button("✂️ Strip Group", use_container_width=True):
            st.session_state["current_smiles"] = mutate_strip_substituent(st.session_state["current_smiles"])
            st.rerun()

    # Editable Text Area for current working SMILES
    def handle_smiles_edit():
        st.session_state["current_smiles"] = st.session_state["smiles_input"]

    st.sidebar.text_area(
        "Working SMILES String:",
        value=st.session_state["current_smiles"],
        key="smiles_input",
        on_change=handle_smiles_edit,
        height=110
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 Benchmark Specs (v1.1.2)
    - **Target**: DprE1 (`CHEMBL3804751`)
    - **Receptor**: PDB `4P8K` (Chain A + rigid FAD)
    - **Active Site Centroid**: `(17.07, -20.26, 1.49)` Å
    - **Live Docking**: Vina `exhaustiveness=8` (~10-15s)
    - **Pose Verification**: RMSD = 1.282 Å (< 2.0 Å)
    """)

    # Active Working SMILES
    working_smiles = st.session_state["current_smiles"].strip()
    mol = Chem.MolFromSmiles(working_smiles) if working_smiles else None

    # Check if matched in benchmark corpus
    matched_row = None
    canon_q = None
    if mol is not None and not df_master.empty:
        try:
            canon_q = Chem.CanonSmiles(working_smiles)
            for _, r in df_master.iterrows():
                if Chem.CanonSmiles(r["canonical_smiles"]) == canon_q:
                    matched_row = r
                    break
        except Exception:
            pass

    # --- TOP SECTION: STRUCTURE & PROFILE ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Chemical Topology & Drug-Likeness")
        if mol is None:
            st.error("❌ Invalid SMILES string. Please provide a chemically valid structure.")
            st.stop()

        # 2D structure image
        img = Draw.MolToImage(mol, size=(450, 260))
        st.image(img, caption="2D Chemical Topology (RDKit Depiction)", use_container_width=True)

        # Physicochemical metrics
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        lipinski_pass = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Mol. Weight", f"{mw:.1f} Da")
        mcol2.metric("cLogP", f"{logp:.2f}")
        mcol3.metric("TPSA", f"{tpsa:.1f} Å²")

        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("H-Donors (HBD)", hbd)
        mcol5.metric("H-Acceptors (HBA)", hba)
        mcol6.metric("Rotatable Bonds", rotb)

        if lipinski_pass:
            st.success("✅ **Lipinski Rule of 5**: Compliant (Oral Drug-like Space)")
        else:
            st.warning("⚠️ **Lipinski Rule of 5**: Violation detected (Bioavailability liability)")

    with col_right:
        st.subheader("2. Target Sentry & Predictive Profiling")

        # Covalent Warhead Sentry
        is_covalent = check_covalent_warhead(mol)
        if is_covalent:
            st.error("""
            🚨 **PREDICTED COVALENT WARHEAD SENTRY ALERT**:
            Detected aromatic nitro/nitroso warhead (`c[N+](=O)[O-]` or BTZ scaffold).
            **Mechanism**: Undergoes FAD-mediated reduction to a nitroso species, forming a covalent semimercaptal adduct with **Cys387**.
            **Physical Reality**: Non-covalent thermodynamic docking scoring (e.g., AutoDock Vina) is **physically invalid** for this molecule.
            """)
        else:
            st.info("🛡️ **Non-Covalent Ligand**: No electrophilic nitro/nitroso warheads detected. Reversible pocket binding mode.")

        # Chemotype Detection
        is_hyd = mol.HasSubstructMatch(_PAT_HYDANTOIN)
        if is_hyd:
            st.warning("🏷️ **Hydantoin Core Identified**: Matches `O=C1NC(=O)NC1`. Belongs to dominant literature chemotype cluster (Baseline 1 Score: 0.843).")

        # Applicability Domain
        best_sim, best_match = compute_ad_similarity(mol, fp_gen, actives_info)
        if best_sim < 0.40:
            st.warning(f"⚠️ **OUT-OF-DOMAIN WARNING**: Max Tanimoto similarity to training actives is **{best_sim:.3f}** (< 0.40 threshold). High extrapolation uncertainty.")
        else:
            st.success(f"✅ **IN-DOMAIN**: Max Tanimoto similarity to training active is **{best_sim:.3f}** (Nearest: `{best_match['chembl_id']}`).")

        # ML Inference Handling
        fp_count = fp_gen.GetCountFingerprint(mol)
        arr_cnt = np.zeros(2048, dtype=np.float32)
        for bit_id, val in fp_count.GetNonzeroElements().items():
            arr_cnt[bit_id] = val

        p_lr_live = float(lr_model.predict_proba([arr_cnt])[0, 1])
        p_rf_live = float(rf_model.predict_proba([arr_cnt])[0, 1])

        if matched_row is not None:
            st.info("ℹ️ **Benchmark Molecule**: Showing precomputed, leak-free Out-Of-Fold (OOF) cross-validation predictions.")
            p_lr_display = float(matched_row["prob_lr_cluster"])
            p_rf_display = float(matched_row["prob_rf_cluster"])
            caption_suffix = "(OOF Cluster CV)"
        else:
            st.warning("⚠️ **Novel / Mutated Structure**: Showing live predictions from models fitted on the full benchmark dataset.")
            p_lr_display = p_lr_live
            p_rf_display = p_rf_live
            caption_suffix = "(Live Model Fit)"

        st.markdown("#### 🤖 2D Machine Learning Predictions")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.metric("Logistic Regression P(Active)", f"{p_lr_display:.1%}")
            st.progress(p_lr_display)
            st.caption(f"Regularized L2 Linear Model {caption_suffix}")
        with pcol2:
            st.metric("Random Forest P(Active)", f"{p_rf_display:.1%}")
            st.progress(p_rf_display)
            st.caption(f"300-Tree Subsampled Ensemble {caption_suffix}")

    # --- SECTION 3: PLAIN-ENGLISH PRIORITIZATION VERDICT & DOCKING VAULT ---
    st.markdown("---")
    st.subheader("3. Structure-Based Docking & Prioritization Verdict")

    # Check Docking Availability
    current_docking_affinity = None
    docking_source = None
    docking_le = None
    docking_percentile_str = None

    if matched_row is not None:
        current_docking_affinity = float(matched_row["vina_affinity"])
        docking_source = "Precomputed Benchmark Vault (Phase 4/5)"
        docking_le = float(matched_row["ligand_efficiency"])

        vina_val = float(matched_row["vina_score"])
        pct = float((df_master["vina_score"] < vina_val).mean() * 100.0)
        docking_percentile_str = f"Top {100.0 - pct:.1f}% ({pct:.1f}th Pct)"

    elif canon_q in st.session_state["docking_cache"]:
        current_docking_affinity, elapsed_time = st.session_state["docking_cache"][canon_q]
        docking_source = f"Live 3D Simulation (Vina exhaustiveness=8, {elapsed_time:.1f}s)"
        docking_le = -current_docking_affinity / max(1, n_heavy)

        score_val = -current_docking_affinity
        pct = float((df_master["vina_score"] < score_val).mean() * 100.0)
        docking_percentile_str = f"Top {100.0 - pct:.1f}% ({pct:.1f}th Pct)"

    # Docking UI Controls
    dcol_left, dcol_right = st.columns([1, 1])

    with dcol_left:
        st.markdown("#### 🏗️ 3D Molecular Docking Engine")
        if current_docking_affinity is not None:
            st.success(f"🎯 **Docking Score Available**: {docking_source}")
            kcol1, kcol2, kcol3 = st.columns(3)
            kcol1.metric("Vina Affinity (ΔG)", f"{current_docking_affinity:.2f} kcal/mol")
            kcol2.metric("Percentile Rank", docking_percentile_str)
            kcol3.metric("Ligand Efficiency (LE)", f"{docking_le:.3f} kcal/mol/HA")
            st.caption(f"Receptor: PDB `4P8K` Chain A with intact rigid FAD | Heavy Atoms: {n_heavy}")
        else:
            st.info("ℹ️ **Live Docking Required**: This mutated or custom molecule has not yet been docked in the 3D binding cavity.")
            if st.button("🚀 Run Live 3D Docking (~15s)", use_container_width=True):
                with st.spinner("Running live 3D conformer embedding, Meeko PDBQT preparation, and AutoDock Vina simulation into PDB 4P8K..."):
                    try:
                        aff, elap = dock_ligand_live(working_smiles, exhaustiveness=8)
                        st.session_state["docking_cache"][canon_q] = (aff, elap)
                        st.success(f"Docking finished in {elap:.1f}s! Affinity: {aff:.2f} kcal/mol")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Docking execution failed: {e}")

    with dcol_right:
        st.markdown("#### ⚖️ Plain-English Prioritization Verdict")

        # 2D ML Classification
        p_avg = (p_lr_display + p_rf_display) / 2.0
        if p_avg >= 0.65:
            ml_verdict = ("Likely Active", "#28a745")
        elif p_avg >= 0.35:
            ml_verdict = ("Uncertain / Boundary", "#ffc107")
        else:
            ml_verdict = ("Likely Inactive", "#dc3545")

        # 3D Docking Classification
        if current_docking_affinity is not None:
            if current_docking_affinity <= -8.5:
                dock_verdict = ("Strong Pocket Fit", "#28a745")
            elif current_docking_affinity <= -7.5:
                dock_verdict = ("Moderate Fit", "#ffc107")
            else:
                dock_verdict = ("Weak Fit / Steric Clash", "#dc3545")
        else:
            dock_verdict = ("Pending Docking", "#6c757d")

        # Consensus Diagnostic
        if is_covalent:
            consensus_text = "🚨 **Covalent Suicide Inhibitor**: 2D ML active probability reflects warhead-bearing series; 3D non-covalent docking cannot evaluate covalent adduct enthalpy."
            box_bg = "#f8d7da"
        elif current_docking_affinity is None:
            consensus_text = "⏳ **Awaiting 3D Docking**: Click the button on the left to compute live 3D binding affinity and establish the consensus diagnosis."
            box_bg = "#e2e3e5"
        elif ml_verdict[0] == "Likely Active" and dock_verdict[0] == "Strong Pocket Fit":
            consensus_text = "✅ **Consensus Hit (Strong Agreement)**: Both 2D circular subgraphs and 3D pocket shape prioritize this compound. High-priority lead candidate."
            box_bg = "#d4edda"
        elif ml_verdict[0] == "Likely Inactive" and dock_verdict[0] == "Weak Fit / Steric Clash":
            consensus_text = "❌ **Consensus Inactive (Strong Agreement)**: Both 2D topological features and 3D steric scoring reject this compound. Low prioritization."
            box_bg = "#f8d7da"
        elif ml_verdict[0] == "Likely Inactive" and dock_verdict[0] == "Strong Pocket Fit":
            consensus_text = "🔍 **Discordant: Docking Rescues ML (Disagreement)**: 2D ML flags this as inactive due to lack of training congeners (novel scaffold), but 3D pocket docking detects favorable hydrophobic/steric complementarity."
            box_bg = "#fff3cd"
        elif ml_verdict[0] == "Likely Active" and dock_verdict[0] == "Weak Fit / Steric Clash":
            consensus_text = "⚠️ **Discordant: ML Rescues Docking (Disagreement)**: 2D ML detects active pharmacophores, but rigid-pocket docking penalizes the ligand due to steric clashes with rigid sidechains (Lys418/Tyr314)."
            box_bg = "#fff3cd"
        else:
            consensus_text = "⚖️ **Intermediate Profile**: Molecule shows intermediate or borderline scores across one or both modalities. Recommend secondary assay triage."
            box_bg = "#e2e3e5"

        st.markdown(f"""
        <div class="verdict-box" style="background-color: {box_bg}; border: 1px solid #ccc;">
            <p style="margin: 0; font-size: 15px; color: #111;"><b>2D Machine Learning:</b> <span style="color: {ml_verdict[1]}; font-weight: bold;">{ml_verdict[0]}</span> (P_avg = {p_avg:.1%})</p>
            <p style="margin: 4px 0; font-size: 15px; color: #111;"><b>3D Structure Docking:</b> <span style="color: {dock_verdict[1]}; font-weight: bold;">{dock_verdict[0]}</span> ({current_docking_affinity if current_docking_affinity is not None else '—'} kcal/mol)</p>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #bbb;">
            <p style="margin: 0; font-size: 14px; color: #222;">{consensus_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 4: VIRTUAL SCREENING DISAGREEMENT MAP (MATPLOTLIB SCATTER) ---
    st.markdown("---")
    st.subheader("4. Virtual Screening Disagreement Map (2D ML vs. 3D Docking)")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Plot 93 benchmark molecules
    actives_bm = df_master[df_master["label"] == "Active"]
    inactives_bm = df_master[df_master["label"] == "Inactive"]

    ax.scatter(
        actives_bm["prob_rf_cluster"],
        actives_bm["vina_affinity"],
        color="#1f77b4",
        alpha=0.75,
        s=55,
        edgecolors="#0d47a1",
        linewidth=0.6,
        label=f"Benchmark Actives (N={len(actives_bm)})"
    )

    ax.scatter(
        inactives_bm["prob_rf_cluster"],
        inactives_bm["vina_affinity"],
        color="#d62728",
        alpha=0.75,
        s=55,
        edgecolors="#b71c1c",
        linewidth=0.6,
        label=f"Benchmark Inactives (N={len(inactives_bm)})"
    )

    # Quadrant threshold lines
    ax.axvline(0.50, color="#888888", linestyle="--", linewidth=1.0)
    ax.axhline(-8.50, color="#888888", linestyle="--", linewidth=1.0)

    # Quadrant text annotations
    ax.text(0.98, -10.4, "Consensus Hits\n(High ML, Strong Docking)", ha="right", va="top", fontsize=8.5, color="#1b5e20", fontweight="bold", backgroundcolor="#ffffffbb")
    ax.text(0.02, -10.4, "Docking Rescues ML\n(Novel Scaffolds)", ha="left", va="top", fontsize=8.5, color="#e65100", fontweight="bold", backgroundcolor="#ffffffbb")
    ax.text(0.98, -7.1, "ML Rescues Docking\n(Steric Clash Traps)", ha="right", va="bottom", fontsize=8.5, color="#0d47a1", fontweight="bold", backgroundcolor="#ffffffbb")
    ax.text(0.02, -7.1, "Consensus Inactives\n(Low ML, Weak Docking)", ha="left", va="bottom", fontsize=8.5, color="#b71c1c", fontweight="bold", backgroundcolor="#ffffffbb")

    # Dynamic Gold Star Overlay for Query Molecule
    q_plot_x = p_rf_display
    if current_docking_affinity is not None:
        q_plot_y = current_docking_affinity
        ax.scatter(
            [q_plot_x],
            [q_plot_y],
            color="#ffd700",
            edgecolor="#222222",
            s=380,
            marker="*",
            zorder=10,
            label="⭐ Your Query Molecule"
        )
        ax.annotate(
            f"Your Molecule\n({q_plot_x:.1%}, {q_plot_y:.2f} kcal/mol)",
            xy=(q_plot_x, q_plot_y),
            xytext=(q_plot_x - 0.12 if q_plot_x > 0.6 else q_plot_x + 0.08, q_plot_y + 0.4),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
            fontsize=9,
            fontweight="bold",
            backgroundcolor="#ffffffcc"
        )
    else:
        ax.axvline(q_plot_x, color="#ffd700", linestyle="-.", linewidth=2.0, label="⭐ Your Query ML Position (Docking Pending)")

    ax.set_xlabel("Ligand Machine Learning Probability P(Active) [Random Forest]", fontsize=10.5, fontweight="semibold")
    ax.set_ylabel("AutoDock Vina Binding Affinity (kcal/mol)", fontsize=10.5, fontweight="semibold")
    ax.set_title("DprE1 Chemical Space: Quadrant Concordance & Disagreement Map", fontsize=12, fontweight="bold")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-10.8, -6.8])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=8.5)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # --- SECTION 5: VIVA DEFENSE ACCORDION ---
    st.markdown("---")
    with st.expander("🎓 View Viva Voce Defense Script & Scientific Findings Accordion", expanded=False):
        st.markdown(r"""
        ### Master Benchmark Performance Matrix (1,000 Bootstrap 95% CIs)
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
        #### 🏛️ Three Tough Viva Questions & Concise Model Answers:

        **Q1: Did molecular docking add predictive value over ligand-based machine learning?**
        > *"No. Under leak-free evaluation on N=93 non-covalent inhibitors, the paired 1,000-sample bootstrap 95% confidence intervals for $\Delta\text{ROC}$ cross zero across all split regimes. Docking standalone achieved an ROC-AUC of 0.610 (CI [0.463, 0.746] crosses chance), while Random Forest achieved 0.700. Hybrid rank fusion (RRF) yielded 0.697. At N=93, docking adds no statistically detectable predictive value."*

        **Q2: How did the trivial Hydantoin Detector baseline perform, and why does it matter?**
        > *"The 1-rule HydantoinDetectorClassifier baseline achieved an ROC-AUC of 0.660 [0.541, 0.758], directly beating standalone AutoDock Vina (0.610). This occurs because 33.3% of the labeled dataset originates from a single hydantoin series. In public databases, chemotype prevalence bias often dominates over uncalibrated 3D scoring functions."*

        **Q3: Why did Random Forest collapse on Leave-Hydantoin-Out while Logistic Regression generalized?**
        > *"Under cross-chemotype transfer (LHO), Random Forest collapsed to 0.527 because unpruned orthogonal decision trees cannot extrapolate to unseen circular subgraphs. Regularized Logistic Regression showed empirical resilience (0.703 [0.563, 0.824]) due to smooth linear weighting across conserved pharmacophores, though wide confidence intervals overlapping RF remind us that this is an exploratory observation in a small biochemical corpus."*
        """)

    # Bottom Disclaimer Banner
    st.markdown("---")
    st.warning("⚠️ **Academic Research Prototype**: In silico predictions do not constitute biological confirmation. Benchmark developed by **Shubham Chaudhary**.")


if __name__ == "__main__":
    main()