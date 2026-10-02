#!/usr/bin/env python3
"""
src/audit_chembl.py

Extracts bioactivity records from ChEMBL for the verified DprE1 target CHEMBL3804751.
Separates data into Tier A (biochemical/binding assays measuring direct DprE1 inhibition)
and Tier B (whole-cell phenotypic assays).
Calculates summary statistics: raw record count, unique SMILES count, standard_type distribution,
and relation distribution (=, >, <).
Saves raw records to data/raw/chembl_dpre1_raw.csv.
"""

import os
import sys
import time
import requests
import pandas as pd
from chembl_webresource_client.new_client import new_client


def fetch_all_activities(target_chembl_id: str, limit_per_page: int = 1000) -> pd.DataFrame:
    """
    Fetch all bioactivity records for a target using ChEMBL REST pagination.
    """
    print(f"\n[INFO] Fetching bioactivity records for {target_chembl_id}...")
    base_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}&limit={limit_per_page}"
    url = base_url
    records = []
    page = 1
    t0 = time.time()
    
    while url:
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code != 200:
                print(f"[ERROR] HTTP {resp.status_code} on page {page}, retrying in 3s...", file=sys.stderr)
                time.sleep(3)
                resp = requests.get(url, timeout=45)
                if resp.status_code != 200:
                    print(f"[FATAL] Failed on page {page}. Exiting fetch.", file=sys.stderr)
                    break
                    
            data = resp.json()
            batch = data.get("activities", [])
            records.extend(batch)
            total = data.get("page_meta", {}).get("total_count", 0)
            next_url = data.get("page_meta", {}).get("next")
            url = f"https://www.ebi.ac.uk{next_url}" if next_url else None
            
            elapsed = time.time() - t0
            print(f"  Page {page:02d}: Retrieved {len(batch)} records (Cumulative: {len(records)}/{total}) [{elapsed:.1f}s]", flush=True)
            page += 1
        except Exception as e:
            print(f"[ERROR] Exception during fetch on page {page}: {e}", file=sys.stderr)
            time.sleep(2)
            
    df = pd.DataFrame(records)
    print(f"[INFO] Successfully retrieved {len(df)} records for {target_chembl_id} in {time.time() - t0:.1f}s.")
    return df


def compute_summary_stats(df: pd.DataFrame, target_id: str, label: str):
    """
    Compute and print summary statistics for the dataset, separated by Tier A and Tier B.
    """
    print("=" * 70)
    print(f"BIOACTIVITY AUDIT SUMMARY: {target_id} ({label})")
    print("=" * 70)
    
    total_records = len(df)
    unique_smiles = df["canonical_smiles"].dropna().nunique()
    print(f"Total Raw Bioactivity Records: {total_records}")
    print(f"Unique Canonical SMILES:       {unique_smiles}")
    
    # Standard type distribution
    print("\n--- Standard Type Distribution (Top 10) ---")
    st_dist = df["standard_type"].value_counts(dropna=False).head(10)
    for st, count in st_dist.items():
        print(f"  {str(st):<20}: {count:>6} ({count/total_records*100:5.1f}%)")
        
    # Relation distribution
    print("\n--- Standard Relation Distribution ---")
    rel_dist = df["standard_relation"].value_counts(dropna=False)
    for rel, count in rel_dist.items():
        rel_str = str(rel) if pd.notna(rel) else "None/Null"
        print(f"  {rel_str:<10}: {count:>6} ({count/total_records*100:5.1f}%)")
        
    # Assay Type breakdown
    print("\n--- Assay Type Distribution ---")
    at_dist = df["assay_type"].value_counts(dropna=False)
    for at, count in at_dist.items():
        print(f"  Type {str(at):<5}: {count:>6} ({count/total_records*100:5.1f}%)")

    # Tier A: assay_type == 'B' measuring IC50 or Ki
    df_tier_a = df[(df["assay_type"] == "B") & (df["standard_type"].isin(["IC50", "Ki"]))].copy()
    # Tier B: assay_type == 'F' (whole-cell phenotypic)
    df_tier_b = df[df["assay_type"] == "F"].copy()
    
    print("\n--- Tier Breakdown ---")
    print(f"Tier A (Biochemical IC50/Ki, assay_type == 'B'): {len(df_tier_a)} records")
    print(f"  - Unique SMILES in Tier A: {df_tier_a['canonical_smiles'].dropna().nunique()}")
    print(f"  - Standard types: {dict(df_tier_a['standard_type'].value_counts())}")
    print(f"  - Relations:     {dict(df_tier_a['standard_relation'].value_counts(dropna=False))}")
    
    print(f"\nTier B (Phenotypic whole-cell, assay_type == 'F'): {len(df_tier_b)} records")
    print(f"  - Unique SMILES in Tier B: {df_tier_b['canonical_smiles'].dropna().nunique()}")
    print("=" * 70)


def main():
    target_id_dpre1 = "CHEMBL3804751"
    
    # Verify target metadata via chembl_webresource_client
    target_api = new_client.target
    try:
        t_meta = target_api.get(target_id_dpre1)
        print(f"[ChEMBL Client] Verified Target {target_id_dpre1}: {t_meta.get('pref_name')} ({t_meta.get('organism')})")
    except Exception as e:
        print(f"[Warning] Could not get metadata for {target_id_dpre1}: {e}")

    # Fetch genuine DprE1 records
    df = fetch_all_activities(target_id_dpre1)
    
    out_dir = os.path.abspath("data/raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "chembl_dpre1_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Saved {len(df)} verified DprE1 raw records to: {out_path}")
        
    compute_summary_stats(df, target_id_dpre1, "M. tuberculosis DprE1 / Rv3790")


if __name__ == "__main__":
    main()
