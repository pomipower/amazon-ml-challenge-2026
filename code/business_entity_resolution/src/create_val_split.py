"""
create_val_split.py — Generate a fast, leak-free local validation benchmark
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\data\student_resource\dataset\train"
VAL_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\data\val_sample"

SAMPLE_S1_SIZE = 50000  # 50k S1 records provides high statistical confidence
RANDOM_STATE = 42

def create_validation_split():
    os.makedirs(VAL_DIR, exist_ok=True)
    print(f"Loading ground truth from {DATA_DIR}...")
    gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    
    print(f"Total training ground truth rows: {len(gt_df):,}")
    
    # Sample S1 IDs
    val_gt = gt_df.sample(n=min(SAMPLE_S1_SIZE, len(gt_df)), random_state=RANDOM_STATE).copy()
    val_s1_ids = set(val_gt["source1_entity_id"])
    
    # Identify all true matching S2 and S3 IDs for these S1 records
    all_matched_ids = set()
    for ids_str in val_gt["matched_entity_ids"]:
        if ids_str.strip():
            all_matched_ids.update(ids_str.split(","))
            
    print(f"Sampled {len(val_s1_ids):,} S1 entities containing {len(all_matched_ids):,} true matched targets.")
    
    # Save sampled ground truth
    val_gt.to_csv(os.path.join(VAL_DIR, "val_ground_truth.tsv"), sep="\t", index=False)

    # 1. Filter S1
    print("Extracting validation Source 1 records...")
    s1_path = os.path.join(DATA_DIR, "train_source1.tsv")
    s1_reader = pd.read_csv(s1_path, sep="\t", chunksize=200000, dtype=str)
    val_s1_list = []
    for chunk in s1_reader:
        sub = chunk[chunk["entity_id"].isin(val_s1_ids)]
        if not sub.empty:
            val_s1_list.append(sub)
    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    val_s1_df.to_csv(os.path.join(VAL_DIR, "val_source1.tsv"), sep="\t", index=False)
    print(f"Saved {len(val_s1_df):,} val_source1 rows.")

    # 2. Extract S2 and S3: must include ALL true matches + a background sample for realistic blocking
    for src_num in:
        src_name = f"train_source{src_num}.tsv"
        out_name = f"val_source{src_num}.tsv"
        src_path = os.path.join(DATA_DIR, src_name)
        print(f"Sampling {src_name}...")
        
        matches_found = []
        background_sample = []
        
        for chunk in pd.read_csv(src_path, sep="\t", chunksize=200000, dtype=str):
            # Keep all mandatory positive matches
            positives = chunk[chunk["entity_id"].isin(all_matched_ids)]
            if not positives.empty:
                matches_found.append(positives)
            
            # Keep 5% background distractors to simulate large search space
            distractors = chunk[~chunk["entity_id"].isin(all_matched_ids)].sample(frac=0.05, random_state=RANDOM_STATE)
            if not distractors.empty:
                background_sample.append(distractors)

        val_src_df = pd.concat(matches_found + background_sample, ignore_index=True).drop_duplicates(subset=["entity_id"])
        val_src_df.to_csv(os.path.join(VAL_DIR, out_name), sep="\t", index=False)
        print(f"Saved {len(val_src_df):,} {out_name} records.")

    print("\nValidation benchmark successfully generated in:", VAL_DIR)

if __name__ == "__main__":
    create_validation_split()