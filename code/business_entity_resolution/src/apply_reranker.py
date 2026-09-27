import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, distance
import lightgbm as lgb

# Base paths
BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
DATA_DIR = os.path.join(BASE_DIR, r"data\student_resource\dataset\test")
OUTPUT_DIR = os.path.join(BASE_DIR, r"code\output")
CANDIDATE_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
MODEL_PATH = os.path.join(BASE_DIR, r"code\reranker_lgb.txt")

TAU = 0.85  # Calibrated threshold for Macro F_0.5
CHUNK_SIZE = 25000  # Number of S1 lines to stream at once

def run_reranking():
    print("[1/4] Loading trained LightGBM booster...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    bst = lgb.Booster(model_file=MODEL_PATH)

    print("[2/4] Loading test source tables into memory...")
    # Load only necessary string columns
    df_s1 = pd.read_csv(
        os.path.join(DATA_DIR, "test_source1.tsv"), 
        sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]
    ).fillna("")
    
    df_s2 = pd.read_csv(
        os.path.join(DATA_DIR, "test_source2.tsv"), 
        sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]
    ).fillna("")
    
    df_s3 = pd.read_csv(
        os.path.join(DATA_DIR, "test_source3.tsv"), 
        sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]
    ).fillna("")
    
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    # Pre-lowercase strings to maximize rapidfuzz throughput
    print("      Optimizing text lookup dictionaries...")
    s1_names = dict(zip(df_s1["entity_id"], df_s1["business_name"].str.lower()))
    s1_addrs = dict(zip(df_s1["entity_id"], df_s1["business_address"].str.lower()))
    
    cand_names = dict(zip(df_cand["entity_id"], df_cand["business_name"].str.lower()))
    cand_addrs = dict(zip(df_cand["entity_id"], df_cand["business_address"].str.lower()))
    
    del df_s1, df_cand
    gc.collect()

    print("[3/4] Streaming candidate pairs and computing predictions...")
    total_written = 0
    total_matches = 0

    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f_in, open(MATCHING_PATH, "w", encoding="utf-8") as f_out:
        header = f_in.readline()
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        batch_s1_ids = []
        batch_cand_ids = []
        batch_pairs = []  # tuples of (s1_id, cand_id)

        pbar = tqdm(total=1732544, desc="Scoring Entities")

        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""

            if not cand_str.strip():
                f_out.write(f"{s1_id}\t\n")
                total_written += 1
                pbar.update(1)
                continue

            candidates = [c.strip() for c in cand_str.split(",") if c.strip()]
            for c_id in candidates:
                batch_pairs.append((s1_id, c_id))

            # When batch size is reached, compute features & predict
            if len(batch_pairs) >= 50000:
                features = []
                for s_id, c_id in batch_pairs:
                    n1 = s1_names.get(s_id, "")
                    n2 = cand_names.get(c_id, "")
                    a1 = s1_addrs.get(s_id, "")
                    a2 = cand_addrs.get(c_id, "")

                    n_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
                    n_tok = fuzz.token_set_ratio(n1, n2) / 100.0
                    
                    if a1 and a2 and a1 != "nan" and a2 != "nan":
                        a_jw = distance.JaroWinkler.normalized_similarity(a1, a2)
                        a_tok = fuzz.token_set_ratio(a1, a2) / 100.0
                    else:
                        a_jw, a_tok = 0.0, 0.0

                    features.append([n_jw, n_tok, a_jw, a_tok, 1 if c_id.startswith("S3") else 0])

                preds = bst.predict(np.array(features, dtype=np.float32))

                # Group accepted candidates by S1 entity
                entity_matches = {}
                for (s_id, c_id), prob in zip(batch_pairs, preds):
                    if s_id not in entity_matches:
                        entity_matches[s_id] = []
                    if prob >= TAU:
                        entity_matches[s_id].append(c_id)

                for s_id, m_list in entity_matches.items():
                    f_out.write(f"{s_id}\t{','.join(m_list)}\n")
                    total_written += 1
                    total_matches += len(m_list)
                    pbar.update(1)

                batch_pairs = []

        # Flush remaining pairs
        if batch_pairs:
            features = []
            for s_id, c_id in batch_pairs:
                n1 = s1_names.get(s_id, "")
                n2 = cand_names.get(c_id, "")
                a1 = s1_addrs.get(s_id, "")
                a2 = cand_addrs.get(c_id, "")

                n_jw = distance.JaroWinkler.normalized_similarity(n1, n2)
                n_tok = fuzz.token_set_ratio(n1, n2) / 100.0

                if a1 and a2 and a1 != "nan" and a2 != "nan":
                    a_jw = distance.JaroWinkler.normalized_similarity(a1, a2)
                    a_tok = fuzz.token_set_ratio(a1, a2) / 100.0
                else:
                    a_jw, a_tok = 0.0, 0.0

                features.append([n_jw, n_tok, a_jw, a_tok, 1 if c_id.startswith("S3") else 0])

            preds = bst.predict(np.array(features, dtype=np.float32))

            entity_matches = {}
            for (s_id, c_id), prob in zip(batch_pairs, preds):
                if s_id not in entity_matches:
                    entity_matches[s_id] = []
                if prob >= TAU:
                    entity_matches[s_id].append(c_id)

            for s_id, m_list in entity_matches.items():
                f_out.write(f"{s_id}\t{','.join(m_list)}\n")
                total_written += 1
                total_matches += len(m_list)
                pbar.update(1)

        pbar.close()

    print(f"\n[4/4] COMPLETE! matching_results.tsv written to {MATCHING_PATH}")
    print(f"      Total S1 entities written : {total_written:,}")
    print(f"      Total matches predicted   : {total_matches:,}")

if __name__ == "__main__":
    run_reranking()