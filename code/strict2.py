import os
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
DATA_DIR = os.path.join(BASE_DIR, r"data\student_resource\dataset\test")
CANDIDATE_PATH = os.path.join(BASE_DIR, r"code\output\candidate_pairs.tsv")
MATCH_PATH = os.path.join(BASE_DIR, r"code\output\matching_results_strict_v2.tsv")

print("Loading names for safety check...")
df_s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna("")
df_cand = pd.concat([
    pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna(""),
    pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna("")
])

s1_names = dict(zip(df_s1["entity_id"], df_s1["business_name"].str.lower()))
cand_names = dict(zip(df_cand["entity_id"], df_cand["business_name"].str.lower()))

total_matches = 0
with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(MATCH_PATH, 'w', encoding='utf-8') as f_out:
    next(f_in)
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            cands = [c.strip() for c in parts[1].split(',') if c.strip()]
            valid_cands = []
            n1 = s1_names.get(s1_id, "")
            
            # Always accept Rank 1 (TF-IDF's most confident guess)
            if len(cands) > 0:
                valid_cands.append(cands[0])
            
            # Accept Rank 2 only if name similarity is >= 85%
            if len(cands) > 1:
                n2 = cand_names.get(cands[1], "")
                if fuzz.token_set_ratio(n1, n2) >= 85:
                    valid_cands.append(cands[1])
                    
            f_out.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            total_matches += len(valid_cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"Done! Created {MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")