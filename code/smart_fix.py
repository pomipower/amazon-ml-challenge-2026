import os
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
DATA_DIR = os.path.join(BASE_DIR, r"data\student_resource\dataset\test")
CANDIDATE_PATH = os.path.join(BASE_DIR, r"code\output\candidate_pairs.tsv")
SMART_MATCH_PATH = os.path.join(BASE_DIR, r"code\output\matching_results_smart.tsv")

print("Loading business names into memory...")
# Load only IDs and names to keep it fast
df_s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna("")
df_s2 = pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna("")
df_s3 = pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name"]).fillna("")

s1_names = dict(zip(df_s1["entity_id"], df_s1["business_name"].str.lower()))
cand_names = dict(zip(pd.concat([df_s2, df_s3])["entity_id"], pd.concat([df_s2, df_s3])["business_name"].str.lower()))

print("Filtering candidates with Token Set Ratio >= 75...")
total_matches = 0

with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(SMART_MATCH_PATH, 'w', encoding='utf-8') as f_out:
    next(f_in) # Skip input header
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            # Extract up to the Top 6 TF-IDF candidates
            cands = [c.strip() for c in parts[1].split(',') if c.strip()][:6]
            n1 = s1_names.get(s1_id, "")
            
            valid_cands = []
            for c in cands:
                n2 = cand_names.get(c, "")
                # Token Set Ratio ignores word order (e.g., "Cafe Paris" == "Paris Cafe")
                if fuzz.token_set_ratio(n1, n2) >= 75:
                    valid_cands.append(c)
            
            f_out.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            total_matches += len(valid_cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"\nDone! Created {SMART_MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")