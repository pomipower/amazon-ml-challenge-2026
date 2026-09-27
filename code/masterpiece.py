import os
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
DATA_DIR = os.path.join(BASE_DIR, r"data\student_resource\dataset\test")
CANDIDATE_PATH = os.path.join(BASE_DIR, r"code\output\candidate_pairs.tsv")
FINAL_MATCH_PATH = os.path.join(BASE_DIR, r"code\output\matching_results_masterpiece.tsv")

print("Loading data for the final run...")
df_s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna("")
df_cand = pd.concat([
    pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna(""),
    pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna("")
])

s1_dict = df_s1.set_index("entity_id").to_dict("index")
cand_dict = df_cand.set_index("entity_id").to_dict("index")
del df_s1, df_cand

print("Applying Rank-Aware Escalating Thresholds...")
total_matches = 0

with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(FINAL_MATCH_PATH, 'w', encoding='utf-8') as f_out:
    next(f_in)
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            # Grab top 5 candidates from the TF-IDF output
            cands = [c.strip() for c in parts[1].split(',') if c.strip()][:5]
            
            s1_data = s1_dict.get(s1_id, {})
            n1 = str(s1_data.get("business_name", "")).lower()
            a1 = str(s1_data.get("business_address", "")).lower()
            
            valid_cands = []
            for i, c in enumerate(cands):
                c_data = cand_dict.get(c, {})
                n2 = str(c_data.get("business_name", "")).lower()
                name_sim = fuzz.token_set_ratio(n1, n2)
                
                # Escalate strictness based on TF-IDF Rank
                if i == 0:
                    a2 = str(c_data.get("business_address", "")).lower()
                    addr_sim = fuzz.token_set_ratio(a1, a2) if a1 and a2 else 0
                    if name_sim >= 40 or addr_sim >= 50:
                        valid_cands.append(c)
                elif i == 1:
                    if name_sim >= 60: valid_cands.append(c)
                elif i == 2:
                    if name_sim >= 75: valid_cands.append(c)
                elif i == 3:
                    if name_sim >= 85: valid_cands.append(c)
                elif i == 4:
                    if name_sim >= 92: valid_cands.append(c)
            
            f_out.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            total_matches += len(valid_cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"\nDone! Created {FINAL_MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")