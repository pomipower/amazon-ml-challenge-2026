import os
import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
DATA_DIR = os.path.join(BASE_DIR, r"data\student_resource\dataset\test")
CANDIDATE_PATH = os.path.join(BASE_DIR, r"code\output\candidate_pairs.tsv")
ULTIMATE_MATCH_PATH = os.path.join(BASE_DIR, r"code\output\matching_results_ultimate.tsv")

print("[1/3] Loading Names and Addresses...")
df_s1 = pd.read_csv(os.path.join(DATA_DIR, "test_source1.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna("")
df_s2 = pd.read_csv(os.path.join(DATA_DIR, "test_source2.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna("")
df_s3 = pd.read_csv(os.path.join(DATA_DIR, "test_source3.tsv"), sep="\t", dtype=str, quoting=3, usecols=["entity_id", "business_name", "business_address"]).fillna("")

df_cand = pd.concat([df_s2, df_s3])
s1_dict = df_s1.set_index("entity_id").to_dict("index")
cand_dict = df_cand.set_index("entity_id").to_dict("index")

# Free up RAM
del df_s1, df_s2, df_s3, df_cand

print("[2/3] Applying Surgical Dual-Field Filter...")
total_matches = 0

with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(ULTIMATE_MATCH_PATH, 'w', encoding='utf-8') as f_out:
    next(f_in) 
    # Ensure exact column headers as per guidelines
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            # Evaluate the Top 5 candidates from the TF-IDF output
            cands = [c.strip() for c in parts[1].split(',') if c.strip()][:5]
            
            s1_data = s1_dict.get(s1_id, {})
            n1 = str(s1_data.get("business_name", "")).lower()
            a1 = str(s1_data.get("business_address", "")).lower()
            
            valid_cands = []
            for c in cands:
                c_data = cand_dict.get(c, {})
                n2 = str(c_data.get("business_name", "")).lower()
                a2 = str(c_data.get("business_address", "")).lower()
                
                # Token Set Ratio ignores word order ("Cafe Paris" == "Paris Cafe")
                name_sim = fuzz.token_set_ratio(n1, n2)
                
                # Rule 1: Both addresses exist
                if a1 and a2 and a1 != "nan" and a2 != "nan":
                    addr_sim = fuzz.token_set_ratio(a1, a2)
                    if name_sim >= 85 and addr_sim >= 75:
                        valid_cands.append(c)
                # Rule 2: Missing Address fallback
                else:
                    if name_sim >= 95:
                        valid_cands.append(c)
            
            f_out.write(f"{s1_id}\t{','.join(valid_cands)}\n")
            total_matches += len(valid_cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"\n[3/3] Done! Created {ULTIMATE_MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")