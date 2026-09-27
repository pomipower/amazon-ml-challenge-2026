import os

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\code"
CANDIDATE_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
STRICT_MATCH_PATH = os.path.join(BASE_DIR, "output", "matching_results_strict.tsv")

print("Generating ultra-strict Top-2 baseline with correct header...")
total_matches = 0

with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(STRICT_MATCH_PATH, 'w', encoding='utf-8') as f_out:
    # Skip the incorrect input header
    next(f_in) 
    
    # Write the exact required output header
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            # Keep ONLY the Top 2 highest-confidence TF-IDF candidates
            cands = [c.strip() for c in parts[1].split(',') if c.strip()][:2]
            f_out.write(f"{s1_id}\t{','.join(cands)}\n")
            total_matches += len(cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"Done! Created {STRICT_MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")