import os

BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\code"
CANDIDATE_PATH = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
FINAL_MATCH_PATH = os.path.join(BASE_DIR, "output", "matching_results_final.tsv")

print("Generating final Top-3 balanced baseline...")
total_matches = 0

with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f_in, open(FINAL_MATCH_PATH, 'w', encoding='utf-8') as f_out:
    next(f_in) 
    # Exact required header
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        parts = line.strip('\n').split('\t')
        s1_id = parts[0]
        
        if len(parts) > 1 and parts[1].strip():
            # Expanding to Top 3 to match the ground truth average (~3.5)
            # This captures missing recall while letting the TF-IDF sorting protect precision.
            cands = [c.strip() for c in parts[1].split(',') if c.strip()][:3]
            f_out.write(f"{s1_id}\t{','.join(cands)}\n")
            total_matches += len(cands)
        else:
            f_out.write(f"{s1_id}\t\n")

print(f"Done! Created {FINAL_MATCH_PATH}")
print(f"Total matches predicted: {total_matches:,}")