import os
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

TRAIN_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\data\student_resource\dataset\train"
TEST_DIR  = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\data\student_resource\dataset\test"
OUTPUT_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\code\output"

TOP_K_CANDIDATES = 15
# Drastically reduced chunk size to prevent RAM overflow during dot product
CHUNK_SIZE = 1000 

def normalize_text(text: str) -> str:
    """Fast basic text normalization."""
    if pd.isna(text) or not text:
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # We strip common suffixes/words entirely to keep the sparse matrix clean
    stopwords = r'\b(corporation|private|limited|company|street|road|avenue|boulevard|corp|pvt|ltd|co|st|rd|ave|blvd|india|us|france|inc|llc)\b'
    text = re.sub(stopwords, ' ', text)
        
    return re.sub(r'\s+', ' ', text).strip()

def load_and_preprocess_country(dir_path: str, country: str, is_test: bool = True) -> tuple:
    print(f"\n[+] Loading Data for Country: {country.upper()}")
    prefix = "test" if is_test else "train"
    
    dfs = []
    for source in ['source1', 'source2', 'source3']:
        path = os.path.join(dir_path, f"{prefix}_{source}.tsv")
        chunks = []
        for chunk in pd.read_csv(path, sep='\t', dtype=str, on_bad_lines='skip', quoting=3, chunksize=250_000):
            chunk['country'] = chunk['country'].fillna('Unknown')
            # Exact country match
            country_chunk = chunk[chunk['country'].str.lower() == country.lower()].copy()
            chunks.append(country_chunk)
            
        df = pd.concat(chunks, ignore_index=True)
        # Combine text and normalize
        df['search_text'] = (df['business_name'].fillna('') + " " + df['business_address'].fillna('')).apply(normalize_text)
        dfs.append(df)
        
    df_s1, df_s2, df_s3 = dfs
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    print(f"    -> S1 Entities: {len(df_s1):,}")
    print(f"    -> S2+S3 Targets: {len(df_s23):,}")
    return df_s1, df_s23

def run_blocking_for_country(df_s1: pd.DataFrame, df_s23: pd.DataFrame) -> Dict[str, List[str]]:
    if len(df_s1) == 0 or len(df_s23) == 0:
        return {}

    print("    -> Fitting Word TF-IDF Vectorizer...")
    # Switched to 'word' analyzer. max_df=0.05 ignores words appearing in >5% of documents (stopwords).
    # This keeps the TF-IDF matrix incredibly sparse and lighting fast.
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 1), 
        min_df=2, 
        max_df=0.05, 
        dtype=np.float32
    )
    
    # Fit only on the target corpus to save time
    X_target = vectorizer.fit_transform(df_s23['search_text'])
    target_ids = df_s23['entity_id'].values
    
    candidate_dict = {}
    print(f"    -> Retrieving Top {TOP_K_CANDIDATES} matches per S1 entity in chunks of {CHUNK_SIZE}...")
    
    for i in tqdm(range(0, len(df_s1), CHUNK_SIZE)):
        chunk_s1 = df_s1.iloc[i : i + CHUNK_SIZE]
        X_query = vectorizer.transform(chunk_s1['search_text'])
        
        # Sparse Matrix Multiplication (1000 x 4.7 million)
        similarity_matrix = X_query.dot(X_target.T)
        
        for row_idx in range(similarity_matrix.shape[0]):
            s1_id = chunk_s1.iloc[row_idx]['entity_id']
            row = similarity_matrix.getrow(row_idx)
            
            if row.nnz == 0:
                candidate_dict[s1_id] = []
                continue
                
            col_indices = row.indices
            scores = row.data
            
            # Fast Top-K extraction
            if len(scores) > TOP_K_CANDIDATES:
                top_indices_unsorted = np.argpartition(scores, -TOP_K_CANDIDATES)[-TOP_K_CANDIDATES:]
                top_indices_sorted = top_indices_unsorted[np.argsort(-scores[top_indices_unsorted])]
            else:
                top_indices_sorted = np.argsort(-scores)
                
            best_target_indices = col_indices[top_indices_sorted]
            candidate_dict[s1_id] = target_ids[best_target_indices].tolist()
            
    return candidate_dict

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_candidates = {}
    
    # France first, as it's smallest, so you see it complete quickly
    for country in ['France', 'US', 'India']:
        df_s1, df_s23 = load_and_preprocess_country(TEST_DIR, country, is_test=True)
        candidates = run_blocking_for_country(df_s1, df_s23)
        all_candidates.update(candidates)
        
    print("\n[+] Writing candidate_pairs.tsv...")
    out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, matches in all_candidates.items():
            matches_str = ",".join(matches)
            f.write(f"{s1_id}\t{matches_str}\n")
            
    print(f"\n[+] Output saved to {out_path}.")
    print("Run: python ../utils/validate_submission.py --candidate output/candidate_pairs.tsv --test-dir ../data/student_resource/dataset/test")

if __name__ == "__main__":
    main()