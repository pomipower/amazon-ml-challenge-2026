# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains my solution for the Amazon ML Challenge 2026, which involved resolving business identities across independent, noisy data sources without common identifiers. 

The primary engineering challenge was scaling the $O(N \times M)$ search space (1.7 million queries against 10 million targets) strictly on local hardware (16GB RAM) within a 24-hour hackathon timeframe.

## 🎯 Approach & Architecture

To avoid Out-Of-Memory (OOM) crashes and bypass the need for deep learning embeddings, I implemented a highly memory-efficient **Two-Stage Pipeline**:

### 1. Stage One: Sparse Lexical Blocking (The Bouncer)
* **Strategy:** Concatenated and normalized business names and addresses (stripping legal suffixes and geographical stopwords). 
* **Execution:** Built a chunked Word-Level TF-IDF vectorizer (`max_df=0.05` to ignore common terms). Ran sparse matrix multiplication (Cosine Similarity) in batches of 2,000 to keep RAM utilization under 5GB.
* **Result:** Successfully reduced the 17-trillion comparison search space to a maximum of 15 highly probable candidates per S1 entity.

### 2. Stage Two: Surgical Dual-Field Filtering (The Inspector)
* **Strategy:** The competition evaluated via Macro $F_{0.5}$, heavily penalizing false positive merges (especially on singletons). 
* **Execution:** Used `rapidfuzz` for pairwise string comparisons. Evaluated the TF-IDF candidate shortlist using a **Rank-Aware Escalating Threshold**:
  * Candidates ranked #1 by TF-IDF were given lenient string-similarity thresholds.
  * Candidates ranked #2 through #5 faced increasingly strict thresholding (up to 92% required similarity) across both Name and Address fields to aggressively prune false positives.

## 🏆 Results
* **Precision-Optimized Baseline:** Achieved a competitive **0.660 Macro $F_{0.5}$** score on the live leaderboard.
* **Pipeline Efficiency:** Processed over 17 million rows end-to-end on a local 16GB laptop in under 4 hours.

## 🛠️ Tech Stack
* **Core:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-Learn (TF-IDF, Sparse Matrices), LightGBM
* **String Matching:** RapidFuzz (Jaro-Winkler, Token Set Ratio)

## 📁 Repository Structure
* `/code/business_entity_resolution/src/` - Core pipeline scripts (Blocking, Reranking, Heuristic Filters).
* `/code/audit_data.py` - Streaming dataset profiler to extract metrics without loading data into RAM.
* `/code/masterpiece.py` - The final rank-aware rapidfuzz filtering script.