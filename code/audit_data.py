import os
import time
from collections import Counter
import pandas as pd

BASE_DATA_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge\data\student_resource\dataset"


def count_file_stats(filepath: str, name: str):
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return None

    t0 = time.time()
    total_rows = 0
    country_counts = Counter()
    missing_names = 0
    missing_addresses = 0

    # Stream in 200,000 row chunks to avoid high RAM consumption
    chunk_iter = pd.read_csv(
        filepath,
        sep="\t",
        chunksize=200000,
        dtype=str,
        on_bad_lines="skip",
        quoting=3,  # QUOTE_NONE
    )

    for chunk in chunk_iter:
        total_rows += len(chunk)
        if "country" in chunk.columns:
            country_counts.update(chunk["country"].dropna().tolist())
        if "business_name" in chunk.columns:
            missing_names += chunk["business_name"].isna().sum()
        if "business_address" in chunk.columns:
            missing_addresses += chunk["business_address"].isna().sum()

    dt = time.time() - t0
    print(f"--- {name} ---")
    print(f"  Path: {filepath}")
    print(f"  Total records : {total_rows:,} (processed in {dt:.1f}s)")
    if country_counts:
        print(f"  Countries     : {dict(country_counts)}")
    if "business_name" in chunk.columns:
        print(f"  Missing names : {missing_names:,} ({100*missing_names/max(total_rows,1):.2f}%)")
        print(f"  Missing addrs : {missing_addresses:,} ({100*missing_addresses/max(total_rows,1):.2f}%)")
    return total_rows


def audit_ground_truth(filepath: str):
    if not os.path.exists(filepath):
        print(f"[!] Ground truth file not found: {filepath}")
        return

    t0 = time.time()
    total_s1 = 0
    singletons = 0
    total_matches = 0
    max_matches = 0
    source_distribution = Counter()

    chunk_iter = pd.read_csv(
        filepath,
        sep="\t",
        chunksize=200000,
        dtype=str,
        on_bad_lines="skip",
        quoting=3,
    )

    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            total_s1 += 1
            m = str(row.get("matched_entity_ids", "")).strip()
            if not m or m.lower() == "nan":
                singletons += 1
            else:
                ids = [x.strip() for x in m.split(",") if x.strip()]
                match_count = len(ids)
                if match_count == 0:
                    singletons += 1
                else:
                    total_matches += match_count
                    max_matches = max(max_matches, match_count)
                    for mid in ids:
                        prefix = mid[:2] if len(mid) >= 2 else "Unknown"
                        source_distribution[prefix] += 1

    dt = time.time() - t0
    print("\n--- train_ground_truth.tsv ---")
    print(f"  Total S1 entities  : {total_s1:,}")
    print(f"  Singletons (no match): {singletons:,} ({100*singletons/max(total_s1,1):.2f}%)")
    print(f"  Matched S1 entities : {total_s1 - singletons:,}")
    print(f"  Total target matches: {total_matches:,}")
    print(f"  Mean matches per S1 : {total_matches / max(total_s1, 1):.2f}")
    if total_s1 - singletons > 0:
        print(f"  Mean matches (active): {total_matches / (total_s1 - singletons):.2f}")
    print(f"  Max matches for an S1: {max_matches}")
    print(f"  Matches by source   : {dict(source_distribution)}")
    print(f"  Audit time          : {dt:.1f}s")


if __name__ == "__main__":
    print("========================================")
    print("      AMAZON ML CHALLENGE 2026 AUDIT    ")
    print("========================================")
    train_dir = os.path.join(BASE_DATA_DIR, "train")
    test_dir = os.path.join(BASE_DATA_DIR, "test")

    # 1. Audit Training Sources
    count_file_stats(os.path.join(train_dir, "train_source1.tsv"), "train_source1.tsv")
    count_file_stats(os.path.join(train_dir, "train_source2.tsv"), "train_source2.tsv")
    count_file_stats(os.path.join(train_dir, "train_source3.tsv"), "train_source3.tsv")

    # 2. Audit Ground Truth
    audit_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    # 3. Audit Test Sources
    print("\n================ TEST SET ================")
    count_file_stats(os.path.join(test_dir, "test_source1.tsv"), "test_source1.tsv")
    count_file_stats(os.path.join(test_dir, "test_source2.tsv"), "test_source2.tsv")
    count_file_stats(os.path.join(test_dir, "test_source3.tsv"), "test_source3.tsv")