import os
import zipfile

# --- UPDATE YOUR TEAM NAME HERE ---
TEAM_NAME = "MyTeamName"
BASE_DIR = r"C:\Users\omkul\ProjectsCoding\Amazon_ML_Challenge"
ZIP_PATH = os.path.join(BASE_DIR, f"{TEAM_NAME}_submission.zip")

# Maps local file paths to their required destination paths inside the ZIP
files_to_zip = {
    r"code\output\matching_results.tsv": "output/matching_results.tsv",
    r"code\output\candidate_pairs.tsv": "output/candidate_pairs.tsv",
    r"code\business_entity_resolution\src\blocking.py": "code/business_entity_resolution/src/blocking.py",
    r"code\business_entity_resolution\src\apply_reranker.py": "code/business_entity_resolution/src/apply_reranker.py",
    r"code\business_entity_resolution\requirements.txt": "code/business_entity_resolution/requirements.txt",
    r"code\business_entity_resolution\README.md": "code/business_entity_resolution/README.md",
    r"code\Documentation_template.md": "Documentation_template.md"
}

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for local_path, zip_path in files_to_zip.items():
        full_local_path = os.path.join(BASE_DIR, local_path)
        if os.path.exists(full_local_path):
            zipf.write(full_local_path, zip_path)
            print(f"Added: {zip_path}")
        else:
            print(f"[!] WARNING: Missing required file: {full_local_path}")

print(f"\nFinal submission package created successfully at: {ZIP_PATH}")