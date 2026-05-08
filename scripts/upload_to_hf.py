import os
from huggingface_hub import HfApi

REPO_ID = "caveman273/aida-typewritten"  # change this
REPO_TYPE = "dataset"

api = HfApi(token=os.environ.get("HF_TOKEN"))

api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

upload_files = [
    ("train.parquet",      "train.parquet"),
    ("test.parquet",       "test.parquet"),
    ("validation.parquet", "validation.parquet"),
    ("README.md",                  "README.md"),
]

for local_path, repo_path in upload_files:
    if not os.path.exists(local_path):
        print(f"Skipping missing file: {local_path}")
        continue
    print(f"Uploading {local_path} -> {repo_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

print(f"Done. Dataset available at: https://huggingface.co/datasets/{REPO_ID}")
