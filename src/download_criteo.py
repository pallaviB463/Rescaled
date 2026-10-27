"""download_criteo.py (OWNER: Member Q).
Downloads the Criteo Uplift v2.1 dataset from the official Hugging Face mirror,
verifies row counts and treatment shares, and extracts a 200k stratified subsample
for the semi-synthetic benchmark (task P12).
"""
import os
import urllib.request
import pandas as pd
import numpy as np

CRITEO_URL = "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
GZ_PATH = os.path.join(DATA_DIR, "criteo-research-uplift-v2.1.csv.gz")
SAMPLE_PATH = os.path.join(DATA_DIR, "criteo_sample_200k.parquet")


def download_criteo():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(GZ_PATH):
        print(f"Downloading Criteo v2.1 (~297 MB) to {GZ_PATH}...")
        req = urllib.request.Request(CRITEO_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(GZ_PATH, "wb") as out_file:
            chunk_size = 1024 * 1024 * 8  # 8MB chunks
            downloaded = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                print(f"  Downloaded {downloaded / (1024 * 1024):.1f} MB...", flush=True)
        print("Download complete.")
    else:
        print(f"Archive already exists at {GZ_PATH}.")


def extract_sample_and_stats(sample_size=200_000, seed=42):
    print("Reading Criteo dataset to compute exact stats and extract stratified sample...")
    # Criteo columns: f0-f11, treatment, conversion, visit, exposure
    chunk_size = 500_000
    total_rows = 0
    treated_rows = 0
    visit_rows = 0
    conv_rows = 0

    chunks_for_sample = []
    # Sample fraction: roughly sample_size / 13.98M
    sample_fraction = (sample_size * 1.2) / 13_979_592

    rng = np.random.default_rng(seed)

    for chunk in pd.read_csv(GZ_PATH, chunksize=chunk_size):
        n_chunk = len(chunk)
        total_rows += n_chunk
        treated_rows += int(chunk["treatment"].sum())
        visit_rows += int(chunk["visit"].sum())
        conv_rows += int(chunk["conversion"].sum())

        # Subsample rows
        mask = rng.random(n_chunk) < sample_fraction
        chunks_for_sample.append(chunk[mask])

    print(f"Exact total rows: {total_rows:,}")
    print(f"Treated rows: {treated_rows:,} ({treated_rows / total_rows * 100:.2f}%)")
    print(f"Visit rows: {visit_rows:,} ({visit_rows / total_rows * 100:.2f}%)")
    print(f"Conversion rows: {conv_rows:,} ({conv_rows / total_rows * 100:.2f}%)")

    # Combine subsample and stratify to exact sample_size
    sampled_df = pd.concat(chunks_for_sample, ignore_index=True)
    # Stratified exact subsample
    e_overall = treated_rows / total_rows
    n_treated_target = int(round(sample_size * e_overall))
    n_control_target = sample_size - n_treated_target

    treated_sub = sampled_df[sampled_df["treatment"] == 1].sample(n=n_treated_target, random_state=seed)
    control_sub = sampled_df[sampled_df["treatment"] == 0].sample(n=n_control_target, random_state=seed)
    final_sample = pd.concat([treated_sub, control_sub], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    final_sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"Saved {len(final_sample):,} rows stratified sample to {SAMPLE_PATH}")
    return total_rows, treated_rows, visit_rows, conv_rows


if __name__ == "__main__":
    download_criteo()
    extract_sample_and_stats()
