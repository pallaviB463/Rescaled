
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "criteo-uplift-v2.1.csv.gz"
OUTPUT_FILE = ROOT / "data" / "processed" / "criteo_covariates_200k.csv"

FEATURES = [f"f{i}" for i in range(12)]
SAMPLE_SIZE = 200_000
CHUNK_SIZE = 250_000
SEED = 2026


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    rng = np.random.default_rng(SEED)
    best_keys = np.empty(0, dtype=np.float64)
    best_frames = []
    rows_seen = 0

    print("Reading compressed Criteo data in chunks...")
    print(f"Target sample size: {SAMPLE_SIZE:,}")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            compression="gzip",
            usecols=FEATURES,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        # Keep only rows with complete covariates.
        chunk = chunk.dropna(subset=FEATURES).reset_index(drop=True)
        if chunk.empty:
            continue

        # Random priorities give a reproducible uniform sample.
        keys = rng.random(len(chunk))
        rows_seen += len(chunk)

        combined_keys = np.concatenate([best_keys, keys])
        combined_frames = best_frames + [chunk]

        combined = pd.concat(combined_frames, ignore_index=True)
        keep_count = min(SAMPLE_SIZE, len(combined))

        keep_indices = np.argpartition(
            combined_keys, keep_count - 1
        )[:keep_count]

        best_keys = combined_keys[keep_indices]
        best_frames = [combined.iloc[keep_indices].reset_index(drop=True)]

        print(
            f"Chunk {chunk_number}: "
            f"{rows_seen:,} valid rows scanned; "
            f"{len(best_keys):,} sample candidates retained"
        )

    if len(best_keys) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Only {len(best_keys):,} valid rows found; "
            f"expected at least {SAMPLE_SIZE:,}."
        )

    sample = best_frames[0].copy()
    sample = sample[FEATURES]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_FILE, index=False)

    print("\nSampling complete.")
    print(f"Rows saved: {len(sample):,}")
    print(f"Columns saved: {len(sample.columns)}")
    print(f"Output: {OUTPUT_FILE}")
    print("\nFeature preview:")
    print(sample.head())
    print("\nMissing values per feature:")
    print(sample.isna().sum())


if __name__ == "__main__":
    main()
