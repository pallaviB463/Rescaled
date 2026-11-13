# Rescaled Project Datasets & Benchmarks
**Deliverable 3: Dataset (Link / Folder) — Advanced Data Analytics (ADA-2026)**  
**Team Rescaled:** Pallavi B (PES1UG24AM906), Khushi Patel (PES1UG24AM134), N Lakshanya (PES1UG24AM169), Porla Neha (PES1UG25AM808) | **Faculty:** Prof. Bhaskarjyoti Das

---

## 1. Overview of Project Datasets
This repository evaluates causal uplift estimators and transformation scale-dependence across three tiers of empirical and synthetic data:

| Dataset | Type | Sample Size ($N$) | Zero-Rate | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Hillstrom MineThatData** | Real Field Trial | 64,000 | 99.1% | Real-world policy value evaluation and commercial turnover measurement |
| **Criteo Uplift v2.1** | Semi-Synthetic Benchmark | 200,000 | ~85–90% | High-dimensional ad-tech covariates ($X_1 \dots X_{12}$) with controlled margin DGP |
| **Synthetic & Oracle Grid** | Simulated Two-Margin | 20,000 / rep | 70%, 90%, 98% | Exact ground-truth CATE ranking and mathematical asymptotic validation |

---

## 2. Dataset Descriptions & Schemas

### Dataset 1: Hillstrom MineThatData Email Marketing Experiment (`Hillstrom.csv`)
* **Source:** Kevin Hillstrom, MineThatData Email Analytics Challenge.
* **Format:** Comma-Separated Values (`Hillstrom.csv`, 4.09 MB). Included directly in this folder.
* **Public Mirror Link:** `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
* **Experimental Design:**
  * Genuine randomized A/B trial with 64,000 individual retail customers.
  * Treatment arms:
    * `Mens E-Mail`: 21,307 customers
    * `Womens E-Mail`: 21,300 customers
    * `No E-Mail` (Control): 21,307 customers
* **Outcome Variables:**
  * `visit`: Binary website visit indicator (overall 14.7%).
  * `conversion`: Binary purchase indicator (overall 0.90%).
  * `spend`: Dollar expenditure within two-week post-campaign window (**99.1% zeros**, positive spend ranges from \$29.99 to \$499.00+).
* **Feature Schema:**
  * `recency`: Months since last purchase.
  * `history`: Historical dollar spend amount.
  * `mens`: Indicator of historical menswear purchase.
  * `womens`: Indicator of historical womenswear purchase.
  * `zip_code`: Categorical urban / suburban / rural classification.
  * `newbie`: Indicator for new customer registration in past 12 months.
  * `channel`: Acquisition channel (`Phone`, `Web`, `Multichannel`).

---

### Dataset 2: Criteo Uplift v2.1 Semi-Synthetic Benchmark (`criteo_sample_200k.parquet`)
* **Source:** Criteo AI Lab Uplift v2.1 Benchmark Dataset (~13.98 Million instances).
* **Sampled File:** `criteo_sample_200k.parquet` (3.67 MB, included directly in this folder).
* **Processed Semi-Synthetic File:** `criteo_semisynthetic_200k.csv` (106 MB).
  * *Google Drive / Cloud Download Link:* Stored in team Google Drive folder under `3_Dataset/processed/`.
* **Public Raw Source Link:** `http://ailab.criteo.com/criteo-uplift-prediction-dataset/` or Zenodo repository `https://zenodo.org/record/4650385`.
* **Feature Schema:**
  * 12 anonymized continuous advertising features ($X_1 \dots X_{12}$) standardized to zero mean and unit variance.
  * `treatment`: Binary randomized ad impression assignment ($W \in \{0, 1\}$).
  * `synthetic_spend`: Controlled zero-heavy spend outcome generated via calibrated two-margin DGP with ground-truth Level CATE and Conversion Probability CATE.
  * Reproducibility manifest: `criteo_semisynthetic_manifest.json` (random seed = 42).

---

### Dataset 3: Synthetic Factorial Simulation Grid Data
* **Generation Engine:** Implemented in `src/dgp.py` and demonstrated in `2_Code_ipynb/Rescaled_CATE_Uplift_Master.ipynb`.
* **Factorial Design:**
  * 6 Causal Scenarios: `Null`, `Homogeneous`, `Extensive`, `Intensive`, `Aligned`, `Opposed`.
  * 3 Zero-Rate Regimes: $p_0 \in \{0.70, 0.90, 0.98\}$.
  * 7 Scale Multipliers: $a \in \{0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0\}$.
  * Known ground-truth treatment effects computed directly from true potential outcomes.

---

## 3. How to Access and Load the Datasets in Python

```python
import pandas as pd

# Load Hillstrom Trial
df_hillstrom = pd.read_csv("Hillstrom.csv")
print(f"Hillstrom Loaded: {df_hillstrom.shape[0]:,} rows, {df_hillstrom.shape[1]} columns")

# Load Criteo Sample
df_criteo = pd.read_parquet("criteo_sample_200k.parquet")
print(f"Criteo Sample Loaded: {df_criteo.shape[0]:,} rows, {df_criteo.shape[1]} columns")
```

---
*Deliverable 3 of 6 | Team Rescaled | Advanced Data Analytics (ADA-2026)*

