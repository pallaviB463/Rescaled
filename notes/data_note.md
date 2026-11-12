# Data Note: Hillstrom and Criteo Benchmarks (Task Q2)

**Owner:** Member Q  
**Project:** Does the Way We Transform Outcomes Change Who We Target?  
**Date:** October 2026  

This document records the empirical characteristics, provenance, verification checks, and usage protocols for the two real-world benchmarks used in this project: the **Kevin Hillstrom MineThatData E-Mail Analytics Challenge** dataset and the **Criteo Uplift Prediction v2.1** benchmark.

---

## 1. Hillstrom MineThatData E-Mail Analytics Dataset

### 1.1 Overview & Provenance
- **Source:** Kevin Hillstrom, *MineThatData E-Mail Analytics Challenge* (2008), accessed via `scikit-uplift` (`sklift.datasets.fetch_hillstrom`) and mirrored locally at [`data/Hillstrom.csv`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/data/Hillstrom.csv).
- **Study Design:** A randomized marketing field experiment tracking 64,000 customers for two weeks following an email campaign.
- **Randomization:** 3 experimental arms with approximately equal allocation (~33.3% each).
- **Evaluated Arms:**
  1. `Mens E-Mail`: 21,307 customers (33.29%)
  2. `Womens E-Mail`: 21,387 customers (33.42%)
  3. `No E-Mail` (Control): 21,306 customers (33.29%)
  - **Total Rows:** Exactly **64,000**.

### 1.2 Binary Evaluation Frameworks
To maintain clean causal contrasts, evaluation is partitioned into two distinct binary experiments against the shared control arm:
- **Primary Comparison (`mens`):** `Mens E-Mail` vs `No E-Mail`
  - Total Sample $N = 42,613$
  - Treatment Probability $e = \frac{21,307}{42,613} = 0.50001 \approx 0.500$
- **Replicate Comparison (`womens`):** `Womens E-Mail` vs `No E-Mail`
  - Total Sample $N = 42,693$
  - Treatment Probability $e = \frac{21,387}{42,693} = 0.50095 \approx 0.501$

### 1.3 Arm-Level Outcome Summary & Zero-Heaviness
Spend is extreme zero-heavy: across all 64,000 customers, only **578** customers purchased ($0.903\%$, or $\approx 1\%$), leaving **$99.1\%$ exact zeros**.

| Arm | $N$ | Buyers ($Y > 0$) | Buyer Rate ($\%$) | Visit Rate ($\%$) | Total Spend (\$) | Mean Spend (Overall) | Mean Spend ($Y > 0$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mens E-Mail** | 21,307 | 267 | 1.253% | 18.28% | \$30,311.69 | \$1.4226 | \$113.53 |
| **Womens E-Mail**| 21,387 | 189 | 0.884% | 15.14% | \$23,038.11 | \$1.0772 | \$121.89 |
| **No E-Mail** (Ctrl)| 21,306 | 122 | 0.573% | 10.62% | \$13,908.33 | \$0.6528 | \$114.00 |
| **Total / Pooled** | **64,000** | **578** | **0.903%** | **14.68%** | **\$67,258.13** | **\$1.0509** | **\$116.36** |

### 1.4 Covariates ($X$)
8 features capturing customer history and geography:
- **Continuous / Numerical (5):**
  - `recency`: Months since last purchase.
  - `history`: Dollar value of purchases in the past year.
  - `mens`: Binary indicator whether customer purchased Men's merchandise in past year.
  - `womens`: Binary indicator whether customer purchased Women's merchandise in past year.
  - `newbie`: Binary indicator whether customer is a new customer within past 12 months.
- **Categorical (3, one-hot encoded in [`src/run_real.py`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/src/run_real.py)):**
  - `history_segment`: Binned historical spend levels.
  - `zip_code`: Urban, Suburban, or Rural.
  - `channel`: Web, Phone, or Multichannel.

### 1.5 Real-Data Evaluation Without Ground Truth
Because true individual effects $\tau_{\text{level}}(x)$ are unknown in real observational/experimental data:
1. **Rank Stability Across Scales ($\rho_{\text{vs\_a1}}$):** Evaluates whether scaling $Y \to aY$ flips uplift rankings out-of-fold.
2. **Top-20% Overlap:** Quantifies customer targeting turnover between units.
3. **Out-of-Fold Cross-Validation:** Repeated 5-fold CV to ensure no model evaluates on training data.
4. **IPW Policy Value:** Inverse propensity weighted empirical payoff:
   $$\text{Value}(k) = \frac{1}{N}\sum_{i=1}^N \frac{Y_i \cdot \mathbf{1}_{\{W_i = T_i(k)\}}}{W_i e + (1 - W_i)(1 - e)}$$
   evaluated at targeting fractions $k \in \{0.10, 0.20, 0.30, 0.50\}$.
5. **Qini / AUUC:** Area under the uplift curve normalized to $[0, 1]$.

---

## 2. Criteo Uplift Prediction v2.1 Dataset

### 2.1 Overview & Provenance
- **Source:** Criteo AI Lab (Diemert et al., 2018; 2021 update), downloaded from the official mirror at `https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz` and saved at [`data/criteo-research-uplift-v2.1.csv.gz`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/data/criteo-research-uplift-v2.1.csv.gz).
- **Study Design:** Large-scale randomized ad incrementality trials where a percentage of ad traffic is held out (control).

### 2.2 Verified Dimensions & Shares
Exact row counts and marginal rates computed directly from the full 13.98M-row dataset:
- **Total Records:** **13,979,592**
- **Treatment Assignment:**
  - Treated ($W = 1$): **11,882,655** (**$85.00\%$**)
  - Control ($W = 0$): **2,096,937** (**$15.00\%$**)
  - **Empirical Propensity:** $e = 0.8500$. (Confirms the requirement: "Criteo is heavily treated").
- **Targets:**
  - `visit`: 656,929 visits ($4.70\%$)
  - `conversion`: 40,774 conversions ($0.29\%$)
- **Covariates:** 12 anonymized continuous features `f0, f1, ..., f11`.

### 2.3 Semi-Synthetic Stratified Benchmark (`data/criteo_sample_200k.parquet`)
As specified in Part 3 (P12) and Part 7 of the playbook:
- Real Criteo outcomes are binary (`visit`, `conversion`), so running revenue transforms directly on raw Criteo would merely be a binary classification problem.
- Instead, Criteo serves as a **semi-synthetic benchmark**: we keep real customer covariates $X$ from Criteo to capture complex, real-world correlation structures, while generating zero-heavy spend outcomes with known ground truth via `simulate_from_X(X, W, scenario)` (Member P's module).
- We extracted an exact **200,000-row stratified subsample** preserved at [`data/criteo_sample_200k.parquet`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/data/criteo_sample_200k.parquet):
  - Preserves exact $85.00\%$ treatment share ($170,000$ treated, $30,000$ control).
  - Fully ready for immediate use once Member P hands off `simulate_from_X`.

---

## 3. Summary of Acceptance Criteria for Task Q2

| Criterion | Target | Verified Status |
| :--- | :--- | :--- |
| **Hillstrom Rows** | 64,000 | **64,000** (Verified) |
| **Hillstrom Arms** | 3 arms (`Mens`, `Womens`, `No E-Mail`) | **3 arms** (~21.3k each, $e \approx 0.50$) |
| **Hillstrom Buyer Rate** | $\sim 1\%$ | **$0.903\%$** ($578 / 64,000$) |
| **Criteo Rows** | $\approx 14\text{M}$ | **13,979,592** (Verified) |
| **Criteo Treatment Share** | Heavily treated ($>80\%$) | **$85.00\%$** ($e = 0.8500$) |
| **Criteo Subsample** | $200\text{k}\text{--}1\text{M}$ stratified rows | **200,000 rows** in [`data/criteo_sample_200k.parquet`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/data/criteo_sample_200k.parquet) |
