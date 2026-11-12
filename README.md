# Rescaled: Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes

[![Tests](https://img.shields.io/badge/tests-192%2F192%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Project](https://img.shields.io/badge/project-Jackfruit%20Problem%20Review-orange.svg)](jackfruit_review_4slides_updated.pdf)
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)](LICENSE)

> **Neurolytics ADA Course Project — Team Rescaled**  
> **Topic:** Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes, with Count-Aware Learners

---

## Executive Summary

In commercial applications with non-negative spend outcomes $Y \in \{0\} \cup (0, \infty)$ (such as e-commerce promotions, ad-tech conversion values, and customer billing), outcomes exhibit **extreme zero inflation (70–99% zeros)** and heavy right tails. Causal machine learning practitioners routinely apply concave transformations:

$$g(y) = \log(1 + ay) \quad \text{or} \quad \text{asinh}(ay)$$

to stabilize variance, rein in outlier leverage, and satisfy MSE convergence in standard regression and tree models.

However, the scaling parameter $a$ reflects arbitrary reporting units ($a=1$ for dollars, $a=0.01$ for cents, $a=1000$ for thousands). Building on the theoretical foundations of **Chen & Roth (2024)**, this project investigates:
1. **Does the choice of reporting scale $a$ change who gets targeted by causal uplift policies?**
2. **What is the mathematical mechanism driving ranking collapse?**
3. **Can count-aware native likelihood models (Tweedie, Two-Part Hurdle) restore scale equivariance and eliminate policy regret?**

---

## Key Empirical & Theoretical Findings

1. **Massive Real-World Customer Turnover (Hillstrom Field Trial, $N=64{,}000$, 99.1% Zeros):**
   * On the real-world Hillstrom randomized marketing trial, changing scale from dollars ($a=1$) to cents ($a=0.01$) under $\log(1+ay)$ replaces **$48.2\%$ of the top-20% targeted customers**.
   * Ad-hoc transformation causes targeting policies to churn and chase frequent low-dollar customers instead of high-incrementality customers.

2. **Theoretical Asymptotic Margin Split (Chen & Roth 2024):**
   * **Small-scale limit ($a \to 0$):** $\lim_{a \to 0} \frac{\log(1+ay)}{a} = y \implies$ targets **Level Dollars** ($\tau_{\text{level}}$), where the intensive margin (spend magnitude) dominates.
   * **Large-scale limit ($a \to \infty$):** $\log(1+ay) \to \log(a)\mathbf{1}(y>0) \implies$ targets purely **Extensive Margin (Conversion Probability)**, completely ignoring spend magnitude.
   * Whenever conversion lift and basket lift diverge across customer segments, varying $a$ violently reverses targeting priorities.

3. **Catastrophic Rank Collapse in Synthetic Benchmarks:**
   * In controlled experiments with diverging margins (`extensive` and `opposed` scenarios), Spearman rank correlation under `log1p` drops from $\rho = 1.000$ to **$\rho \approx 0.566$**, with top-20% customer overlap plummeting to **$35.87\%$**.

4. **Strict Scale Equivariance of Native Count Models:**
   * Native likelihood models—Compound Poisson-Gamma (**Tweedie** deviance, $p=1.5$) and Two-Part **Hurdle** (LightGBM classifier $\times$ Gamma GLM)—maintain **exact $\rho = 1.0000$ and 100% targeting overlap** across all scales without ad-hoc transformations.

5. **Empirical Policy Value Superiority:**
   * On the Hillstrom trial, Tweedie achieves the highest empirical IPW policy value of **\$0.8566 / targeted user** at budget fraction $k=0.20$, outperforming naive level regression (\$0.7302) and log-transformed regression (\$0.7831) by **+\$0.12 / user**.

---

## Core Hypotheses Tested

| Hypothesis | Description | Status | Evidence |
|---|---|---|---|
| **H1 (Instability)** | CATE rank stability drops sharply as scale $a$ changes when extensive $\ne$ intensive margins. | **Validated** | Spearman $\rho$ drops to 0.566 in `extensive` and `opposed` scenarios. |
| **H2 (Transform Rate)** | Slower-growth concave transforms (`log1p`, `asinh`) destabilize far worse than power transforms (`sqrt`). | **Validated** | `sqrt` achieves $\rho \equiv 1.0000$ (scale-equivariant); `log1p` degrades rapidly. |
| **H3 (Count Equivariance)** | Count-aware models (Tweedie, Hurdle) preserve exact scale equivariance across all scales. | **Validated** | Tweedie and Hurdle maintain $\rho = 1.0000$ across all 6 synthetic DGPs. |
| **H4 (Policy Dominance)** | Count-aware models minimize empirical policy regret and maximize realized revenue. | **Validated** | Tweedie yields \$0.8566 vs \$0.7302 on Hillstrom field marketing trial. |

---

## Repository Architecture

```text
Rescaled/
├── config/
│   ├── grid.yaml                          # Simulation grid configurations
│   ├── grid_pilot.yaml
│   └── grid_time.yaml
├── data/
│   ├── Hillstrom.csv                      # Hillstrom marketing trial (N=64,000, 99.1% zeros)
│   ├── criteo_sample_200k.parquet         # High-dimensional Criteo ad trial sample
│   └── processed/
│       ├── criteo_semisynthetic_200k.csv  # Semi-synthetic spend benchmark
│       └── criteo_semisynthetic_manifest.json
├── figures/
│   ├── fig2_rank_stability.png            # Spearman rho vs scale a
│   ├── fig3_policy_value.png              # Hillstrom policy value vs budget k
│   └── fig4_regret_heatmap.png            # Regret heatmap across learners & scenarios
├── report/
│   ├── main.tex                           # Master IEEE conference manuscript
│   ├── references.bib                     # Academic bibliography
│   ├── estimands.md                       # Formal causal estimands
│   ├── backtransform.md                   # Back-transformation analysis
│   ├── scenarios_stress_grid.md           # Stress grid documentation
│   └── simulation_design.md               # Experimental simulation design
├── results/
│   ├── hillstrom_mens.csv                 # Out-of-fold repeated CV metrics (Men's campaign)
│   ├── hillstrom_womens.csv               # Out-of-fold repeated CV metrics (Women's campaign)
│   ├── oracle_analysis/                   # Figure 1 oracle ranking stability
│   ├── tuning/                            # Hyperparameter tuning benchmarks
│   └── null_audit/                        # Null scenario audit metrics
├── src/
│   ├── dgp.py                             # 6 Synthetic DGP generators + semi-synthetic simulator
│   ├── metalearners.py                    # S, T, X, DR, and Transformed Outcome learners
│   ├── metrics.py                         # Spearman rho, top-k overlap, policy value, regret
│   ├── models.py                          # L2, Tweedie (p=1.5), Hurdle, Poisson, NegBin
│   ├── run_real.py                        # Hillstrom field trial cross-validation pipeline
│   ├── run_sim.py                         # Simulation experiments runner
│   ├── transforms.py                      # identity, log1p, sqrt, asinh, rank transformations
│   ├── tuning.py                          # Hyperparameter tuning framework
│   └── oracle_analysis.py                 # Oracle ranking stability pipeline
└── tests/
    └── 16 test suites (192 automated unit tests)
```

---

## Setup & Testing

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Test Suite
```bash
python -m pytest tests
```
*Current test status:* **192 passed** across all 16 test suites.

---

## Deliverables & Review Submission

The project deliverables package for the course evaluation is located under `Team_Rescaled_Deliverables/`:
1. `1_Final_One_Pager/`: Course-format one-pager summary.
2. `2_Code_ipynb/`: Self-contained master Jupyter notebook (`Rescaled_CATE_Uplift_Master.ipynb`).
3. `3_Dataset/`: Hillstrom trial ($N=64{,}000$) and Criteo sample ($N=200{,}000$).
4. `4_Project_PPT/`: 8-slide presentation deck + 4-slide review deck.
5. `5_Video_Presentation_Script/`: Rehearsal presentation script (4m 15s).
6. `6_IEEE_Report/`: Formal 2-column IEEE conference paper (`main.tex`).
