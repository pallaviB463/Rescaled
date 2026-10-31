# Rescaled: Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes

[![Tests](https://img.shields.io/badge/tests-98%2F98%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Project](https://img.shields.io/badge/project-Jackfruit%20Problem%20Review-orange.svg)](presentation/jackfruit_review_4slides.pptx)
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)](LICENSE)

> **Neurolytics ADA Course Project — Jackfruit Problem Review**  
> **Team Rescaled:** Member P & Member Q  
> **Submission Deadline:** Saturday End of Day | **Review Form:** [Google Form](https://forms.gle/GXGKUTMAdAEYLbyV6)

---

## Executive Summary

In commercial applications with non-negative spend outcomes $Y \in \{0\} \cup (0, \infty)$ (e.g., e-commerce trials, ad-tech conversion value, and healthcare billing), outcomes exhibit **extreme zero inflation (90–99% zeros)** and heavy right tails. Causal machine learning practitioners routinely apply concave transformations:

$$g(y) = \log(1 + ay) \quad \text{or} \quad \text{asinh}(ay)$$

to stabilize variance, rein in outlier leverage, and satisfy MSE convergence in standard regression and tree models.

However, the scaling parameter $a$ reflects arbitrary reporting units ($a=1$ for dollars, $a=0.01$ for cents, $a=1000$ for thousands). Building on the theoretical foundations of **Chen & Roth (2024)**, this project investigates:
1. **Does the choice of reporting scale $a$ change who gets targeted by causal uplift policies?**
2. **What is the mathematical mechanism driving ranking collapse?**
3. **Can count-aware native likelihood models (Tweedie, Two-Part Hurdle) restore scale equivariance and eliminate policy regret?**

---

## Key Empirical & Theoretical Findings

1. **Massive Real-World Customer Turnover (Hillstrom Trial, $N=64{,}000$):**
   * On the real-world Hillstrom randomized marketing trial (99.1% zero spend), changing scale from dollars ($a=1$) to cents ($a=0.01$) under $\log(1+ay)$ replaces **$48.2\%$ of the top-20% targeted customers**.
   * Transforming outcomes causes marketing budgets to chase frequent low-dollar coupon clippers instead of high-incrementality customers.

2. **Theoretical Asymptotic Margin Split (Chen & Roth 2024):**
   * **Small-scale limit ($a \to 0$):** $\lim_{a \to 0} \frac{\log(1+ay)}{a} = y \implies$ targets **Level Dollars** ($\tau_{\text{level}}$), where the intensive margin (basket spend) dominates.
   * **Large-scale limit ($a \to \infty$):** $\log(1+ay) \to \log(a)\mathbf{1}(y>0) \implies$ targets purely **Extensive Margin (Conversion Probability)**, completely ignoring spend magnitude.
   * Whenever conversion lift and basket lift diverge across customer segments, varying $a$ violently reverses targeting priorities.

3. **Catastrophic Rank Collapse in Synthetic Benchmarks:**
   * In controlled experiments with diverging margins (`extensive` and `opposing` scenarios), Spearman rank correlation under `log1p` drops from $\rho = 1.000$ to **$\rho \approx 0.566$**, with top-20% customer overlap plummeting to **$35.87\%$**.

4. **Strict Scale Equivariance of Native Count Models:**
   * Native likelihood models—Compound Poisson-Gamma (**Tweedie** deviance, $p=1.5$) and Two-Part **Hurdle** (LightGBM classifier $\times$ Log-Gamma GLM)—maintain **exact $\rho = 1.0000$ and 100% targeting overlap** across all scales without ad-hoc transformations.

5. **Empirical Policy Value Superiority:**
   * On the Hillstrom trial, Tweedie achieves the highest empirical IPW policy value of **\$0.8566 / targeted user** at budget fraction $k=0.20$, outperforming naive level regression (\$0.7302) and log-transformed regression (\$0.7831) by **+\$0.12 / user**.

---

## Core Hypotheses Tested

| Hypothesis | Description | Status | Evidence |
|---|---|---|---|
| **H1 (Instability)** | CATE rank stability drops sharply as scale $a$ changes when extensive $\ne$ intensive margins. | **Validated** | Spearman $\rho$ drops to 0.566 in `extensive` and `opposing` scenarios. |
| **H2 (Transform Rate)** | Slower-growth concave transforms (`log1p`, `asinh`) destabilize far worse than power transforms (`sqrt`). | **Validated** | `sqrt` achieves $\rho \equiv 1.0000$ (scale-equivariant); `log1p` degrades rapidly. |
| **H3 (Count Equivariance)** | Count-aware models (Tweedie, Hurdle) preserve exact scale equivariance across all scales. | **Validated** | Tweedie and Hurdle maintain $\rho = 1.0000$ across all 6 synthetic DGPs. |
| **H4 (Policy Dominance)** | Count-aware models minimize empirical policy regret and maximize realized revenue. | **Validated** | Tweedie yields \$0.8566 vs \$0.7302 on Hillstrom field marketing trial. |

---

## Repository Architecture

```text
Rescaled/
├── config/
│   └── grid.yaml                 # Factorial simulation grid configuration
├── data/
│   ├── Hillstrom.csv             # Hillstrom marketing field trial (N=64,000)
│   ├── criteo_sample_200k.parquet # High-dimensional Criteo ad trial sample
│   └── processed/
│       └── criteo_semisynthetic_200k.csv # Controlled semi-synthetic spend benchmark
├── figures/
│   ├── fig2_rank_stability.png   # Spearman rho vs log10(a) across 6 DGP scenarios
│   ├── fig3_policy_value.png     # Real-world Hillstrom IPW policy value vs budget k
│   └── fig4_regret_heatmap.png   # Mean regret heatmap across learners & scenarios
├── notes/
│   ├── data_note.md              # Documentation of Hillstrom & Criteo datasets
│   ├── handoff_to_p.md           # Member Q to Member P integration contract
│   └── learner_cards.md          # Theoretical properties of transformed vs count learners
├── presentation/
│   └── jackfruit_review_4slides.pptx # Official 4-slide review presentation deck
├── report/
│   ├── main.tex                  # Master academic LaTeX manuscript with asymptotic proofs
│   ├── references.bib            # 16 academic citations (Chen & Roth, Athey, Wager, etc.)
│   ├── estimands.md              # Formal definitions of causal target estimands
│   └── simulation_design.md      # Detailed experimental simulation design
├── results/
│   ├── hillstrom_mens.csv        # Out-of-fold repeated CV metrics (Men's campaign)
│   ├── hillstrom_womens.csv      # Out-of-fold repeated CV metrics (Women's campaign)
│   └── main/                     # Stage 1 production Monte Carlo grid outputs
├── src/
│   ├── analysis.py               # Figure generation (Fig 2, Fig 3, Fig 4)
│   ├── dgp.py                    # 6 Synthetic DGP generators + semi-synthetic simulator
│   ├── metalearners.py           # Cross-fitted T-Learner and Doubly Robust (DR) Learner
│   ├── metrics.py                # Spearman rho, top-k overlap, Kendall tau, IPW policy value
│   ├── models.py                 # L2, Tweedie (p=1.5), Hurdle, and Poisson regressors
│   ├── run_real.py               # Hillstrom field trial cross-validation pipeline
│   ├── run_sim.py                # Parallel multi-core simulation runner (joblib)
│   └── transforms.py             # identity, log1p, sqrt, asinh, rank transformations
└── tests/
    ├── test_analysis.py          # Validation for plotting and metric aggregation
    ├── test_dgp.py               # Invariant tests for 6 synthetic DGP scenarios
    ├── test_metrics.py           # Mathematical unit tests for ranking metrics
    └── test_models.py            # Fit and prediction tests for count and L2 models
```

---

## Datasets

1. **Hillstrom E-Commerce Email Marketing Trial ($N = 64{,}000$):**
   * Real randomized field experiment testing email campaign incrementality.
   * Two active treatment arms (Men's Campaign: $N=42{,}613$, Women's Campaign: $N=42{,}693$) vs. uncontacted control ($N=21{,}307$).
   * Outcome: 2-week spend ($) with **$99.1\%$ zeros** (0.9% conversion rate) and heavy positive spend ($>\$500$).
   * Evaluated under repeated 5-fold cross-fitting across scale grid $a \in [10^{-3}, 10^3]$ and budget fractions $k \in [0.01, 0.50]$.

2. **Criteo Uplift v2.1 Semi-Synthetic Benchmark ($N = 200{,}000$):**
   * Real high-dimensional ad-tech covariates ($X_1 \dots X_{12}$) and empirical propensity scores.
   * Paired with controlled synthetic spend outcomes to isolate extensive vs. intensive treatment effect margins under real covariate distributions.

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10 or 3.11
* Windows, Linux, or macOS

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/pallaviB463/Rescaled.git
cd Rescaled

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Running Tests

The test suite validates DGP mathematical invariants, model fitting, numerical safety on sparse zero partitions, and metric correctness:

```bash
# Run the entire test suite (98 tests passing)
pytest -v

# Run module-specific tests
pytest tests/test_dgp.py -v
pytest tests/test_models.py -v
pytest tests/test_metrics.py -v
```

---

## Reproducing Experiments & Figures

### 1. Run the Hillstrom Field Trial Pipelines
Executes full 5-fold cross-fitting across budget fractions $k \in [0.01, 0.50]$ and scale factors $a \in [10^{-3}, 10^3]$:
```bash
python -m src.run_real
```

### 2. Run the Synthetic Factorial Simulation Grid
Executes the $6 \times 4 \times 5$ Monte Carlo experimental design across all scenarios (`null`, `homogeneous`, `extensive`, `intensive`, `opposing`, `correlated`) with 4 parallel worker cores:
```bash
python src/run_sim.py config/grid.yaml results/main --jobs 4 --tier must
```

### 3. Rebuild Publication Figures
Generates Figures 2, 3, and 4 into `figures/`:
```bash
python src/rebuild_figures.py
```

---

## Deliverables for the Jackfruit Problem Review

1. **4-Slide Presentation Deck (`presentation/jackfruit_review_4slides.pptx`):**
   * **Slide 1:** Problem statement (zero-heavy spend, Chen & Roth 2024 limiting regimes, business turnover).
   * **Slide 2:** Implementation so far (modular causal architecture, 6 DGPs, Hillstrom trial, 98/98 unit tests).
   * **Slide 3:** Result plots (Figure 2 synthetic rank collapse vs. Figure 3 Hillstrom policy value dominance of Tweedie).
   * **Slide 4:** Remaining work (Phase 1 simulation completion, Phase 2 Criteo benchmarks, Phase 3 final report & defense).
2. **Master LaTeX Manuscript (`report/main.tex`):**
   * Contains complete theoretical asymptotic proofs ($a \to 0$ and $a \to \infty$), Table 1 (DGP design), Table 2 (Learner Cards), Table 3 (Hillstrom field performance), and 16 citations in `report/references.bib`.

---

## Team Contributions

* **Member P:** Base learner implementation, hyperparameter tuning architecture, oracle-based sensitivity analysis, and semi-synthetic Criteo baseline.
* **Member Q:** Simulation engine, Hillstrom empirical pipelines, figure generation, theoretical asymptotic proofs, and final review presentation deck.

---

## License

This project is licensed for academic and research purposes as part of the Neurolytics ADA Course Project.
