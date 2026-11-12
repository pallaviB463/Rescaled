# Does the Way We Transform Outcomes Change Who We Target?
## Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes, with Count-Aware Learners
**Jackfruit Project – Final One-Pager | Advanced Data Analytics (ADA-2026)**  
**Target Submission:** November 16, 2026 | **Course Faculty:** Prof. Bhaskarjyoti Das | **Teaching Team:** ADA TA Team

---

### Team Members
| Name | SRN |
| :--- | :--- |
| **Pallavi B** | PES1UG24AM906 |
| **Khushi Patel** | PES1UG24AM134 |
| **N Lakshanya** | PES1UG24AM169 |
| **Porla Neha** | PES1UG25AM808 |

---

### 1. Project Summary
In commercial and policy domains—such as e-commerce marketing, ad-tech conversion bidding, and clinical trial expenditures—outcomes $Y \in \{0\} \cup (0, \infty)$ are characterized by **severe zero inflation (90–99% exact zeros)** and heavy right tails. Causal machine learning practitioners routinely apply non-linear concave transformations, universally $g(y) = \log(1 + ay)$ or $\text{asinh}(ay)$, to stabilize variance and satisfy mean squared error loss in tree-based meta-learners. While Chen & Roth (2024) recently proved that the average treatment effect (ATE) of log-like transformations depends arbitrarily on the scale unit $a$ (e.g., dollars vs. cents), its implications for **individual-level uplift rankings and commercial targeting policies had remained unstudied**.

This project investigates whether the choice of reporting scale $a$ changes *who* gets targeted, quantifies the resulting policy regret, and evaluates whether count-aware native likelihood models restore scale equivariance without ad-hoc transformations. Across controlled Monte Carlo simulations, oracle benchmark grids, and a randomized commercial email field trial ($N=64{,}000$), we show that:
1. **Targeting Turnover:** Varying $a$ under $\log(1+ay)$ replaces up to **$48.2\%$ of top-20% targeted customers** on real-world marketing spend.
2. **Asymptotic Margin Split:** As $a \to 0$, rankings converge to Level Dollars ($\tau_{\text{level}}$); as $a \to \infty$, rankings converge strictly to Conversion Probability ($\tau_{\text{extensive}}$), entirely erasing spend magnitude.
3. **Count-Aware Equivariance:** Compound Poisson-Gamma (**Tweedie**, $p=1.5$) and Two-Part **Hurdle** estimators achieve **exact $\rho = 1.0000$ scale equivariance** and attain the highest empirical policy value (\$0.8566 vs. \$0.7302 per user).

---

### 2. Work Execution & Feasibility
The project was executed following an 8-week modular timeline, verified by a strict test suite of **192 automated unit tests (100% passing)** across 16 test modules:
* **Weeks 1–2 (Literature & Mathematical Foundation):** Mapped Chen & Roth (2024) and Mullahy & Norton (2004) to uplift meta-learners; formalized asymptotic proofs for $a \to 0$ and $a \to \infty$.
* **Weeks 3–4 (Modular Causal Engine & DGPs):** Implemented 6 synthetic DGP scenarios (`Null`, `Homogeneous`, `Extensive`, `Intensive`, `Aligned`, `Opposed`) with zero rates $p_0 \in \{0.70, 0.90, 0.98\}$; built cross-fitted T-, S-, X-, and DR-learners with L2, Tweedie ($p=1.5$), Hurdle, and Quasi-Poisson base models.
* **Weeks 5–6 (Oracle Analysis & Real-World Field Runs):** Executed oracle ranking analysis isolating estimand warping from estimation noise; ran repeated 5-fold cross-validation across budgets $k \in [0.01, 0.50]$ on Hillstrom ($N=64{,}000$) and Criteo semi-synthetic ($N=200{,}000$).
* **Weeks 7–8 (Synthesis & Artifact Compilation):** Built unified replication notebook (`.ipynb`), LaTeX IEEE manuscript, and 16:9 presentation deck. Tools: Python 3.11, LightGBM, Scikit-learn, SciPy, Joblib.

---

### 3. Datasets Evaluated
1. **Synthetic & Oracle Zero-Heavy Benchmarks ($N=20{,}000$, 6 Scenarios, 100 Replications):**
   * Configurable two-margin DGPs with known ground-truth CATE. Tests margin divergence across zero proportions $p_0 \in \{0.70, 0.90, 0.98\}$.
2. **Hillstrom MineThatData Randomized Marketing Field Trial ($N=64{,}000$):**
   * Genuine randomized retail marketing trial: Men's Campaign ($N=42{,}613$) and Women's Campaign ($N=42{,}693$) vs. Uncontacted Control ($N=21{,}307$). Outcome: 2-week spend ($) with **99.1% zeros** (0.9% buyer conversion rate).
3. **Criteo Uplift v2.1 Semi-Synthetic Benchmark ($N=200{,}000$):**
   * Real 12-dimensional advertising covariates and empirical propensity scores paired with controlled two-margin synthetic spend.

---

### 4. Novelty & Research Contributions
* **First Formal Extension of Chen & Roth (2024) to CATE & Uplift:** Proves both mathematically and empirically that non-linear outcome transformations silently alter the targeted estimand, converting monetary revenue targeting into binary conversion targeting as scale increases.
* **Count-Aware Meta-Learners for Uplift:** First benchmark introducing compound Poisson-Gamma (Tweedie deviance, $p=1.5$) and two-part Hurdle likelihoods inside cross-fitted DR- and T-learners for zero-heavy continuous spend.
* **Separation of Estimand Warping from Estimation Error:** Oracle ranking analysis proves ranking collapse is an inherent mathematical pathology of concave transformations, not an artifact of small-sample estimation variance.
* **Practical Decision Framework:** Provides an actionable decision flowchart for industry practitioners: *when is $\log(1+ay)$ acceptable, and when is Tweedie mandatory?*

---

### 5. Key Empirical Learnings & Results
* **Hypothesis H1 (Ranking Instability) — Validated:** Under $\log(1+ay)$, Spearman rank stability drops to $\rho = 0.5660$ and Top-20% overlap drops to $35.87\%$ when extensive and intensive margins diverge.
* **Hypothesis H2 (Transformation Growth Rate) — Validated:** Strictly power transforms ($y^{1/2}$) and non-parametric rank percentiles preserve scale equivariance ($\rho \equiv 1.000$), whereas slow-growth logarithmic transforms suffer severe ranking inversion.
* **Hypothesis H3 (Count Model Equivariance) — Validated:** Tweedie and Hurdle maintain exact scale-equivariance ($\rho = 1.0000$, $100\%$ overlap) across all unit scalings $a \in [10^{-3}, 10^3]$ and zero rates ($70\% - 98\%$).
* **Hypothesis H4 (Policy Welfare Maximization) — Validated:** On the Hillstrom field trial, Tweedie attains the peak empirical IPW policy value of **\$0.8566 / user** at $k=0.20$, outperforming naive level models (\$0.7302) and log-transformed regression (\$0.7831) by **+\$0.12 / targeted user**.

---
*Deliverable 1 of 6 | Team Rescaled | Advanced Data Analytics (ADA-2026)*

