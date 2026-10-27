# Learner Cards & Reference (Task Q1)

Verified formulas and properties for **Table 2** (Learner Cards) in the manuscript.
Covers S-learner, T-learner, X-learner (Künzel et al., 2019), DR-learner (Kennedy, 2023), R-learner (Nie & Wager, 2021), and the Transformed Outcome baseline.

---

## 1. Table 2: Meta-Learner Cards for Zero-Heavy Uplift

| Learner | Target Estimand $\hat{\tau}(x)$ | Nuisance Models Needed | Cross-Fitting Required? | 2nd Stage Model Class | Key Weakness on Zero-Heavy Outcomes ($90\%+$ Zeros) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S-Learner**<br>*(Single model)* | $\hat{\mu}(x, 1) - \hat{\mu}(x, 0)$ from single model $\hat{\mu}(x, w) = \mathbb{E}[Y \mid X=x, W=w]$ | None | No | None (single stage) | **Severe regularization bias toward zero:** Tree-based split criteria prioritize the dominant extensive-margin variance ($0$ vs $>0$) and often fail to split on $W$, shrinking $\hat{\tau}$ artificially toward zero. |
| **T-Learner**<br>*(Two models)* | $\hat{\mu}_1(x) - \hat{\mu}_0(x)$ where $\hat{\mu}_w(x) = \mathbb{E}[Y \mid X=x, W=w]$ | None | No | None (direct subtraction) | **No information sharing between arms:** When only $\sim 1\text{--}10\%$ of customers spend, each arm's positive-spend model trains on very sparse data. Independent estimation errors do not cancel upon subtraction, producing high estimation variance. |
| **X-Learner**<br>*(Crossover)* | $(1 - e(x))\hat{\tau}_1(x) + e(x)\hat{\tau}_0(x)$<br>where $\hat{\tau}_1$ regresses $D_1 = Y_1 - \hat{\mu}_0(X_1)$ on $X_1$, and $\hat{\tau}_0$ regresses $D_0 = \hat{\mu}_1(X_0) - Y_0$ on $X_0$ | $\hat{\mu}_0(x), \hat{\mu}_1(x)$, propensity $e(x)$ | No | **Plain $L_2$ regression only** (imputed effects $D_w$ take negative continuous values) | **Second stage cannot use count-aware models:** Imputed effects $D_1 = Y_1 - \hat{\mu}_0(X_1)$ are negative whenever a treated customer does not buy ($0 - \hat{\mu}_0 < 0$). Thus Hurdle/Poisson/Tweedie cannot be used in stage 2; only nuisance models can be count-aware. |
| **DR-Learner**<br>*(Doubly Robust)* | Regress pseudo-outcome $\phi$ on $X$, where:<br>$\phi = \hat{\mu}_1(X) - \hat{\mu}_0(X) + \frac{W(Y - \hat{\mu}_1(X))}{e(X)} - \frac{(1-W)(Y - \hat{\mu}_0(X))}{1 - e(X)}$ | $\hat{\mu}_0(x), \hat{\mu}_1(x)$, propensity $e(x)$ | **Yes ($K$-fold cross-fitting)** to prevent overfitting bias | **Plain $L_2$ regression only** ($\phi$ takes continuous positive and negative values) | **Extreme residual variance from positive spend tail:** The inverse-propensity term $\frac{W(Y - \hat{\mu}_1)}{e}$ explodes when a sparse buyer spends a large dollar amount ($Y \gg 0$). Like the X-learner, $\phi$ is real-valued, so the final regression cannot be count-aware. |
| **R-Learner**<br>*(Residualized, context only)* | $\arg\min_\tau \frac{1}{n}\sum_{i=1}^n \left[(Y_i - \hat{m}(X_i)) - (W_i - \hat{e}(X_i))\tau(X_i)\right]^2$<br>where $\hat{m}(x) = \mathbb{E}[Y \mid X=x]$ | Baseline mean $\hat{m}(x)$, propensity $\hat{e}(x)$ | Recommended ($K$-fold) | Weighted $L_2$ loss | **Objective formulation tied to squared error:** Robinson's residualization assumes an additive level scale $Y - m(X) = (W - e)\tau(X) + \varepsilon$. It does not naturally extend to two-part hurdle structures or non-linear log-link models. |
| **Transformed Outcome (TO)**<br>*(Ablation baseline)* | Regress $Y^* = Y \frac{W - e(X)}{e(X)(1 - e(X))}$ directly on $X$ | Propensity $e(x)$ | No | Plain $L_2$ regression | **Catastrophic variance:** While $\mathbb{E}[Y^* \mid X] = \tau(X)$ is unbiased in expectation, multiplying zero-heavy and skewed $Y$ by inverse propensity yields noisy, unregularized targets. |

---

## 2. In-Depth Theoretical Foundations

### A. Künzel, Sekhon, Bickel, & Yu (2019, PNAS)
- **Paper:** *Metalearners for estimating heterogeneous treatment effects using machine learning*.
- **Core Insights:**
  - Standard supervised algorithms (especially regularized regression, gradient boosted trees, and random forests) suffer from regularization bias when estimating CATE via an S-learner. If the feature dimension $d$ is large or baseline variance dominates treatment effect variance, the tree builder selects splits on prognostic covariates $X$ rather than the treatment indicator $W$.
  - The X-learner is designed for structural design asymmetry (e.g. heavily unbalanced sample sizes where $N_1 \ll N_0$ or $N_1 \gg N_0$). In our zero-heavy setting, even when $N_1 \approx N_0$, the number of *buyers* ($Y > 0$) is tiny ($\sim 1\text{--}10\%$).
  - **Combining Stage:** The weighting by propensity $\hat{\tau}_X = (1 - e)\hat{\tau}_1 + e\hat{\tau}_0$ ensures that the estimate relying on the larger arm's imputed effect receives greater weight. In randomized experiments with $e = 0.5$, $\hat{\tau}_X = 0.5 \hat{\tau}_1 + 0.5 \hat{\tau}_0$.
  - **Application to Zero-Heavy Outcomes:** Imputed effects $D_1 = Y_1 - \hat{\mu}_0(X_1)$ and $D_0 = \hat{\mu}_1(X_0) - Y_0$ represent deviations from the estimated counterfactual. Since $Y \ge 0$ while $\hat{\mu} > 0$, for non-buyers ($90\%$ of rows) $D_1 = -\hat{\mu}_0 < 0$. Therefore, **the second stage regression must always be $L_2$**, regardless of whether the first-stage nuisance models used Poisson, Tweedie, or Hurdle.

### B. Kennedy (2023, JASA / 2020 arXiv)
- **Paper:** *Towards optimal doubly robust estimation of heterogeneous causal effects*.
- **Core Insights:**
  - Derives the uncentered efficient influence function (EIF) pseudo-outcome:
    $$\phi(Z) = \mu_1(X) - \mu_0(X) + \frac{W(Y - \mu_1(X))}{e(X)} - \frac{(1-W)(Y - \mu_0(X))}{1 - e(X)}$$
  - Proves that regressing $\phi(Z)$ on $X$ achieves the "rate-doubling" property: the error in estimating $\tau$ is bounded by the product of errors in estimating nuisance functions:
    $$\|\hat{\tau}_{\text{DR}} - \tau\| \le O_P\left(\|\hat{\mu} - \mu\| \cdot \|\hat{e} - e\| + \text{complexity of second stage}\right)$$
  - In a randomized trial where $e = 0.5$ is known exactly, $\|\hat{e} - e\| = 0$, meaning the pseudo-outcome is strictly unbiased: $\mathbb{E}[\phi(Z) \mid X] = \tau(X)$, even if $\hat{\mu}$ is misspecified.
  - **Cross-Fitting Requirement:** $K$-fold sample splitting (typically 5 folds) is mathematically required. If nuisance models $\hat{\mu}$ are evaluated on the same observations used to train them, the empirical covariance between $\hat{\mu}$ and $Y - \hat{\mu}$ introduces an optimistic bias of order $O_P(1/\sqrt{n})$ that breaks asymptotic normality and valid ranking.
  - **Application to Zero-Heavy Outcomes:** For a non-buyer with $Y = 0$, $Y - \hat{\mu} = -\hat{\mu}$ is small and negative. For a large spender with $Y = \$200$ and $\hat{\mu} = \$10$, $Y - \hat{\mu} = +\$190$, which gets scaled by $1/e = 2$, yielding a pseudo-outcome spike of $+380$. This heavy tail increases the variance of $\phi$, necessitating robust tree splitting in the second stage.

### C. Nie & Wager (2021, Biometrika)
- **Paper:** *Quasi-oracle estimation of heterogeneous treatment effects*.
- **Core Insights:**
  - Solves the CATE problem via Robinson's transformation (1988), reformulating causal effect estimation as an empirical risk minimization with an induced Neyman-orthogonal score:
    $$Y - m(X) = (W - e(X))\tau(X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid X, W] = 0$$
  - Directly isolates $\tau(X)$ by residualizing both the outcome $Y$ and treatment indicator $W$, minimizing:
    $$\mathcal{L}_R(\tau) = \frac{1}{n}\sum_{i=1}^n \left[ (Y_i - \hat{m}(X_i)) - (W_i - \hat{e}(X_i))\tau(X_i) \right]^2$$
  - Demonstrates that the R-learner is quasi-oracle: it performs as well as an oracle learner that knows the true baseline mean $m(X)$, up to second-order nuisance errors.
  - **Application to Zero-Heavy Outcomes:** The R-learner loss function inherently minimizes squared error. Unlike classification or Poisson regression, it has no natural link function to restrict predictions or model two-stage zero-inflation. It serves as valuable theoretical context for orthogonal Neyman scores in related work, but cannot directly accommodate count-aware nuisance families.

### D. Curth & van der Schaar (2021, AISTATS)
- **Paper:** *Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Practice*.
- **Core Insights:**
  - Establishes a rigorous taxonomy of inductive biases across meta-learners.
  - Shows that performance depends heavily on the relationship between the baseline prognostic function $\mu_0(X)$ and the treatment effect function $\tau(X)$:
    1. If $\tau(X)$ is simple/sparse but $\mu_0(X)$ is complex, S-learners fail because the hypothesis space is overwhelmed by $\mu_0(X)$.
    2. If $\mu_0(X)$ and $\mu_1(X)$ share common structure (e.g. spend capacity), T-learners fail because they cannot pool representation.
    3. Two-step learners (X-learner and DR-learner) decouple the estimation of the baseline surface from the estimation of the treatment contrast.
  - **Relevance to Zero-Heavy Uplift:** In marketing campaigns, whether a customer has disposable income (baseline spend $\mu_0$) is governed by demographic factors (e.g. income, tenure), whereas uplift $\tau$ is governed by persuadability (responsiveness to discounts/ads). Decoupling baseline spend from uplift is essential for zero-heavy spend outcomes.

---

## 3. Implementation Contract Checks

All learners implemented in [`src/metalearners.py`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/src/metalearners.py) adhere strictly to the frozen interface:
```python
LEARNERS[name](M, X, W, y, Xte, e=0.5) -> tau_hat
```
1. **Uniform signature:** Every learner accepts `M` (nuisance model class), `X`, `W`, `y`, `Xte`, and `e`.
2. **Propensity parameter `e`:** Handled cleanly across `S`, `T`, `X`, and `DR`.
3. **Cross-fitting in DR:** 5-fold cross-fitting ensures out-of-fold nuisance evaluation without leakage (verified by `test_dr_crossfitting_never_predicts_a_row_it_trained_on`).
4. **Second-stage L2:** `X` and `DR` strictly use `L2` for the second stage regression, allowing count-aware nuisance models (`Poisson`, `Tweedie`, `Hurdle`) in the first stage while handling negative continuous pseudo-outcomes in the second stage.
