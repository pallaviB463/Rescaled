# 5-Minute Presentation Outline (Task Q12)

**Title:** Does the Way We Transform Outcomes Change Who We Target?  
**Subtitle:** Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes, with Count-Aware Learners  
**Presenters:** Team Neurolytics (Member P & Member Q)  
**Total Duration:** 5 minutes (approx. 50–60 seconds per slide)

---

### Slide 1: The Ubiquitous Industry Practice & The Silent Flaw (0:00 – 0:50)
- **The Problem:** In marketing, e-commerce, and healthcare, spend and revenue outcomes are overwhelmingly zero ($>90\%$ zeros).
- **The Reflexive Solution:** Practitioners routinely apply $\log(1 + y)$ or $\text{arcsinh}(y)$ before training gradient boosting models or causal meta-learners to curb positive skewness.
- **The Warning from Econometrics:** Chen & Roth (2024, QJE) showed that for average treatment effects (ATE), log-like transforms yield estimates that arbitrarily depend on the unit scale ($y$ in dollars vs. cents).
- **The Critical Question:** Does this scale dependence infect individual-level heterogeneous treatment effects (CATE)? Does changing currency units from dollars to cents silently change **who we choose to target**?

---

### Slide 2: Estimands & Theoretical Foundations (0:50 – 1:40)
- **Two Estimands That Must Never Be Mixed:**
  - $\tau_{\text{level}}(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$: Incremental revenue (what businesses actually care about).
  - $\tau_{m, a}(X) = \mathbb{E}[m(aY(1)) - m(aY(0)) \mid X]$: Transformed proxy estimand.
- **The Core Hypotheses (H1 & H2):**
  - **H1:** When the extensive margin (conversion probability) and intensive margin (spend given purchase) disagree, log-like rankings change with the scale parameter $a$.
  - **H2 (Asymptotic Limits):** As $a \to 0$, log-rankings converge to dollar (level) effects; as $a \to \infty$, they collapse to pure conversion probability effects.
- **The Proposed Fix (H3 & H4):** Count-aware log-link models ($\mathbb{E}[Y \mid X] = \exp(\theta(X))$: Poisson, Tweedie, Hurdle) are unit-equivariant—predictions scale linearly by $a$, preserving exact rankings across units.
- **Negative Controls:** $\sqrt{ay} = \sqrt{a}\sqrt{y}$ (scale-homogeneous) and rank transforms must have $\rho_{\text{vs\_a1}} = 1.0000$.

---

### Slide 3: Synthetic Experiments & Exact Ground Truth (1:40 – 2:30)
- **Setup:** 6 synthetic scenarios ($n=20{,}000$, $90\%$ zeros), exact counterfactual ground truth, common random numbers, evaluating S-, T-, X-, and DR-learners across 7 unit scales ($a \in [0.001, 1000]$).
- **Key Findings in the Opposed Scenario:**
  - **Negative Controls Pass:** Square root and rank transforms achieve exact $\rho_{\text{vs\_a1}} = 1.0000$.
  - **Count-Aware Immunity:** Hurdle, Poisson, and Tweedie maintain $\rho_{\text{vs\_a1}} \approx 1.0000$ across all scales.
  - **Transform Collapse:** For $\log(1+ay)$ and $\text{arcsinh}(ay)$, rank correlation drops from $1.00$ down to $0.53 - 0.56$ at small scales and $0.87 - 0.89$ at large scales.
  - **Customer Turnover:** Under a $20\%$ budget, nearly **two-thirds ($65\%$) of targeted customers are substituted** simply by changing currency denominations!

---

### Slide 4: Real-World Evidence: Kevin Hillstrom Email Field Trial (2:30 – 3:30)
- **The Benchmark:** 64,000 customers randomized into Men's E-Mail, Women's E-Mail, and No E-Mail ($99.1\%$ zeros; only $0.9\%$ buyers).
- **Evaluation:** Repeated 5-fold CV with out-of-fold scoring; transforms fitted on training folds only.
- **Empirical Confirmation (Table 3):**
  - In Men's E-Mail ($N = 42{,}613$), unit scaling to hundreds of dollars drops $\log(1+y)$ rank correlation to $\mathbf{0.7503}$, replacing **$48.4\%$** of the top-20% customer list.
  - In Women's E-Mail ($N = 42{,}693$), rank correlation drops to $\mathbf{0.7756}$, replacing **$45.8\%$** of targeted customers.
  - **Superior Payoff:** Tweedie and Hurdle models preserve $100\%$ customer overlap and deliver the highest empirical IPW policy value (\$0.8566 vs. \$0.8206 for $\log 1p$).

---

### Slide 5: Practical Guidance & The Decision Rule (3:30 – 4:20)
- **When Does It Matter?**
  - **Aligned Margins:** If buyers with high conversion uplift also spend more, rank changes are mild ($\rho \ge 0.97$).
  - **Opposed/Divergent Margins:** If promotions attract discount seekers or drive high spend among infrequent shoppers, transformed models cause severe revenue loss.
- **The Prescriptive Decision Rule for Data Science Teams:**
  1. **Directly Model the Level Scale:** Use count-aware models (compound Poisson-Gamma / Tweedie or two-part Hurdle) inside T- or DR-learners.
  2. **Mandatory Sensitivity Audit:** If using $\log(1+y)$ or $\text{arcsinh}$, audit rankings across $a \in [0.01, 1, 100]$. If top-tier overlap drops below $90\%$, reject the transformation.
  3. **Sanity Check:** Always test square root or rank transforms to ensure code invariance.

---

### Slide 6: Summary & Impact (4:20 – 5:00)
- **Takeaway:** Data preparation is not causally neutral. Arbitrary unit choices quietly reorder who receives treatments, promotions, or interventions.
- **Solution:** Unit-equivariant count-aware learners eliminate this vulnerability, aligning algorithmic targeting with true business value.
- **Artifacts & Reproducibility:** Entire benchmark suite is fully reproducible from raw parquet/CSV files via `python src/rebuild_figures.py`.
- **Q&A:** Ready for questions.
