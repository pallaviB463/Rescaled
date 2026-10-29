# P2 — Literature Track A: CATE Estimation, Uplift, and Policy Learning

## 1. Motivation

Conditional average treatment effect (CATE) estimation aims to identify how treatment effects vary across individuals with different covariates. Uplift modeling addresses a closely related problem: ranking individuals according to the incremental impact of a treatment. These methods are relevant when resources are limited and only a subset of individuals can be targeted.

## 2. Foundational literature

### Heterogeneous treatment effects with zero-inflated outcomes

Oganisian, Mitra, and Roy (2021) develop a Bayesian nonparametric model for continuous zero-inflated outcomes and show how it can support causal effect estimation. Kim et al. (2024) specifically develop a Bayesian nonparametric approach for heterogeneous causal effects with zero-inflated data.

**Relevance:** These studies establish that zero inflation and heterogeneous causal effects are already an active research area. Our project therefore focuses more narrowly on sensitivity to outcome transformations and scale, oracle ranking stability, and targeting consequences across a controlled set of scenarios and outcome-model families. A broader literature search is still needed before asserting novelty. <Cite refs={["turn860281search7","turn476330search3"]}/>

### Meta-learners

Künzel et al. (2019) present a framework for meta-learners and introduce the X-learner. Their work demonstrates that no single meta-learner performs best in every setting; performance depends on factors such as treatment-group imbalance and the structure of treatment-effect heterogeneity.

**Relevance:** This motivates comparing multiple CATE learners rather than treating one estimator as universally optimal.

### R-learner and quasi-oracle estimation

Nie and Wager (2021) develop an approach that separates nuisance-function estimation from optimization of the treatment-effect signal. Their results provide theoretical motivation for flexible CATE estimation and careful model selection.

**Relevance:** Estimator choice and fitting procedures can affect the estimated treatment-effect function. Our project adds outcome transformation and scale as further choices to evaluate.

### Policy learning

Athey and Wager (2021) study learning treatment-allocation policies under practical constraints. Their work connects estimated causal effects to decisions about whom to treat.

**Relevance:** Ranking quality matters because errors in the ordering of individuals can change the selected target group and the value of the resulting policy.

### Uplift-modeling surveys

Gutierrez and Gérardy (2017) review causal inference and uplift-modeling approaches. Zhang, Li, and Liu (2022) provide a broader survey connecting uplift modeling with treatment-effect heterogeneity estimation.

**Relevance:** These reviews establish the methodological context and help position the project at the intersection of CATE estimation and treatment targeting.

## 3. Candidate research gap

The project investigates whether the choice of outcome transformation and positive scale factor changes the ranking implied by the transformed-scale CATE relative to the level-scale CATE, particularly for zero-inflated outcomes.

The experimental design varies the transformation, scale, outcome-generating scenario, and estimator family. Oracle comparisons use rank correlation and top-20% overlap, while policy-value and regret measures assess the decision consequences.

The literature reviewed here establishes the broader CATE, uplift, and policy-learning context. It does not yet establish that the specific transformation-by-scale question is novel. A targeted search for work on transformed causal estimands, count outcomes, zero-inflated outcomes, and ranking stability is required before making a strong novelty claim.

## 4. Research questions

1. How does the outcome transformation and its scale affect the oracle ranking relative to the level-scale CATE?
2. Are the effects of transformation and scale consistent across null, homogeneous, extensive, intensive, aligned, and opposed treatment-effect scenarios?
3. Do changes in ranking quality translate into changes in targeting overlap, policy value, or regret?
4. Do conclusions depend on the outcome model, including L2, Poisson, Tweedie, and Hurdle models?

## References

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Meta-learners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165.
- Nie, X., and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319.
- Athey, S., and Wager, S. (2021). Policy learning with observational data. *Econometrica*, 89(1), 133–161.
- Gutierrez, P., and Gérardy, J.-Y. (2017). Causal inference and uplift modelling: A review of the literature. *Proceedings of the 3rd International Conference on Predictive Applications and APIs*, 1–13.
- Zhang, W., Li, J., and Liu, L. (2022). A unified survey of treatment effect heterogeneity modelling and uplift modelling. *ACM Computing Surveys*, 54(8), Article 162.
- Oganisian, A., Mitra, N., and Roy, J. A. (2021). A Bayesian nonparametric model for zero-inflated outcomes: Prediction, clustering, and causal estimation. *Biometrics*, 77(1), 125–135. https://doi.org/10.1111/biom.13244
- Kim, C., Li, Y., Xu, T., and Liao, Z. (2024). Bayesian nonparametric model for heterogeneous treatment effects with zero-inflated data. *Statistics in Medicine*, 43(30), 5968–5982. https://doi.org/10.1002/sim.10266