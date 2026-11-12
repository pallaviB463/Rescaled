# P2 — Literature Track A: CATE Estimation, Uplift, and Policy Learning

## 1. Motivation

Conditional average treatment effect (CATE) estimation aims to identify how treatment effects vary across individuals with different covariates. Uplift modeling addresses a closely related problem: ranking individuals according to the incremental impact of a treatment. These methods are relevant when resources are limited and only a subset of individuals can be targeted.

Our project studies causal effects when the outcome is nonnegative and contains a substantial proportion of zero observations. In such settings, researchers may transform the outcome before estimating treatment effects. However, the choice of transformation and its positive scale parameter can influence the resulting treatment-effect estimand and, consequently, the ranking of individuals.

This literature review covers heterogeneous treatment effects, uplift modeling, policy learning, transformations of zero-valued outcomes, and revenue/profit uplift modeling.

## 2. Foundational Literature

### 2.1 Heterogeneous Treatment Effects with Zero-Inflated Outcomes

Oganisian, Mitra, and Roy (2021) develop a Bayesian nonparametric model for continuous zero-inflated outcomes and show how it can support prediction, clustering, and causal effect estimation. Kim et al. (2024) specifically develop a Bayesian nonparametric approach for heterogeneous treatment effects with zero-inflated data.

**Relevance to our project:** These studies establish that zero inflation and heterogeneous causal effects are already an active research area. Our project therefore focuses more narrowly on sensitivity to outcome transformations and scale, oracle ranking stability, and targeting consequences across a controlled set of scenarios and outcome-model families.

### 2.2 Meta-Learners

Künzel et al. (2019) present a framework for meta-learners for estimating heterogeneous treatment effects using machine learning. Their work introduces the X-learner and compares different meta-learning approaches, including the S-learner, T-learner, and X-learner.

Their results demonstrate that no single meta-learner performs best in every setting. Performance depends on factors such as treatment-group imbalance, sample size, and the structure of treatment-effect heterogeneity.

**Relevance to our project:** This motivates comparing multiple CATE learners rather than treating one estimator as universally optimal. It also highlights the importance of separating the effects of the outcome representation from the effects of the learning algorithm.

### 2.3 R-Learner and Quasi-Oracle Estimation

Nie and Wager (2021) develop the R-learner, an approach that separates nuisance-function estimation from optimization of the treatment-effect signal. Their work provides theoretical motivation for flexible CATE estimation and careful model selection.

The method uses an objective based on residualized outcomes and treatment assignments, allowing the treatment-effect function to be estimated while accounting for nuisance components.

**Relevance to our project:** Estimator choice and fitting procedures can affect the estimated treatment-effect function. Our project adds outcome transformation and scale as further choices to evaluate. It is important to distinguish ranking changes caused by the estimand itself from ranking changes caused by estimation error.

### 2.4 Policy Learning

Athey and Wager (2021) study learning treatment-allocation policies using observational data. Their work connects estimated causal effects to decisions about whom to treat.

Policy learning is important because a model is often used not merely to predict treatment effects but to make decisions under resource constraints.

**Relevance to our project:** Ranking quality matters because errors in the ordering of individuals can change the selected target group and the value of the resulting policy. Our use of top-20% overlap and policy-value/regret measures connects CATE ranking quality to targeting decisions.

### 2.5 Uplift-Modeling Surveys

Gutierrez and Gérardy (2017) review causal inference and uplift-modeling approaches. Their work provides an overview of methods for estimating incremental treatment effects and using these estimates for customer targeting.

Zhang, Li, and Liu (2022) provide a broader survey connecting uplift modeling with treatment-effect heterogeneity estimation.

**Relevance to our project:** These reviews establish the methodological context and help position the project at the intersection of CATE estimation and treatment targeting.

## 3. Transformation of Outcomes with a Mass at Zero

### 3.1 Chen and Roth (2024): Logs with Zeros? Some Problems and Solutions

Chen and Roth (2024) investigate the interpretation of treatment effects when the outcome is nonnegative but may equal zero. Researchers commonly apply log-like transformations such as log(1 + Y) and the inverse hyperbolic sine transformation to handle zero observations while retaining properties similar to logarithmic outcomes for large positive values.

The central argument is that treatment effects on these transformed outcomes should not automatically be interpreted as percentage effects. Unlike a conventional percentage change, transformed treatment effects depend on the units in which the outcome is measured. When treatment changes the probability of receiving a positive outcome, changing the measurement scale before applying a log-like transformation can change the magnitude of the estimated treatment effect. In particular, the extensive margin can generate substantial scale sensitivity.

**Relevance to our project:** This paper establishes that the scale parameter inside a log-like transformation is not an innocuous computational choice. Our study extends the investigation toward heterogeneous treatment-effect rankings. Rather than examining only an average treatment effect, we investigate whether changing the outcome scale changes the ordering of individuals according to their transformed-scale CATE relative to the level-scale CATE.

**Important distinction:** Chen and Roth study average treatment effects of transformed outcomes and their interpretation. Their results do not, by themselves, establish the behavior of heterogeneous CATE rankings under every transformation or data-generating process.

### 3.2 Mullahy and Norton (2024): Why Transform Y? The Pitfalls of Transformed Regressions with a Mass at Zero

Mullahy and Norton (2024) examine regression models that transform nonnegative, right-skewed outcomes with a substantial mass at zero. They focus on transformations including the logarithm of the outcome plus a constant and the inverse hyperbolic sine. Their analysis highlights that the presence of zeros changes the interpretation and behavior of transformed regressions relative to settings where outcomes are strictly positive.

A central issue is the sensitivity of estimated marginal effects to transformation parameters. When zero observations are common, transformations can separate zero outcomes from positive observations in ways that affect the relative contribution of extensive-margin and intensive-margin variation. Consequently, estimated coefficients may reflect changes in the probability of a positive outcome as well as changes in the magnitude of positive outcomes. The authors discuss alternatives such as two-part models and regressions using untransformed outcomes.

**Relevance to our project:** This paper motivates distinguishing purchase-probability effects from conditional spending effects. It also supports comparing log1p and arcsinh transformations with level-scale and models that explicitly represent the occurrence and magnitude of positive outcomes.

**Important distinction:** The paper primarily studies regression coefficients and marginal-effect interpretation. It does not directly provide the oracle ranking analysis across heterogeneous treatment-effect scenarios that we conduct.

## 4. Revenue and Profit Uplift Modeling

### 4.1 Gubela, Lessmann, and Jaroszewicz (2020): Response Transformation and Profit Decomposition for Revenue Uplift Modeling

Gubela, Lessmann, and Jaroszewicz (2020) study uplift modeling when the objective is to maximize incremental revenue rather than merely incremental conversion. They argue that revenue is a more direct business objective when customers differ substantially in their spending behavior.

Their approach transforms customer-level revenue responses to enable the use of standard machine-learning algorithms for uplift estimation. A central challenge is that revenue data commonly contain many zeros because a large fraction of customers do not purchase. The authors address this issue through two-stage revenue uplift approaches and develop a decomposition of campaign profit to evaluate the financial consequences of targeting decisions.

**Relevance to our project:** This work establishes that zero-inflated revenue uplift is an existing research area. It demonstrates that transforming the response can facilitate uplift modeling, while decision quality should ultimately be evaluated through revenue or profit objectives. Our project investigates a different question: whether changing the positive scale inside a transformation changes the oracle ranking itself, even before estimation error is introduced.

**Important distinction:** The paper's primary contribution is revenue uplift modeling and profit decomposition. It should not be interpreted as directly evaluating oracle ranking sensitivity over the seven positive outcome scales in our study.

### 4.2 Baier and Stöcker (2022): Profit Uplift Modeling for Direct Marketing Campaigns

Baier and Stöcker (2022) examine uplift modeling with profit as the target rather than traditional response outcomes such as visits or conversions. They argue that selecting customers based on response probability alone can be suboptimal when the business objective depends on the economic value of the response.

The authors propose several profit-uplift approaches, including one-stage ordinary regression and random forests, as well as two-stage Heckman sample-selection and zero-inflated negative-binomial regression methods. Their empirical applications use real-world online retail data, including discount campaigns and the Hillstrom dataset.

**Relevance to our project:** This paper motivates the inclusion of count-aware outcome models and two-part modeling ideas in our simulation comparisons. It demonstrates that zero-inflated models can be used in profit-oriented uplift analysis.

**Important distinction:** The zero-inflated negative-binomial approach is a model for profit uplift in the paper's setting. Its inclusion does not imply that negative-binomial regression is automatically appropriate for every continuous-spending outcome.

### 4.3 He et al. (2024): Rankability-Enhanced Revenue Uplift Modeling Framework for Online Marketing

He et al. (2024) study revenue uplift modeling for online marketing, emphasizing that ranking customers by their incremental revenue response is central to effective targeting. They identify difficulties associated with continuous, long-tailed revenue distributions and note that conventional response-regression approaches may not adequately optimize uplift ranking.

Their framework uses a zero-inflated lognormal (ZILN) loss to model revenue responses. In addition to response modeling, the authors introduce ranking-oriented objectives, including a listwise uplift-ranking loss, to directly address ranking errors across the population.

**Relevance to our project:** This paper is particularly important for positioning our work because it already combines zero-inflated lognormal outcome modeling with an explicit uplift-ranking objective. Therefore, we must not claim that applying a ZILN model to revenue uplift or considering ranking quality in revenue targeting is itself novel.

Our proposed distinction is to study how the transformation and positive scale alter the underlying oracle CATE estimand and its ranking, independently of the learner's estimation error.

**Important distinction:** The paper focuses on developing and evaluating a rankability-enhanced learning framework. Our oracle study instead isolates ranking changes induced by the choice of outcome transformation and scale across controlled data-generating scenarios.

## 5. Candidate Research Gap

The literature establishes several relevant research directions:

1. Treatment-effect estimation with zero-inflated outcomes has been studied using Bayesian nonparametric approaches.
2. Log-like transformations and their interpretation can be sensitive to outcome units and the presence of zeros.
3. Two-stage revenue and profit uplift models already address zero-inflated monetary outcomes.
4. Zero-inflated lognormal models and explicit uplift-ranking objectives have already been applied to revenue targeting.
5. CATE estimation and policy-learning methods provide established frameworks for ranking individuals and selecting treatment recipients.

Our project investigates a more specific question: **whether the choice of outcome transformation and positive scale factor changes the ranking implied by the transformed-scale CATE relative to the level-scale CATE, particularly for zero-inflated outcomes.**

The experimental design varies:

- Outcome transformation.
- Positive outcome scale.
- Outcome-generating scenario.
- Control-group zero-rate parameter.
- Outcome-model family.

Oracle comparisons use rank correlation and top-20% overlap, while policy-value and regret measures assess the decision consequences.

The central distinction in our investigation is between:

- **Estimand-induced ranking changes:** changes in the true CATE ranking caused by the transformation and scale.
- **Estimator-induced ranking changes:** changes caused by finite samples, model fitting, and estimation error.

By studying oracle rankings separately, we can isolate the first source of variation before evaluating fitted learners.

This literature review does not establish that the specific transformation-by-scale question is definitively novel. A broader literature search and comparison with related ranking and revenue-uplift studies are required before making a strong novelty claim.

## 6. Research Questions

1. How does the outcome transformation and its scale affect the oracle ranking relative to the level-scale CATE?
2. Are the effects of transformation and scale consistent across null, homogeneous, extensive, intensive, aligned, and opposed treatment-effect scenarios?
3. Do changes in ranking quality translate into changes in targeting overlap, policy value, or regret?
4. Do conclusions depend on the outcome model, including L2, Poisson, Tweedie, and Hurdle models?
5. How much of the observed ranking change is caused by the transformed estimand itself, as opposed to estimation error?

## 7. Summary and Research Positioning

The reviewed literature establishes that zero-inflated outcomes, transformed regression, revenue uplift, profit uplift, and ranking-based targeting are all active research areas.

Our project should therefore avoid claiming novelty in any of these broad areas individually. Its proposed contribution is a controlled investigation of the sensitivity of oracle heterogeneous treatment-effect rankings to outcome transformations and positive scale parameters, followed by comparisons of fitted outcome-model and CATE-learning approaches.

The oracle analysis is particularly important because it distinguishes changes in the target estimand from errors introduced by model estimation. This distinction allows us to examine whether a transformation can change who is selected for treatment even when the underlying data-generating process is known.

## 8. References

- Athey, S., and Wager, S. (2021). Policy learning with observational data. *Econometrica*, 89(1), 133–161. https://doi.org/10.3982/ECTA15732
- Baier, D., and Stöcker, B. (2022). Profit uplift modeling for direct marketing campaigns: Approaches and applications for online shops. *Journal of Business Economics*, 92, 631–662. https://doi.org/10.1007/s11573-021-01068-3
- Chen, J., and Roth, J. (2024). Logs with zeros? Some problems and solutions. *The Quarterly Journal of Economics*, 139(2), 891–936. https://doi.org/10.1093/qje/qjad054
- Gubela, R. M., Lessmann, S., and Jaroszewicz, S. (2020). Response transformation and profit decomposition for revenue uplift modeling. *European Journal of Operational Research*, 283(2), 647–661. https://doi.org/10.1016/j.ejor.2019.11.030
- Gutierrez, P., and Gérardy, J.-Y. (2017). Causal inference and uplift modelling: A review of the literature. *Proceedings of the 3rd International Conference on Predictive Applications and APIs*, 1–13.
- He, B., Weng, Y., Tang, X., Cui, Z., Sun, Z., Chen, L., He, X., and Ma, C. (2024). Rankability-enhanced revenue uplift modeling framework for online marketing. arXiv:2405.15301. https://arxiv.org/abs/2405.15301
- Kim, C., Li, Y., Xu, T., and Liao, Z. (2024). Bayesian nonparametric model for heterogeneous treatment effects with zero-inflated data. *Statistics in Medicine*, 43(30), 5968–5982. https://doi.org/10.1002/sim.10266
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116
- Mullahy, J., and Norton, E. C. (2024). Why transform Y? The pitfalls of transformed regressions with a mass at zero. *Oxford Bulletin of Economics and Statistics*, 86(2), 417–447. https://doi.org/10.1111/obes.12583
- Nie, X., and Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076
- Oganisian, A., Mitra, N., and Roy, J. A. (2021). A Bayesian nonparametric model for zero-inflated outcomes: Prediction, clustering, and causal estimation. *Biometrics*, 77(1), 125–135. https://doi.org/10.1111/biom.13244
- Zhang, W., Li, J., and Liu, L. (2022). A unified survey of treatment effect heterogeneity modelling and uplift modelling. *ACM Computing Surveys*, 54(8), Article 162. https://doi.org/10.1145/3466818