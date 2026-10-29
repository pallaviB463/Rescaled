# P2 — Research-Gap Table

| Literature area | What prior work establishes | Remaining question for this project | Planned investigation |
|---|---|---|---|
| Meta-learners (Künzel et al., 2019) | Flexible approaches for estimating heterogeneous treatment effects; relative performance depends on the data-generating structure. | How sensitive are treatment-effect rankings to the outcome transformation and its scale? | Compare the implemented learners under common simulation scenarios and evaluation splits. |
| CATE estimation (Nie and Wager, 2021) | Flexible CATE estimation can be structured around nuisance estimation and treatment-effect optimization. | How do transformations interact with the outcome model and ranking objective? | Compare level-scale and transformed-scale estimates against known oracle effects. |
| Policy learning (Athey and Wager, 2021) | Estimated causal effects can inform treatment-allocation policies and constrained targeting. | Do transformation-driven ranking changes alter the value of targeting a fixed top fraction? | Evaluate top-20% overlap, policy value, and regret. |
| Uplift surveys (Gutierrez and Gérardy, 2017; Zhang et al., 2022) | Uplift modeling and heterogeneous treatment-effect estimation share a causal framework and related targeting goals. | How should transformation and scale be assessed jointly for nonnegative, zero-inflated outcomes? | Conduct a controlled simulation study across six treatment-effect scenarios and multiple scales. |
| Count-aware outcome modeling | Specialized outcome models can represent nonnegative outcomes and their distributional features. | Do the observed ranking conclusions depend on the outcome-model family? | Compare L2, Poisson, Tweedie, and Hurdle approaches under a documented, fair tuning protocol. |
| Zero-inflated causal outcomes (Oganisian et al., 2021; Kim et al., 2024) | Flexible causal-effect estimation for zero-inflated outcomes has been studied. | How does changing the outcome transformation and scale affect rankings relative to the level-scale oracle and the resulting targeting decisions? | Evaluate rank correlation, top-20% overlap, policy value, and regret across transformations, scales, and DGP scenarios. |

## Proposed contribution

The proposed contribution is a controlled evaluation of transformation-and-scale sensitivity for CATE ranking and treatment targeting in a zero-inflated outcome setting. It combines known oracle effects, multiple data-generating scenarios, scale-equivariance checks, and decision-oriented metrics.

## Novelty caveat

This table states the project's research questions, not a proven claim that the questions have never been studied. The novelty claim must be revised after a targeted literature search for transformed causal estimands, zero-inflated/count outcomes, and ranking stability. The current sources establish the broader CATE and policy-learning context but do not settle that narrower novelty question.
