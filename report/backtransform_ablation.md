# P13: Back-Transformation Baseline and Ablation

## 1. Objective

P13 evaluates a naive back-transformation baseline for estimating conditional treatment effects from log-transformed outcomes.

The baseline is compared against the existing `TransformedOutcomeLearner` to assess predictive accuracy, ranking quality, and policy performance on the original outcome scale.

## 2. Methods

### 2.1 Transformed-outcome learner

The existing transformed-outcome learner uses:

\[
\tilde{Y}=\log(1+Y)
\]

and constructs the pseudo-outcome:

\[
Z=\frac{W-e}{e(1-e)}\log(1+Y)
\]

A model is fitted to estimate the conditional treatment effect.

### 2.2 Back-transform baseline

The new `BackTransformLearner` fits separate control and treatment outcome models using log-transformed outcomes:

\[
\hat f_w(X)\approx E[\log(1+Y)\mid X,W=w]
\]

The predictions are inverse-transformed:

\[
\hat{\mu}_w(X)=\exp(\hat f_w(X))-1
\]

The estimated treatment effect is:

\[
\hat{\tau}_{BT}(X)=
\exp(\hat f_1(X))-\exp(\hat f_0(X))
\]

Both models use the existing L2 LightGBM regression implementation.

## 3. Experimental Design

| Parameter | Configuration |
|---|---|
| Scenario | Opposed |
| Sample size per seed | 5,000 |
| Zero-rate parameter | 0.90 |
| Lognormal sigma | 0.8 |
| Treatment propensity | 0.5 |
| Random seeds | 42, 123, 2026, 2027, 2028 |
| Training fraction | 80% |
| Test fraction | 20% |
| Policy selection fraction | Top 20% |
| Outcome model | L2 LightGBM |
| Transformation | log1p |
| Evaluation scale | Original outcome / level CATE scale |

For every seed, both learners use the same simulated dataset, training observations, and held-out test observations.

The true level-scale conditional treatment effect from the DGP is used as the evaluation reference.

## 4. Evaluation Metrics

- **CATE RMSE:** Measures prediction error against the true level-scale conditional treatment effect.
- **Spearman rank correlation:** Measures agreement between predicted and true treatment-effect rankings.
- **Top-20% overlap:** Intersection-over-union between predicted and oracle top-20% selections.
- **Policy value:** Mean true treatment effect among units selected by the estimated policy.
- **Oracle regret:** Difference between oracle policy value and estimated-policy value.

All metrics are evaluated on the held-out test observations.

## 5. Results

### 5.1 Aggregate results across five seeds

Values are reported as mean ± standard deviation.

| Metric | TransformedOutcome | BackTransform |
|---|---:|---:|
| CATE RMSE ↓ | 1.4545 ± 0.0681 | **1.3935 ± 0.0755** |
| Spearman rank correlation ↑ | 0.2873 ± 0.0297 | **0.3716 ± 0.0416** |
| Top-20% overlap ↑ | 0.2942 ± 0.0470 | **0.3527 ± 0.0482** |
| Policy value ↑ | 1.8462 ± 0.1460 | **2.0419 ± 0.1796** |
| Oracle regret ↓ | 1.0527 ± 0.1740 | **0.8570 ± 0.1600** |

The average oracle policy value was 2.8989 across the five seeds.

### 5.2 Interpretation

The back-transform baseline achieved better average performance on all five evaluation metrics in this experiment.

Its mean CATE RMSE was approximately 4.2% lower than that of the transformed-outcome learner.

Its Spearman rank correlation improved from 0.2873 to 0.3716, indicating better agreement with the true treatment-effect ranking.

Top-20% overlap improved from 0.2942 to 0.3527. The corresponding policy value increased from 1.8462 to 2.0419.

Mean oracle regret decreased from 1.0527 to 0.8570, an improvement of approximately 18.6%.

These results indicate that the naive inverse-transform baseline performed better under the evaluated opposed-scenario configuration.

## 6. Methodological Limitations

### 6.1 Jensen's inequality

The inverse transformation does not generally recover the conditional arithmetic mean:

\[
\exp(E[\log(1+Y)\mid X])-1
\neq E[Y\mid X]
\]

Therefore, the baseline should not be interpreted as an unbiased conditional-mean estimator.

### 6.2 Prediction constraints

The L2 models are unconstrained regressors. Consequently, inverse-transformed outcome predictions can occasionally be negative.

The implementation does not silently clip these predictions. The negative-prediction fraction recorded in the experiment refers to negative estimated treatment effects, which are valid, rather than negative predicted outcomes.

### 6.3 Scope of the comparison

The comparison uses five simulation seeds, one scenario, one zero-rate parameter, one sample size, and a fixed L2 model configuration.

It does not establish that back-transformation is superior across all DGP scenarios, transformations, sample sizes, or model families.

The two approaches also differ in their estimation strategies, so the comparison is an estimator-level ablation rather than an isolated test of the inverse transformation alone.

## 7. Reproducibility

Experiment command:

```powershell
python -m src.run_backtransform_ablation --scenario opposed --n 5000 --zero-rate 0.9 --sigma 0.8 --e 0.5 --seeds 42 123 2026 2027 2028 --test-size 0.2 --output results/p13_backtransform_ablation.csv
```

Output:

- `results/p13_backtransform_ablation.csv`

Implementation:

- `src/metalearners.py`
- `src/run_backtransform_ablation.py`

Tests:

- `tests/test_metalearners.py`
- `tests/test_backtransform_ablation.py`

Validation:

- Meta-learner tests: 11 passed.
- P13 experiment tests: 3 passed.
- Full project suite: 163 passed in 23.02 seconds.

## 8. Conclusion

The back-transform baseline achieved lower mean CATE RMSE, stronger ranking agreement, greater top-20% policy value, and lower oracle regret than the transformed-outcome learner in the evaluated opposed scenario.

The findings support including the naive back-transform estimator as a comparative baseline. Further experiments across scenarios and seeds are required before making broader claims about its effectiveness.