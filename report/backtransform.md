# P13 — Naive Log1p Back-Transformation Baseline

## Objective

Evaluate a naive back-transformation baseline for estimating conditional treatment effects when outcomes are zero-inflated and nonnegative.

## Method

1. Generate training and independent test data using the existing DGP.
2. Split training observations into control (\(W=0\)) and treatment (\(W=1\)) groups.
3. Fit separate LightGBM squared-error regression models to `log1p(Y)` in each group.
4. Predict transformed outcomes for the test observations.
5. Apply `expm1` to both sets of predictions, clipping negative back-transformed predictions to zero.
6. Estimate the level-scale treatment effect as the difference between the back-transformed treated and control predictions.
7. Evaluate the estimated effects against the known level-scale CATE using rank correlation, top-20% overlap, policy value, and regret.

## Experimental Configuration

- Scenario: `opposed`
- Training sample size: 1,000
- Random seed: 42
- Control zero-rate target: 0.90
- Outcome noise parameter: 0.8
- Treatment assignment probability: 0.5
- Test sample size: 2,000, as specified by the experiment runner

## Results

| Metric | Naive back-transformation | Level-scale T-learner |
|---|---:|---:|
| Rank correlation | 0.388257 | 0.133217 |
| Top-20% overlap | 0.374570 | 0.178203 |
| Policy value | 2.022257 | 1.180922 |
| Regret | 0.771362 | 1.612696 |

In this replicate, the naive back-transformation baseline has higher rank correlation, higher top-20% overlap, higher policy value, and lower regret than the level-scale T-learner.

These findings are preliminary and apply only to the specified scenario, sample size, and random seed.

## Statistical Limitation

In general,

\[
\operatorname{expm1}\!\left(
\mathbb{E}[\log(1+Y)\mid X]
\right)
\neq \mathbb{E}[Y\mid X].
\]

Consequently, exponentiating a fitted conditional mean on the transformed scale does not generally recover the conditional mean on the original outcome scale. The method is therefore treated as a naive baseline rather than an unbiased estimator of the level-scale conditional mean or treatment effect.

## Reproducibility

Run the baseline:

```powershell
.\.venv\Scripts\python.exe -c "from src.run_backtransform import run_backtransform; r = run_backtransform(scenario='opposed', n=1000, seed=42); print(r.to_string(index=False)); r.to_csv('results/backtransform_baseline.csv', index=False)"
```

Run the level-scale comparison:

```powershell
.\.venv\Scripts\python.exe -c "from src.run_sim import one_rep; r = one_rep(scenario='opposed', n=1000, seed=42, learners=('T',), include_transformed=False); print(r.to_string(index=False)); r.to_csv('results/level_t_learner_comparison.csv', index=False)"
```

The output CSVs are generated locally and are subject to the repository's results-directory ignore rules.
