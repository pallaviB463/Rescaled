# Sanity Audit

## 1. Objective

P14 validates the mathematical and implementation-level sanity properties of the Neurolytics treatment-effect estimation pipeline.

The audit covers:
- Null treatment-effect ground truth.
- Transformed-outcome estimands under the null.
- Scale invariance controls.
- Model prediction validity.
- Empirical prediction behaviour under repeated null simulations.

## 2. Null Ground-Truth Validation

The null DGP sets both treatment shifts to zero:

- Purchase log-odds shift: 0.
- Log-spend shift: 0.

Consequently:

\[
p_1(X)=p_0(X),\qquad \mu_1(X)=\mu_0(X)
\]

and the exact level-scale treatment effect is zero for every observation.

The audit verifies:
- Exact zero level-scale CATE.
- Equal treatment and control purchase probabilities.
- Equal treatment and control log-spend means.
- Zero transformed CATE for identity, log1p, arcsinh and square-root transformations.

All corresponding tests passed.

## 3. Transformation Invariance

The audit evaluates the seven scales:

\[
a\in\{0.001,0.01,0.1,1,10,100,1000\}
\]

The following mathematical controls were validated:

- Identity: \(m(ay)=a\,m(y)\).
- Square root: \(m(ay)=\sqrt{a}\,m(y)\).
- Rank: positive rescaling preserves ranks.

Identity and square-root rankings were invariant across all seven scales. Rank transformation values were also invariant under positive rescaling.

The repeated-seed fitted experiment reported `rho_vs_a1 = 1.0` for identity and square-root transformations at every scale.

Rank was evaluated once at scale 1.0, consistent with the pipeline configuration.

Log1p and arcsinh are not scale-equivariant. Their fitted rankings varied across scales.

## 4. Model Prediction Validity

The audit evaluated L2, Poisson, Tweedie and Hurdle models.

Verified properties:
- Predictions have the expected shape.
- Predictions are finite.
- Poisson, Tweedie and Hurdle outcome predictions are nonnegative.

Negative estimated treatment effects were not treated as invalid because treatment-effect differences can legitimately be negative.

All model prediction validity tests passed.

## 5. Repeated-Seed Null Simulation

The null simulation used:

| Parameter | Value |
|---|---:|
| Sample size | 5,000 |
| Target control zero rate | 90% |
| Log-spend noise SD | 0.8 |
| Treatment propensity | 0.5 |
| Seeds | 42, 123, 2026, 2027, 2028 |

The experiment covered:
- S-, T-, X- and DR-learners.
- Transformed-outcome learners.
- Identity, log1p, arcsinh, square-root and rank transformations.
- Seven scales for non-rank transformations.

Results:
- 165 successful configurations.
- 0 failed configurations.
- Null ranking correlations and top-20% overlap were undefined, as expected for constant zero ground truth.
- Policy value and regret were zero.

## 6. Prediction-Magnitude Audit

A separate audit measured fitted CATE magnitudes under the null across five seeds.

It included:
- Four standard meta-learners.
- All transformed-outcome configurations.
- The P13 BackTransform baseline.

Results:
- 170 successful runs.
- 0 failed runs.

### Reference-scale results

Mean absolute estimated CATE averaged across five seeds:

| Learner/configuration | Mean absolute CATE |
|---|---:|
| S-learner | 0.1554 |
| T-learner | 2.2218 |
| X-learner | 1.6496 |
| DR-learner | 2.2214 |
| BackTransform | 0.3668 |
| TransformedOutcome — identity | 2.5063 |
| TransformedOutcome — log1p | 0.2569 |
| TransformedOutcome — arcsinh | 0.3118 |
| TransformedOutcome — sqrt | 0.4081 |
| TransformedOutcome — rank | 0.1512 |

The fitted learners produced nonzero estimates despite the exact zero ground truth. Their magnitudes varied across learner families and transformations.

This is reported as finite-sample estimation behaviour, not as evidence of a coding defect or as proof of statistical significance.

Identity and square-root prediction magnitudes changed with outcome scale in accordance with their mathematical scaling properties.

## 7. Limitations

- The repeated-seed experiment uses five seeds and is descriptive.
- No formal statistical significance or calibration test was performed.
- Nonzero estimated CATEs under the null are not expected to be exactly zero in finite samples.
- The null evaluation metrics alone do not establish that fitted predictions are negligible.
- Q review and sign-off remain separate from implementation and test completion.

## 8. Conclusion

The P14 mathematical invariants, null ground-truth checks, transformation controls and model prediction validity checks passed.

The repeated-seed simulations completed without failures. However, prediction-magnitude results show variation across learner families, particularly for T-, X- and DR-learners and the identity transformed-outcome learner.

These findings are retained for Q review. The audit does not claim that all fitted null effects are negligible or statistically insignificant.