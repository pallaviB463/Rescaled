# P12 — Criteo Semi-Synthetic Experiment

## 1. Objective

Evaluate causal treatment-effect estimation methods using real Criteo covariates and synthetic treatment assignments and outcomes. Synthetic data provide known ground-truth treatment effects for model evaluation.

## 2. Data source and preparation

- Source: Criteo Uplift Prediction Dataset, version 2.1.
- Input file: `data/raw/criteo-uplift-v2.1.csv.gz`.
- Sampled covariates: `data/processed/criteo_covariates_200k.csv`.
- Number of sampled rows: 200,000.
- Covariates: `f0` through `f11`.
- Sampling seed: 2026.
- Sampling procedure: streaming through the compressed input and retaining rows using reproducible random priorities.
- Rows with missing covariates are excluded.
- Covariates are standardized using the sampled dataset's feature means and standard deviations.

The original Criteo treatment and outcome columns are not used to generate the synthetic experiment.

## 3. Synthetic experiment design

| Parameter | Value |
|---|---|
| Scenario | `opposed` |
| Target control zero rate | 90% |
| Log-spend noise standard deviation | 0.8 |
| Treatment assignment probability | 0.5 |
| Treatment assignment seed | 2026 |
| Outcome generation seed | 2027 |

Treatment is assigned independently of the observed covariates using a Bernoulli distribution with probability 0.5.

The existing project data-generating process generates a purchase indicator and a nonnegative spending outcome. The opposed scenario defines heterogeneous treatment shifts in purchase log-odds and log-spend.

## 4. Output

The generated dataset is saved to `data/processed/criteo_semisynthetic_200k.csv`.

- `f0`–`f11`: standardized model features.
- `raw_f0`–`raw_f11`: original sampled covariates.
- `W`: synthetic binary treatment assignment.
- `Y`: synthetic nonnegative outcome.
- `tau_level`: ground-truth level-scale conditional average treatment effect.
- `p0`, `p1`: control and treatment purchase probabilities.

Generation settings are recorded in `data/processed/criteo_semisynthetic_manifest.json`.

## 5. Validation results

Initial validation on the generated sample:

- 200,000 rows and 12 standardized model features.
- No missing or non-finite values in the generated data.
- Treatment rate: 49.78%.
- Overall zero-outcome rate: 87.83%.
- Control purchase rate: 9.52%.
- Mean predicted control purchase probability: approximately 9.55%.
- Treated purchase rate: 14.84%.
- Mean predicted treated purchase probability: approximately 14.76%.
- Ground-truth CATE values are finite.
- Outcomes are nonnegative.
- 465 duplicate feature vectors and 152 duplicate complete records.

The observed purchase rates are consistent with the mean predicted probabilities. The nominal 90% control zero-rate target is approximate for this real-covariate sample because the existing intercept calibration uses a standard-normal reference distribution rather than the empirical Criteo feature distribution.

The ground-truth regression test verifies that the saved `tau_level` and `p1` values agree with the opposed-scenario equations.

## 6. Automated tests

The following checks have been implemented:

- Dataset dimensions and required columns.
- Missing and non-finite values.
- Binary treatment and nonnegative outcomes.
- Valid probabilities and finite ground-truth effects.
- Agreement between observed purchase rates and predicted probabilities.
- Standardization of model features.
- Reproducibility metadata.
- Ground-truth level-CATE equation consistency.

Initial results:
- P12 structural tests: 7 passed.
- Ground-truth equation test: 1 passed.
- Existing full suite before adding the ground-truth test: 91 passed.

Rerun the complete suite before final sign-off.

## 7. Limitations and reproducibility

This experiment uses real covariates but synthetic treatments and outcomes. It is not an evaluation on the original observed Criteo outcomes, and its results should be reported as semi-synthetic.

The empirical covariate distribution differs from the standard-normal reference distribution used by the existing DGP calibration. The control zero-rate target is therefore not guaranteed to hold exactly for every covariate sample.

The source dataset has its own license terms. Check those terms before redistributing the raw dataset or derived artifacts.

## 8. Completion status

The data generation and initial validation steps have succeeded. Final P12 sign-off requires rerunning the full test suite and reviewing the generated outputs and methodology together.
