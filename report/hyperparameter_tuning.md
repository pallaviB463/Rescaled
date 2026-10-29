# Hyperparameter Tuning Protocol

## 1. Purpose

Define a reproducible and fair tuning procedure for comparing outcome models
used in treatment-effect estimation. The current simulation runner uses fixed
parameters; this document specifies the protocol to implement before making
tuned-model performance claims.

## 2. Models in scope

The current runner supports these outcome-model choices for the T-learner:

- L2: LightGBM squared-error regression.
- Tweedie: LightGBM Tweedie regression, currently using power 1.5.
- Hurdle: a two-part model comprising an occurrence classifier and a
  positive-outcome Gamma regression model.

The other meta-learners currently use L2 in the simulation runner. They should
not be described as independently tuned Tweedie or Hurdle experiments.

## 3. Data splitting and leakage prevention

For each scenario and simulation seed:

1. Generate the simulated dataset using the specified DGP parameters.
2. Split the data into training, validation, and test partitions using a
   reproducible 60% / 20% / 20% split.
3. Use the same partition assignments for every model being compared.
4. Fit all preprocessing and models using training data only.
5. Select hyperparameters using validation data only.
6. Evaluate the selected configuration on the held-out test set once.

Where treatment-effect ranking is evaluated, compute the ranking metrics on
the test covariates using the known DGP ground truth. Oracle truth must not be
used to select hyperparameters.

## 4. Search budget

Use 12 candidate configurations per model and per simulation seed.

- Generate candidate configurations using a reproducible random search.
- Use the same number of candidates for L2, Tweedie, and Hurdle.
- Fix and record the search seed.
- Record all attempted configurations, validation scores, failures, and the
  selected configuration.
- Do not increase one model's budget after inspecting its test performance.

Equal candidate counts provide a comparable search budget, although total
runtime may differ across model classes.

## 5. Initial search spaces

Use the following proposed candidate values:

| Parameter | Candidate values |
|---|---|
| `n_estimators` | 100, 200, 400 |
| `learning_rate` | 0.03, 0.05, 0.10 |
| `num_leaves` | 7, 15, 31 |
| `min_child_samples` | 20, 50, 100 |

Sample 12 valid configurations reproducibly from this space for each model.
Do not evaluate the full Cartesian product.

For Tweedie, additionally evaluate `power` from 1.2, 1.5, and 1.8.

For Hurdle, apply the selected LightGBM settings consistently to its
occurrence and positive-outcome components unless a separate component-level
search is explicitly specified in advance.

These ranges are proposed search spaces, not settings already evaluated.

## 6. Model selection metric

Use validation RMSE of the factual outcome prediction as the primary selection
metric, calculated on the same validation observations for every candidate.

For a T-learner, the factual prediction is the control-model prediction for
control observations and the treated-model prediction for treated observations.

Select the candidate with the lowest validation RMSE. Record the validation
sample size and score.

If a candidate fails to fit or produces non-finite predictions, record the
failure and do not silently replace its score with a favorable value.

## 7. Final evaluation

After selection, refit the selected configuration on the combined training
and validation partitions, then evaluate on the held-out test partition.

Report test outcome-prediction performance separately from treatment-effect
metrics, including:

- Rank correlation between estimated and true treatment effects.
- Top-20% ranking overlap, defined by the implementation's intersection over
  union metric.
- Policy value.
- Regret relative to the oracle policy.
- Ranking stability against the scale-1 reference where applicable.

Report the number of successful and failed candidates and the selected
hyperparameters for each model and seed.

## 8. Reproducibility requirements

Record the following for each experiment:

- Scenario, sample size, control zero-rate setting, and outcome noise.
- Simulation seed and search seed.
- Training, validation, and test split assignments or their reproducible seeds.
- Candidate configurations and validation scores.
- Selected hyperparameters and test metrics.
- Software versions and any failed runs.

Use multiple simulation seeds for final robustness claims. A single-seed run
is an initial experiment, not evidence of multi-seed stability.

## 9. Implementation status

Status: protocol specified; tuning implementation and tuned experiments are
pending.

The current runner uses fixed LightGBM settings and fixed Tweedie power.
This document must not be interpreted as evidence that the proposed search
has already been executed.
