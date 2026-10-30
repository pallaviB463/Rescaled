
# Hyperparameter Tuning Protocol and Results

## 1. Purpose

This experiment implements a reproducible hyperparameter-tuning procedure for comparing outcome models used in treatment-effect estimation.

The objective is to provide L2, Tweedie, and Hurdle models with equal candidate budgets, use validation data for model selection, and reserve the test set for final evaluation.

The tuning experiment is separate from the main transformation-scale simulation runner.

## 2. Models in Scope

The tuning implementation supports three outcome models:

- **L2:** LightGBM squared-error regression.
- **Tweedie:** LightGBM Tweedie regression with a tunable variance power.
- **Hurdle:** A two-part model consisting of an occurrence classifier and a Gamma regression model for positive outcomes.

The tuning implementation evaluates these models as arm-specific outcome models for the T-learner.

The other meta-learners are not independently tuned as Tweedie or Hurdle experiments. Their pseudo-outcomes may contain negative values, so the main implementation retains L2 for those learners.

## 3. Data Splitting and Leakage Prevention

The tuning experiment uses a reproducible:

- 60% training partition.
- 20% validation partition.
- 20% held-out test partition.

The same split is used across the outcome models being compared.

The procedure is:

1. Generate the simulated dataset.
2. Split observations into training, validation, and test partitions.
3. Fit candidate models using training data.
4. Select hyperparameters using validation performance.
5. Refit the selected configuration on the combined training and validation observations.
6. Evaluate the selected model on the held-out test partition.

The test set is not used for hyperparameter selection.

## 4. Search Budget and Candidate Spaces

Each model receives 12 candidate configurations in the documented experiment.

The base search space is:

| Parameter | Candidate values |
|---|---|
| `n_estimators` | 100, 200, 400 |
| `learning_rate` | 0.03, 0.05, 0.10 |
| `num_leaves` | 7, 15, 31 |
| `max_depth` | 3, 5, 7, -1 |
| `min_child_samples` | 20, 50, 100 |

Tweedie additionally searches over:

| Parameter | Candidate values |
|---|---|
| `power` | 1.1, 1.3, 1.5, 1.7 |

The implementation uses reproducible candidate generation and records candidate configurations, validation scores, selection metrics, and failures.

Equal candidate counts provide a comparable search budget in terms of the number of configurations evaluated. They do not imply equal computational cost across model classes.

## 5. Model Selection Metrics

Model selection uses validation performance:

- **L2:** Validation RMSE.
- **Hurdle:** Validation RMSE.
- **Tweedie:** Mean Tweedie deviance.

For the T-learner, factual predictions are obtained from the relevant treatment-arm outcome model.

The selection metrics differ across model classes. In particular, Tweedie deviance and RMSE are not directly comparable numerical scores. Model selection is performed within each model class using its specified metric.

Failed candidates and non-finite results are recorded rather than silently treated as successful evaluations.

## 6. Documented Validation Experiment

A focused validation experiment was executed with the following settings:

| Setting | Value |
|---|---|
| Scenario | Opposed |
| Sample size | 5,000 |
| Control zero-rate target | 0.90 |
| Positive-outcome log-scale standard deviation | 0.8 |
| Treatment propensity | 0.5 |
| Simulation seed | 42 |
| Candidate budget per model | 12 |
| Outcome models | L2, Tweedie, Hurdle |

The experiment command was:

```powershell
python -m src.run_tuning --n 5000 --zero-rate 0.9 --sigma 0.8 --e 0.5 --scenarios opposed --models L2 Tweedie Hurdle --seeds 42 --candidates 12 --output results/p8_tuning_validation.csv
```

### 6.1 Candidate Evaluation Results

| Model | Successful candidates | Failed candidates | Selected validation metric |
|---|---:|---:|---:|
| L2 | 12 | 0 | RMSE = 10.323362 |
| Tweedie | 12 | 0 | Tweedie deviance = 8.034729 |
| Hurdle | 12 | 0 | RMSE = 10.254938 |

All 36 candidate evaluations completed successfully.

The selected Tweedie configuration used power 1.3.

### 6.2 Held-Out Test RMSE

| Model | Test RMSE |
|---|---:|
| L2 | 13.144430 |
| Tweedie | 13.139679 |
| Hurdle | 13.169340 |

These test RMSE values describe factual outcome prediction in this particular experiment. They should not be interpreted as a comprehensive ranking of treatment-effect estimation quality.

Because the models use different validation selection metrics, the validation scores in Section 6.1 should not be compared directly across model classes.

## 7. Interpretation and Limitations

The validation experiment establishes that the implemented tuning pipeline can evaluate all three outcome models under an equal candidate-count budget and complete the train-validation-test workflow without candidate failures.

The test RMSE values are close in this single experiment. This result alone does not establish that one model is consistently better than another.

The experiment is limited to:

- One treatment-effect scenario: opposed.
- One sample size: 5,000.
- One control zero-rate target: 0.90.
- One simulation seed: 42.
- Twelve candidate configurations per model.

Consequently, these results are a validation run, not evidence of multi-scenario or multi-seed robustness.

The optional comparison of Gamma and lognormal positive-outcome components within the Hurdle model has not been implemented as a separate tuning experiment and should not be claimed as completed.

## 8. Reproducibility and Output Files

The experiment records candidate-level history and summary results.

Primary output:

- `results/p8_tuning_validation.csv`

The candidate history contains the attempted configurations, validation metrics, selection metric, and run status. The summary records the selected model configurations and their evaluation results.

The implementation is maintained in:

- `src/tuning.py`
- `src/run_tuning.py`
- `tests/test_tuning.py`
- `tests/test_run_tuning.py`

## 9. Implementation Status

**Status: Implemented and validation experiment completed.**

The tuning implementation and its tests are complete. The documented validation run evaluated 12 candidates each for L2, Tweedie, and Hurdle, with all 36 candidate evaluations successful.

Broader robustness claims require additional scenarios and simulation seeds.
