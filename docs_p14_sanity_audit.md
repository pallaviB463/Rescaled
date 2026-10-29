# P14 — Sanity Audit Report

## 1. Objective

Review the correctness, validation, and reproducibility of selected components implemented for the Neurolytics project, with particular attention to the data-generating process, evaluation metrics, Criteo semi-synthetic experiment, and naive log1p back-transformation baseline.

## 2. Audit Scope

The review covered:

- `src/dgp.py`
- `src/transforms.py`
- `src/metrics.py`
- `src/backtransform.py`
- `src/run_backtransform.py`
- `src/validate_criteo_semisynthetic.py`
- `src/prepare_criteo_semisynthetic.py`
- Criteo, metric, DGP, and back-transformation tests

## 3. Findings

### 3.1 Data-Generating Process

The reviewed level-scale treatment-effect formula is consistent with the specified purchase-and-spending DGP:

\[
\tau(x)=p_1(x)e^{\mu_1(x)+\sigma^2/2}
-p_0(x)e^{\mu_0(x)+\sigma^2/2}.
\]

The DGP tests check scenario behavior, outcome validity, reproducibility, supplied-covariate handling, and agreement between theoretical and simulated treatment effects.

**Limitation:** The control purchase-probability intercept is calibrated against a standard-normal reference distribution. Therefore, the nominal control zero-rate target is not guaranteed to hold exactly on the empirical Criteo covariate distribution.

### 3.2 Transformations and Evaluation Metrics

The transformation tests cover the supported transformations, scale handling, rank invariance, ties, invalid inputs, and finite outputs.

Two additional metric tests were added during P14:

- `test_topk_overlap_matches_intersection_over_union`: verifies that top-k overlap is the intersection-over-union of the selected sets.
- `test_policy_regret_is_positive_for_suboptimal_selection`: verifies policy value and regret for a deliberately suboptimal selection.

The metric test module passed all 12 tests.

**Interpretation note:** `topk_overlap` represents Jaccard overlap, not the fraction of the oracle's top-k group recovered. These quantities should not be described interchangeably.

A complete mathematical review of every metric, including the IPW policy-value estimator and Qini-style summary, remains a separate follow-up item.

### 3.3 Back-Transformation Baseline

The reviewed implementation:

- Applies `expm1` to invert predictions on the log1p scale.
- Validates input shape and finiteness.
- Handles overflow explicitly.
- Supports optional clipping of negative back-transformed predictions.
- Evaluates estimated effects against level-scale ground truth.

The experiment runner fits separate models for the treatment and control groups and evaluates predictions on a separately simulated test dataset.

**Statistical limitation:** In general,

\[
\operatorname{expm1}\!\left(
\mathbb{E}[\log(1+Y)\mid X]
\right)
\neq \mathbb{E}[Y\mid X].
\]

The method is therefore a naive back-transformation baseline rather than an unbiased estimator of the original-scale conditional mean. This limitation is documented in the implementation.

### 3.4 Criteo Semi-Synthetic Experiment

The generator uses standardized real Criteo covariates while generating synthetic randomized treatment assignments and outcomes. The original observed treatment and outcome fields are not used.

The validation and regression tests cover:

- Expected row count and required columns.
- Missing and non-finite values.
- Binary treatment and nonnegative outcomes.
- Valid purchase probabilities.
- Approximate standardization of model features.
- Reproducibility metadata.
- Agreement between observed purchase rates and average predicted probabilities.
- Consistency of stored ground-truth effects with the opposed-scenario equations.

The Criteo-specific test run passed **8 tests in 4.41 seconds**.

The documentation correctly identifies this as a semi-synthetic experiment, not an evaluation on the original observed Criteo outcomes.

## 4. Test Results

The complete test suite passed:

- **107 passed**
- **18.42 seconds**
- No failures or skipped tests reported in the test output

The metric sanity tests were committed in Git commit `701abcd` with message `test: strengthen causal metric sanity checks`.

The working tree was clean immediately after the commit.

## 5. Outstanding Limitations and Follow-Up

1. Complete the mathematical review of `ipw_policy_value` and `qini_auc`, including their estimands, normalization, and interpretation.
2. Consider stronger parameter validation in `run_backtransform`, particularly integer validation for `n`.
3. Verify that all experiment documentation clearly distinguishes preliminary results from replicated comparisons.
4. Keep the Criteo calibration caveat visible when interpreting results.
5. Continue the remaining Member P deliverables separately; a passing test suite does not establish completion of missing documentation or manuscript sections.

## 6. Conclusion

The reviewed components passed their relevant tests, including the full project test suite. The audit identified no demonstrated test failure in the reviewed areas. However, passing tests alone do not establish complete methodological correctness. The metric estimators identified above and remaining project deliverables require further review before the overall Member P work can be considered fully signed off.
