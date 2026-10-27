# Hillstrom Uplift Modeling and Evaluation

## Project Overview

This project evaluates uplift modeling using the Hillstrom men's email marketing experiment. The objective is to determine whether machine learning models can identify customers who benefit from a marketing campaign and whether targeting customers with the highest predicted uplift performs better than random targeting.

The project includes data preparation, outcome modeling, uplift prediction, repeated-split evaluation, data-integrity audits, and nested cross-validation.

## Research Question

Does targeting the 20% of customers with the highest predicted uplift produce a positive incremental outcome value compared with random targeting?

## Dataset

The analysis uses the filtered Hillstrom men's email dataset.

- **Total customers:** 42,613
- **Treatment group:** 21,307
- **Control group:** 21,306
- **Treatment propensity:** Approximately 0.5
- **Primary outcome:** Customer spend
- **Additional outcomes:** Visit and conversion

## Methodology

For each customer, predicted uplift is calculated as:

\[
\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
\]

where:

- \(\hat{\mu}_1(x)\) is the predicted outcome under treatment.
- \(\hat{\mu}_0(x)\) is the predicted outcome under control.
- \(\hat{\tau}(x)\) is the estimated incremental effect of treatment.

Customers are ranked by predicted uplift, and the top 20% are selected for policy evaluation. Inverse-propensity weighting is used to estimate policy value based on the randomized treatment assignment.

### Models Evaluated

Six model and transformation configurations were evaluated for customer spend:

1. L2 with identity transformation
2. L2 with log1p transformation
3. Tweedie with identity transformation
4. Tweedie with log1p transformation
5. Hurdle with identity transformation
6. Hurdle with log1p transformation

Repeated random train/test splits were used for the initial comparison, followed by nested cross-validation with five outer folds and three inner folds.

## Evaluation Results

### Initial Repeated-Split Evaluation

L2 with log1p produced the strongest initial mean estimated targeting advantage over random targeting.

- **Mean advantage:** Approximately +$0.0217 per customer
- **Positive splits:** 13 out of 20
- **Customer-cluster bootstrap 95% confidence interval:** Approximately −$0.0805 to +$0.1252 per customer

The confidence interval includes zero. Therefore, the initial evaluation does not establish a statistically reliable positive targeting advantage.

### Nested Cross-Validation

For the fixed L2 + log1p configuration:

- **Outer folds:** 5
- **Inner folds:** 3
- **Mean outer-fold targeting advantage:** Approximately −$0.0542 per customer
- **Standard deviation across outer-fold advantages:** Approximately $0.1263 per customer
- **Positive outer folds:** 2 out of 5
- **Mean policy value:** Approximately $0.0998 per customer
- **Mean random-targeting value:** Approximately $0.1540 per customer

The model selected through inner-fold evaluation varied across the outer folds. This suggests that model selection and performance were not stable across the evaluated splits.

The standard deviation reported here is descriptive and is not a confidence interval.

## Key Findings

- L2 + log1p showed the strongest initial repeated-split point estimate.
- The initial confidence interval included zero.
- Nested cross-validation produced a negative mean outer-fold targeting advantage for the fixed L2 + log1p configuration.
- The model's performance varied across outer folds.
- The current results do not establish that uplift-based targeting reliably outperforms random targeting.

## Limitations

1. **Overlapping test sets:** The repeated test splits share customers, so their results are not independent.
2. **Conditional bootstrap:** The bootstrap analysis conditions on fitted models and predictions and does not capture all model-training or split-generation uncertainty.
3. **Model selection:** The dataset was already used for earlier model comparisons, so nested cross-validation is not an independent external validation.
4. **Fold variability:** The five outer folds show substantial variability, and the selected model changes across folds.
5. **Generalization:** The results apply to this dataset and evaluation setup and should not automatically be generalized to future marketing campaigns.

## Project Structure

The following is an indicative project structure. Actual filenames may vary depending on the scripts and outputs present in the repository.

```text
Rescaled/
├── src/
│   ├── hillstrom_validation.py
│   ├── hillstrom_evaluation_audit.py
│   ├── hillstrom_split_integrity_audit.py
│   ├── hillstrom_final_validation_report.py
│   └── hillstrom_nested_cv.py
└── results/
    └── hillstrom/
        ├── hillstrom_test_predictions.csv
        ├── hillstrom_evaluation_audit_by_split.csv
        ├── hillstrom_evaluation_audit_summary.csv
        ├── hillstrom_test_split_overlap.csv
        ├── hillstrom_final_validation_report.md
        ├── hillstrom_final_validation_policy_summary.csv
        ├── hillstrom_nested_cv_outer_results.csv
        ├── hillstrom_nested_cv_inner_selection.csv
        ├── hillstrom_nested_cv_summary.csv
        ├── hillstrom_final_evaluation_report.md
        └── hillstrom_independent_validation_protocol.md
```

## Reproducibility

- Run scripts from the project root so relative paths resolve correctly.
- Preserve the random seeds and dataset preparation steps.
- Keep the scripts used to generate the results.
- Do not manually edit generated prediction or summary CSV files.
- Record the Python and package versions used for the final run.
- Verify that the reported numbers match the generated result files.
- Treat repeated-split estimates cautiously because the test sets overlap.

## Conclusion

This project implements and audits an uplift-model evaluation workflow using the Hillstrom men's email dataset. L2 + log1p was a promising candidate in the initial repeated-split analysis, but its confidence interval included zero and nested cross-validation produced a negative mean outer-fold targeting advantage.

The available evidence does not establish that uplift-based targeting reliably outperforms random targeting. The current findings should therefore be presented as an exploratory evaluation.

**Future work:** Evaluate a frozen model on genuinely new campaign data or through a suitably designed prospective randomized experiment before making a deployment or business-impact claim.