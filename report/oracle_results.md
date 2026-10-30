# Oracle Ranking Analysis and Policy-Value Benchmarks

## 1. Objective

The oracle analysis investigates whether changing the outcome transformation and its positive scale changes treatment-effect rankings, even when transformed conditional average treatment effects (CATEs) are calculated directly from the data-generating process rather than estimated using fitted models.

For each transformation and scale, the analysis compares the transformed CATE ranking with the level-scale CATE ranking using:

- Spearman rank correlation.
- Top-20% intersection over union (IoU).

It also calculates the oracle top-20% policy value, defined as the mean true level-scale treatment effect among the 20% of units with the largest true level-scale treatment effects.

The purpose is to isolate transformation-induced differences in the estimand from model-fitting error.

## 2. Experimental design

### 2.1 Scenarios

The analysis covers all six simulation scenarios:

1. Null
2. Homogeneous
3. Extensive
4. Intensive
5. Aligned
6. Opposed

The opposed scenario is particularly useful for examining ranking sensitivity because its extensive and intensive treatment effects move in opposite directions across a covariate.

### 2.2 Zero-rate settings

The target control-group zero-rate parameter is varied across:

- 0.70
- 0.90
- 0.98

### 2.3 Transformations and scales

The four transformations are:

| Transformation | Definition |
|---|---|
| Identity | \(m(y)=y\) |
| Log1p | \(m(y)=\log(1+y)\) |
| Arcsinh | \(m(y)=\operatorname{arcsinh}(y)\) |
| Square root | \(m(y)=\sqrt{y}\) |

The positive outcome scales are:

\[
a\in\{0.001,0.01,0.1,1,10,100,1000\}.
\]

For every scenario and zero-rate combination, transformed CATEs are calculated at all four transformations and seven scales.

### 2.4 Simulation settings

| Parameter | Setting |
|---|---:|
| Observations per scenario/zero-rate combination | 50,000 |
| Number of scenarios | 6 |
| Number of zero rates | 3 |
| Total simulated observations across combinations | 900,000 |
| Log-spending noise parameter | 0.8 |
| Base random seed | 2026 |
| Policy selection fraction | 20% |

A deterministic, distinct seed is derived for each scenario/zero-rate combination.

The full ranking table contains:

\[
6\times3\times4\times7=504
\]

rows.

The oracle policy-value table contains:

\[
6\times3=18
\]

rows.

## 3. Evaluation metrics

### 3.1 Spearman rank correlation

For each transformation and scale, Spearman correlation measures the agreement between the transformed-CATE ranking and the level-CATE ranking.

A value of 1 indicates identical rankings, while lower values indicate increasing disagreement.

### 3.2 Top-20% intersection over union

The top-20% IoU measures the overlap between the units selected by the transformed-CATE ranking and those selected by the level-CATE ranking:

\[
\operatorname{IoU}(A,B)=
\frac{|A\cap B|}{|A\cup B|}.
\]

This complements rank correlation because strong overall rank correlation does not necessarily imply agreement among the highest-ranked treatment-effect groups.

### 3.3 Oracle policy value

The oracle policy value is calculated using the true level-scale CATE to select the top 20% of units:

\[
V_{\mathrm{oracle},20}
=
\frac{1}{|A^*|}
\sum_{i\in A^*}\tau_{\mathrm{level},i},
\]

where \(A^*\) is the set of units with the largest true level-scale treatment effects.

This is an oracle benchmark, not an estimated policy value from a fitted learner.

## 4. Main ranking findings

### 4.1 Identity transformation

The identity transformation preserves the level-CATE ranking across all positive scales.

For every non-null scenario and zero-rate combination:

- Spearman rank correlation = 1.
- Top-20% IoU = 1.

This confirms the expected positive-scale invariance of the identity transformation.

### 4.2 Square-root transformation

The square-root transformation is invariant to positive scale in the oracle analysis, but its ranking can differ from the level-CATE ranking.

For the opposed scenario:

| Zero rate | Spearman correlation | Top-20% IoU |
|---:|---:|---:|
| 0.70 | 0.627 | 0.314 |
| 0.90 | 0.865 | 0.593 |
| 0.98 | 0.909 | 0.707 |

These values are unchanged across the seven positive scales.

Thus, scale invariance does not imply equivalence to the level-scale estimand.

### 4.3 Log1p and arcsinh scale sensitivity

Both log1p and arcsinh exhibit scale-dependent ranking differences.

For the opposed scenario, the results at the smallest and largest scales are:

| Zero rate | Transformation | Rank correlation at 0.001 | Rank correlation at 1000 | IoU at 0.001 | IoU at 1000 |
|---:|---|---:|---:|---:|---:|
| 0.70 | Log1p | 0.998 | 0.317 | 0.958 | 0.142 |
| 0.70 | Arcsinh | 1.000 | 0.313 | 0.997 | 0.141 |
| 0.90 | Log1p | 0.999 | 0.739 | 0.974 | 0.445 |
| 0.90 | Arcsinh | 1.000 | 0.738 | 0.998 | 0.444 |
| 0.98 | Log1p | 0.999 | 0.819 | 0.984 | 0.580 |
| 0.98 | Arcsinh | 1.000 | 0.817 | 0.999 | 0.579 |

The results demonstrate that transformation-induced ranking disagreement depends on both the scale and the data-generating scenario.

### 4.4 Differences across scenarios

The opposed scenario exhibits particularly strong ranking changes at high scales.

For comparison, at zero rate 0.90 and scale 1000:

| Scenario | Log1p rank correlation | Log1p top-20% IoU |
|---|---:|---:|
| Homogeneous | 0.862 | 0.516 |
| Extensive | 0.981 | 0.716 |
| Intensive | 0.974 | 0.688 |
| Aligned | 0.983 | 0.747 |
| Opposed | 0.739 | 0.445 |

The opposed scenario produces the greatest disagreement among these five non-null scenarios in this comparison.

The extensive and intensive scenarios retain relatively high overall rank correlations, although their top-20% overlap can still be substantially below one.

These results show that the relationship between transformed and level-scale rankings depends on the structure of treatment-effect heterogeneity.

## 5. Oracle policy-value results

The following table reports the mean true level-scale treatment effect among the oracle-selected top 20% of units.

| Scenario | Zero rate 70% | Zero rate 90% | Zero rate 98% |
|---|---:|---:|---:|
| Aligned | 20.9627 | 11.7444 | 3.4480 |
| Extensive | 6.2001 | 4.0898 | 1.2924 |
| Homogeneous | 5.3184 | 2.8084 | 0.7613 |
| Intensive | 8.0345 | 2.8112 | 0.5821 |
| Null | 0.0000 | 0.0000 | 0.0000 |
| Opposed | 5.1321 | 2.8916 | 0.8899 |

The aligned scenario has the largest oracle policy value at all three zero rates.

The null scenario has exactly zero oracle policy value at every zero rate, as expected.

Oracle policy values generally decrease as the zero-rate parameter increases, although the magnitude of the change differs across scenarios.

These values are expressed in the level-scale treatment-effect units of the simulation and should not be interpreted as monetary values unless the outcome-generating process assigns monetary units.

## 6. Null scenario and metric interpretation

In the null scenario, the true level-scale CATE is constant at zero.

Consequently, there is no meaningful level-scale treatment-effect ranking to recover.

Rank correlation and top-20% IoU against this constant truth are therefore undefined and are stored as missing values in the CSV.

The oracle policy value remains exactly zero.

This distinction is important: undefined ranking metrics in the null scenario are not evidence of poor or good ranking performance.

When loading the CSV files with pandas, use:

```python
pd.read_csv("results/oracle_ranking_full_grid.csv", keep_default_na=False)
```

The explicit setting preserves the literal scenario label `null`, which pandas otherwise interprets as a missing-value token.

## 7. Interpretation and limitations

The oracle results isolate transformation-induced differences because the transformed CATEs are calculated from the DGP rather than estimated by fitted models.

The main findings are:

1. Positive scaling of the identity transformation preserves level-CATE rankings.
2. The square-root transformation is invariant to positive scaling but can produce rankings different from the level-scale CATE.
3. Log1p and arcsinh can induce substantial scale-dependent ranking changes.
4. The magnitude of ranking changes depends on the treatment-effect scenario and zero-rate parameter.
5. Overall rank correlation and top-20% overlap provide complementary information.
6. Oracle policy value provides a scenario-specific benchmark for subsequent learner evaluation.

The analysis is descriptive and uses one deterministic simulation run per scenario/zero-rate combination, with 50,000 observations in each. It does not quantify Monte Carlo uncertainty through repeated seeds or confidence intervals.

The oracle policy value is a benchmark based on true treatment effects. It does not establish that a fitted learner can achieve the same policy value.

## 8. Reproducibility artifacts

The full-grid analysis produces:

| Artifact | Description |
|---|---|
| `results/oracle_ranking_full_grid.csv` | 504 transformed-CATE ranking comparisons |
| `results/oracle_policy_values.csv` | 18 scenario/zero-rate oracle policy values |
| `results/figure1_oracle_aligned_zero_0p70.png` and corresponding files | 18 scenario/zero-rate ranking figures |

The 18 figures are generated using the `figure1_oracle_<scenario>_zero_<rate>.png` naming convention.

The earlier standalone `results/figure1_oracle_ranking.png` is retained as a legacy output and is not part of the 18-figure full-grid set.

To reproduce the experiment:

```powershell
python -m src.oracle_analysis
```

To run the oracle-analysis tests:

```powershell
python -m pytest tests/test_oracle_analysis.py -v
```

The expanded oracle-analysis tests passed 9/9, and the complete project test suite passed 157/157 in the recorded run.

**Review status:** P11 implementation, experiment execution, and output validation are complete. Q review and sign-off remain pending.