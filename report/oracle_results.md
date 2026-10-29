# Oracle Analysis: Transformation-Induced Ranking Changes

## 1. Objective

The oracle analysis investigates whether changing the outcome transformation and its positive scale changes the treatment-effect ranking, even when the transformed conditional average treatment effects (CATEs) are calculated using the data-generating process rather than estimated by a fitted model.

The analysis compares each transformed CATE ranking with the level-scale CATE ranking. It evaluates Spearman rank correlation and top-20% intersection over union (IoU).

## 2. Experimental setup

The analysis uses a shared covariate sample of 50,000 observations, with random seed 2026 and log-spending noise parameter \(\sigma=0.8\). The zero-rate parameter is varied across 0.70, 0.90, and 0.98 in the robustness analysis.

The transformations include identity, square root, log1p, and arcsinh. Positive scales range from \(10^{-3}\) to \(10^3\) in the main oracle analysis. The consolidated robustness table retains five scales: \(10^{-3}\), \(10^{-1}\), \(1\), \(10\), and \(10^3\).

For each setting, the transformed CATE is calculated from the expected transformed outcome under treatment and control. Its ranking is then compared with the ranking of the level-scale CATE.

## 3. Main findings

### 3.1 Identity and square-root transformations

The identity transformation preserves the level-CATE ranking across positive scales, producing a rank correlation of 1 and a top-20% IoU of 1 in the non-null scenarios.

The square-root transformation is also invariant to positive scale in this analysis. However, its ranking need not coincide with the level-CATE ranking. In the opposed scenario, its rank correlation with the level CATE is approximately 0.866, and its top-20% IoU is approximately 0.589.

### 3.2 Scale sensitivity of log1p and arcsinh

The log1p and arcsinh transformations show scale-dependent ranking differences. In the opposed scenario with a zero-rate parameter of 0.90, the log1p rank correlation decreases from approximately 0.999 at scale \(10^{-3}\) to 0.742 at scale \(10^3\). Its top-20% IoU decreases from approximately 0.975 to 0.449.

For arcsinh, the corresponding rank correlation decreases from approximately 1.000 to 0.740, while top-20% IoU decreases from approximately 0.997 to 0.447.

These results show that high rank correlation does not necessarily imply strong agreement among the highest-ranked treatment-effect groups. The top-20% overlap provides a complementary measure of targeting disagreement.

### 3.3 Heterogeneity scenario matters

The magnitude of ranking changes depends on the treatment-effect scenario. In the aligned scenario, log1p at scale \(10^3\) retains a rank correlation of approximately 0.983, but top-20% IoU falls to approximately 0.744. In the opposed scenario, the same transformation and scale produce substantially larger ranking differences.

Thus, transformation-induced ranking changes are not uniform across data-generating processes.

## 4. Robustness to the zero-rate parameter

The opposed scenario provides a useful stress test. At scale \(10^3\), the log1p rank correlation is approximately 0.320, 0.742, and 0.818 for zero-rate parameters of 0.70, 0.90, and 0.98, respectively. At the same settings, the top-20% IoU is approximately 0.141, 0.449, and 0.581.

Within this scenario and these simulation settings, ranking disagreement is strongest at the lower zero-rate parameter. This pattern should not be generalized to all scenarios without further evidence.

## 5. Null scenario and metric interpretation

In the null scenario, the true level CATE is constant at zero. Consequently, there is no true level-CATE ranking to recover. Rank correlation and top-20% overlap against this constant truth are therefore reported as undefined (`NaN`), rather than interpreted using arbitrary tie-breaking among equal scores.

This distinction is important: undefined ranking metrics in the null scenario are not evidence of poor or good ranking performance.

## 6. Interpretation and limitations

The oracle results isolate differences between estimands: the observed ranking changes arise from the transformations and their scales, not from model-fitting error.

The findings support three conclusions:

1. Positive scaling of the identity transformation preserves the level-CATE ranking.
2. Square-root ranking is invariant to positive scale but can differ from the level-CATE ranking.
3. Log1p and arcsinh can induce scale-dependent ranking changes, with magnitude depending on the data-generating scenario and zero-rate parameter.

These results are preliminary and descriptive. The reported comparisons use one random seed and a fixed sample size of 50,000. Multiple seeds and additional sensitivity analyses are needed to assess Monte Carlo variability and determine how stable the patterns are across simulation settings.

## 7. Reproducibility artifacts

The analysis produces the following outputs:

- `results/oracle_ranking_table.csv` — main oracle ranking comparisons.
- `results/figure1_oracle_ranking.png` — main oracle ranking figure.
- `results/oracle_robustness_summary.csv` — consolidated robustness results across six scenarios, three zero-rate parameters, four transformations, and five selected scales.

The robustness summary contains 360 rows. The null-scenario ranking metrics are undefined for all its rows.

**Figure placement:** Insert `results/figure1_oracle_ranking.png` here after confirming that the plotted scenario and scale settings match the final manuscript text.