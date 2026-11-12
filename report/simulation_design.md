# Simulation Design: Outcome-Scale Sensitivity

## 1. Objective

Evaluate how outcome transformations and positive outcome rescaling affect
treatment-effect ranking and policy evaluation under different data-generating
scenarios and degrees of zero inflation.

## 2. Data-generating scenarios

The simulation includes six scenarios defined in `src/dgp.py`:

| Scenario | Treatment-effect structure |
|---|---|
| null | No treatment effect on purchase probability or positive spend |
| homogeneous | Constant shifts in purchase log-odds and log-spend |
| extensive | Heterogeneous purchase-probability effect; no log-spend shift |
| intensive | No purchase-probability shift; heterogeneous log-spend effect |
| aligned | Purchase and log-spend effects vary with the same covariate |
| opposed | Purchase and log-spend effects vary in opposing directions |

## 3. Zero-inflation settings

The control-group zero-rate parameter is evaluated at:

- 0.70
- 0.90
- 0.98

The DGP calibrates the control-group purchase probability to `1 - zero_rate`.
The realized treatment-group zero rate can differ because treatment changes
purchase probability in non-null scenarios.

## 4. Outcome transformations and scales

The transformation families are:

- `identity`
- `log1p`
- `arcsinh`
- `sqrt`
- `rank`

The positive scale factors are:

`0.001, 0.01, 0.1, 1, 10, 100, 1000`

The four non-rank transformations are evaluated at all seven scales.
The rank transformation is evaluated once at scale 1 because ranks are
invariant to positive rescaling.

## 5. Initial experiment configuration

| Parameter | Value |
|---|---|
| Sample size | 5,000 |
| Random seed | 0 |
| Outcome noise parameter (`sigma`) | 0.8 |
| Treatment assignment parameter (`e`) | 0.5 |
| Meta-learner | T-learner |
| Outcome model | L2 |
| Scenarios | All six |
| Control zero-rate settings | 0.70, 0.90, 0.98 |

## 6. Recorded metrics

The simulation runner records:

- `rank_corr`: correlation between estimated and true treatment-effect rankings
- `topk_overlap`: top-k ranking overlap, implemented using intersection over union
- `policy_value`: value of the estimated treatment policy
- `regret`: difference between oracle and estimated policy value
- `rho_vs_a1`: ranking correlation against the same transformation at scale 1
- `status` and `error`: execution status and failure information

Ranking metrics are expected to be undefined (`NaN`) in the null scenario
when the ground-truth treatment effect has no ranking variation.

## 7. Initial grid results

The initial grid was run separately for control zero-rate settings 0.70,
0.90, and 0.98.

| Control zero rate | Rows | Successful | Failed |
|---|---:|---:|---:|
| 0.70 | 180 | 180 | 0 |
| 0.90 | 180 | 180 | 0 |
| 0.98 | 180 | 180 | 0 |
| Total | 540 | 540 | 0 |

Output files:

- `results/grid_zero_70.csv`
- `results/grid_zero_90.csv`
- `results/grid_zero_98.csv`

Summary files are saved beside each output file with the `_summary.csv`
suffix.

## 8. Limitations and next steps

This is an initial grid using one random seed and sample size 5,000.
It establishes that the configurations execute successfully; it does not
establish multi-seed robustness or statistical uncertainty.

Next steps include repeating the experiment across multiple seeds, examining
the resulting metric summaries, and deciding whether larger samples or
additional outcome models are needed for the final analysis.
