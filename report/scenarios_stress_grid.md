# Member P: Simulation Scenarios and Stress-Grid Protocol

## 1. Purpose

This document specifies the simulation scenarios and the outcome-transformation and scale settings used in the Neurolytics project.

## 2. Treatment-effect scenarios

The data-generating process defines six scenarios:

| Scenario | Purchase log-odds shift | Log-spend shift |
|---|---|---|
| null | 0 | 0 |
| homogeneous | 0.4 | 0.15 |
| extensive | 0.3 + 0.8 Z3 | 0 |
| intensive | 0 | 0.2 + 0.4 Z4 |
| aligned | 0.3 + 0.8 Z3 | 0.2 + 0.4 Z3 |
| opposed | 0.3 + 0.8 Z3 | 0.2 - 0.4 Z3 |

Here, Z3 and Z4 denote the fourth and fifth covariates, respectively, using zero-based indexing in the implementation.

The extensive scenario varies the purchase response, while the intensive scenario varies the spending response conditional on purchase. The aligned and opposed scenarios combine purchase and spending effects.

## 3. Outcome transformations

The implementation supports the following transformations:

- **identity:** \(m(y)=y\)
- **log1p:** \(m(y)=\log(1+y)\)
- **arcsinh:** \(m(y)=\operatorname{arcsinh}(y)\)
- **sqrt:** \(m(y)=\sqrt{y}\)
- **rank:** normalized empirical ranks
- **level:** the untransformed outcome scale used by the simulation pipeline

The rank transformation is invariant to positive rescaling, apart from the handling of ties.

### Mathematical properties and transformation controls

The square-root transformation satisfies the scale-equivariance identity:

\[
\sqrt{ay}=\sqrt{a}\sqrt{y}, \qquad a>0,\ y\geq 0.
\]

This identity was numerically verified for all seven configured scale factors.

The identity, log1p, arcsinh, and square-root transformations all satisfy \(m(0)=0\).

The normalized empirical-rank transformation is used as a negative control for scale sensitivity. Its output at zero is not necessarily zero because normalized ranks depend on the empirical sample and rank convention. With average ranks divided by sample size, the zero observation in a sample of four distinct values has normalized rank \(1/4\).

Positive rescaling preserves empirical ranks, apart from numerical or tie-handling considerations.

## 4. Stress-grid configuration

The stress grid evaluates the following control-group zero rates:

- 70%
- 90%
- 98%

The outcome scale factors are:

\[
0.001,\ 0.01,\ 0.1,\ 1,\ 10,\ 100,\ 1000
\]

The positive-spend component follows a lognormal distribution with standard deviation:

\[
\sigma=0.8
\]

The grid CSVs are:

- `results/grid_zero_70.csv`
- `results/grid_zero_90.csv`
- `results/grid_zero_98.csv`

Each CSV contains 180 rows. The three files therefore contain 540 rows in total.

### Configuration coverage

Each zero-rate CSV contains:

- Six scenarios: `null`, `homogeneous`, `extensive`, `intensive`, `aligned`, and `opposed`.
- Four main transformations: `identity`, `log1p`, `arcsinh`, and `sqrt`.
- Seven scale factors for each main transformation.
- One `rank` control per scenario.
- One `level` / T-learner control per scenario.

The observed learner distribution per zero-rate setting is:

| Learner | Number of rows |
|---|---:|
| TransformedOutcome | 174 |
| T | 6 |
| **Total** | **180** |

This corresponds to 29 configurations per scenario for the TransformedOutcome learner (28 main transformation-scale combinations plus one rank control) and one level/T-learner control per scenario.

## 5. Reproducibility and limitations

The CSV files record the scenario, learner, transformation, scale, seed, sample size, zero rate, noise parameter, metrics, status, and error field.

The current grid audit establishes successful execution and configuration uniqueness. It does not, by itself, establish statistical significance, scale equivariance for every model, or completion of the separate semi-synthetic benchmark.

## 6. Related implementation files

- `src/dgp.py`
- `src/transforms.py`
- `src/run_sim.py`
- `src/oracle_analysis.py`

## 7. Validation Status

The scenario and stress-grid experiment was executed successfully for all three zero-rate settings.

### 7.1 Scenario coverage

All six scenarios are present in each CSV:

- `null`
- `homogeneous`
- `extensive`
- `intensive`
- `aligned`
- `opposed`

Each scenario has exactly 30 configurations per zero-rate setting, giving 180 rows per CSV file and 540 rows across all three files.

### 7.2 Stress-grid settings

The audit confirmed:

- Zero rates: `0.70`, `0.90`, `0.98`
- Outcome scale factors: `0.001`, `0.01`, `0.1`, `1`, `10`, `100`, `1000`
- Positive-spend lognormal standard deviation: `sigma = 0.8`
- Main transformations: `identity`, `log1p`, `arcsinh`, `sqrt`
- Controls: `rank` and `level`

### 7.3 Execution audit

| Result file | Rows | Zero rate | Execution status |
|---|---:|---:|---|
| `results/grid_zero_70.csv` | 180 | 0.70 | All `ok` |
| `results/grid_zero_90.csv` | 180 | 0.90 | All `ok` |
| `results/grid_zero_98.csv` | 180 | 0.98 | All `ok` |
| **Total** | **540** | — | **All successful** |

Additional checks confirmed:

- All six scenarios appear exactly 30 times in each CSV.
- No duplicate configurations were found using scenario, learner, transformation, scale, and seed.
- All rows use `sigma = 0.8`.
- Each CSV contains only its intended zero-rate setting.
- No execution errors were recorded.
- The observed learner counts are 174 TransformedOutcome rows and 6 T rows per CSV.

### 7.4 CSV parsing note

The literal scenario label `null` is a valid scenario name, not missing data. Pandas may interpret this string as a missing value when reading CSV files with its default settings.

To preserve the scenario label during audits, use:

```python
pd.read_csv(path, keep_default_na=False)
```

### 7.5 Validation conclusion

The generated stress-grid results pass the structural and execution checks performed.

The six scenarios, three zero-rate settings, transformation controls, scale factors, learner counts, and result-file coverage are present as specified. All 540 rows have successful execution status, and no duplicate configurations were detected.

These checks establish structural and execution validity; they do not replace the separate statistical, equivariance, or semi-synthetic validation requirements.