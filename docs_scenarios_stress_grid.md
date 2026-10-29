# Member P: Simulation Scenarios and Stress-Grid Protocol

## 1. Purpose

This document specifies the simulation scenarios and the outcome-transformation
and scale settings used in the Neurolytics project.

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

Here, Z3 and Z4 denote the fourth and fifth covariates, respectively,
using zero-based indexing in the implementation.

The extensive scenario varies the purchase response, while the intensive
scenario varies the spending response conditional on purchase. The aligned
and opposed scenarios combine purchase and spending effects.

## 3. Outcome transformations

The implementation supports the following transformations:

- identity: m(y) = y
- log1p: m(y) = log(1 + y)
- arcsinh: m(y) = arcsinh(y)
- sqrt: m(y) = sqrt(y)
- rank: normalized empirical ranks
- level: the untransformed outcome scale used by the simulation pipeline

The rank transformation is invariant to positive rescaling, apart from
the handling of ties.

## 4. Stress-grid configuration

The stress grid evaluates the following control-group zero rates:

- 70%
- 90%
- 98%

The outcome scale factors are:

0.001, 0.01, 0.1, 1, 10, 100, 1000.

The grid CSVs currently available are:

- results/grid_zero_70.csv
- results/grid_zero_90.csv
- results/grid_zero_98.csv

Each CSV contains 180 rows. The three files therefore contain 540 rows
in total. The latest audit found status "ok" for every row, no recorded
errors, and no duplicate configuration rows.

The grid contains the six scenarios and six transformation labels.
Its observed learner counts are 174 TransformedOutcome rows and 6 T rows
per zero-rate setting. This learner distribution must be checked against
the final experimental specification before declaring coverage complete.

## 5. Reproducibility and limitations

The CSV files record the scenario, learner, transformation, scale, seed,
sample size, zero rate, noise parameter, metrics, status, and error field.

The current grid audit establishes successful execution and configuration
uniqueness. It does not, by itself, establish statistical significance,
scale equivariance for every model, or completion of the separate
semi-synthetic benchmark.

## 6. Related implementation files

- src/dgp.py
- src/transforms.py
- src/run_sim.py
- src/oracle_analysis.py

## 7. Validation status

The complete project test suite most recently reported 84 passing tests.
The model and transformation test files subsequently reported 28 passing
tests together.

The stress-grid audit reported 540 successful rows across the three
zero-rate settings. The learner distribution and exact required grid
configuration remain subject to final specification review.
