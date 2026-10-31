# Rescaled — Causal Uplift Modeling Across Outcome Scales

## Project Overview

Rescaled is a research project investigating how outcome transformations and measurement scales affect the estimation and evaluation of heterogeneous treatment effects.

The project studies whether uplift models produce consistent treatment-effect estimates and targeting decisions when outcomes are represented on different scales, particularly for nonnegative and zero-inflated outcomes.

The implementation includes data-generating processes, outcome transformations, causal estimators, model tuning, evaluation metrics, simulation experiments, and supporting analyses.

## Research Objective

The central research question is:

**How does the choice of outcome scale and transformation affect the estimation, ranking, and evaluation of heterogeneous treatment effects?**

The project examines:

- Treatment-effect estimands on the original and transformed outcome scales.
- The behavior of outcome transformations at different scales.
- The effect of model choice and tuning on uplift estimation.
- Scale equivariance and ranking consistency.
- The relationship between prediction accuracy and policy performance.

## Methodology

### 1. Estimands

The project distinguishes between the level-scale conditional average treatment effect:

\[
\tau_{\text{level}}(x)
=
E[Y(1)-Y(0)\mid X=x]
\]

and the transformed-scale treatment effect:

\[
\tau_{m,a}(x)
=
E[m(aY(1))-m(aY(0))\mid X=x]
\]

where:

- \(Y(1)\) and \(Y(0)\) represent potential outcomes under treatment and control.
- \(m\) is an outcome transformation.
- \(a\) is a scale parameter.
- \(X\) represents observed covariates.

These estimands need not produce identical treatment-effect rankings.

See `report/estimands.md` for the formal definitions and theoretical discussion.

### 2. Data-Generating Process

The simulation framework supports multiple treatment-effect scenarios:

| Scenario | Description |
|---|---|
| Null | No treatment effect |
| Homogeneous | Constant treatment effects |
| Extensive | Treatment effects on the probability of a positive outcome |
| Intensive | Treatment effects on the positive-outcome magnitude |
| Aligned | Extensive and intensive effects move together |
| Opposed | Extensive and intensive effects move in opposite directions |

The DGP models nonnegative outcomes with a zero-inflated structure.

Implementation: `src/dgp.py`

### 3. Outcome Transformations

The transformation module supports the transformations used in the scale-comparison experiments, including:

- Identity
- Logarithmic transformation using `log1p`
- `arcsinh`
- Scale-dependent transformation variants

Implementation: `src/transforms.py`

### 4. Outcome Models

The project includes several outcome-model families:

- L2 regression
- Poisson regression
- Tweedie regression
- Hurdle models
- Negative Binomial models

Implementation: `src/models.py`

### 5. Meta-Learners

The causal estimation framework includes:

- S-learner
- T-learner
- X-learner
- Doubly Robust (DR) learner
- Transformed Outcome learner

Implementation: `src/metalearners.py`

### 6. Hyperparameter Tuning

The tuning framework evaluates model configurations using separate training, validation, and test data.

The objective is to support fair comparisons across model configurations while keeping final evaluation separate from model selection.

Relevant implementation:

- `src/tuning.py`
- `src/run_tuning.py`
- `src/run_tuned_causal_sim.py`

Further details: `report/hyperparameter_tuning.md`

## Evaluation Framework

The project evaluates treatment-effect estimates using multiple metrics.

These include:

- Rank correlation
- Top-k overlap
- Policy value
- Policy regret
- Qini AUC
- Inverse-propensity-weighted policy value
- RMSE on the relevant outcome scale

Implementation: `src/metrics.py`

The evaluation distinguishes prediction performance from the quality of treatment-targeting decisions.

## Additional Analyses

The completed Member P implementation includes the following supporting analyses:

| Analysis | Purpose |
|---|---|
| Scale equivariance | Examine how rescaling outcomes affects model predictions and treatment-effect rankings |
| Oracle analysis | Compare estimated targeting performance with oracle benchmarks |
| Back-transform analysis | Evaluate back-transformed predictions and treatment-effect estimates |
| Back-transform ablation | Compare alternative back-transformation approaches |
| Sanity audit | Check important implementation and experimental assumptions |
| Criteo semi-synthetic experiment | Evaluate the simulation framework using real covariates and synthetic potential outcomes |

Relevant files include:

- `src/oracle_analysis.py`
- `src/run_backtransform.py`
- `src/run_backtransform_ablation.py`
- `src/prepare_criteo_semisynthetic.py`
- `src/validate_criteo_semisynthetic.py`
- `src/run_prediction_audit.py`

Supporting documentation and results are maintained in `report/` and the project documentation files.

## Criteo Semi-Synthetic Experiment

The project includes a semi-synthetic experiment using a sample of Criteo covariates.

The workflow uses real covariates while generating synthetic treatment assignments and potential outcomes through the project's simulation framework.

The experiment is intended to examine the behavior of the methods under a more realistic covariate distribution.

Important distinction: the generated outcomes are synthetic; this is not an evaluation using the original Criteo treatment and outcome labels.

Relevant files:

- `src/prepare_criteo_sample.py`
- `src/prepare_criteo_semisynthetic.py`
- `src/validate_criteo_semisynthetic.py`
- `docs_p12_criteo_semisynthetic.md`

## Project Structure

```text
Rescaled/
│
├── config/
│
├── data/
│
├── notebooks/
│
├── report/
│   ├── estimands.md
│   ├── hyperparameter_tuning.md
│   ├── literature_track_a.md
│   ├── manuscript.md
│   ├── oracle_results.md
│   ├── research_gap.md
│   ├── simulation_design.md
│   ├── backtransform_ablation.md
│   └── sanity_audit.md
│
├── src/
│   ├── backtransform.py
│   ├── dgp.py
│   ├── metalearners.py
│   ├── metrics.py
│   ├── models.py
│   ├── oracle_analysis.py
│   ├── transforms.py
│   ├── tuning.py
│   ├── run_sim.py
│   ├── run_tuning.py
│   ├── run_tuned_causal_sim.py
│   ├── run_backtransform.py
│   ├── run_backtransform_ablation.py
│   ├── run_prediction_audit.py
│   ├── prepare_criteo_sample.py
│   ├── prepare_criteo_semisynthetic.py
│   └── validate_criteo_semisynthetic.py
│
├── tests/
│   ├── test_dgp.py
│   ├── test_transforms.py
│   ├── test_models.py
│   ├── test_metalearners.py
│   ├── test_metrics.py
│   ├── test_tuning.py
│   ├── test_run_sim.py
│   ├── test_oracle_analysis.py
│   ├── test_backtransform.py
│   ├── test_backtransform_ablation.py
│   ├── test_criteo_ground_truth.py
│   └── test_criteo_semisynthetic.py
│
├── results/
│
├── docs_p12_criteo_semisynthetic.md
├── docs_p13_backtransform.md
├── docs_scenarios_stress_grid.md
├── requirements.txt
└── README.md
```

Some generated outputs and supporting files may be added as experiments progress.

## Installation and Setup

### Requirements

- Python 3.11
- pip
- Git

### Setup

Clone the repository:

```bash
git clone https://github.com/pallaviB463/Rescaled.git
cd Rescaled
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest tests/ -v
```

Run an individual test module:

```powershell
python -m pytest tests/test_dgp.py -v
```

The test suite covers the DGP, transformations, outcome models, meta-learners, metrics, tuning, simulation runner, oracle analysis, back-transformation, and Criteo semi-synthetic workflow.

## Reproducibility

For reproducible experiments:

1. Run commands from the repository root.
2. Preserve the random seeds used by simulation scripts.
3. Keep training, validation, and test data separate.
4. Record model configurations and experimental settings.
5. Avoid manually modifying generated result files.
6. Run the relevant tests after changing implementation code.
7. Document deviations from the experimental protocol.

## Current Scope and Limitations

The completed Member P contribution establishes the core simulation, modeling, transformation, evaluation, and analysis framework.

The following limitations should be considered when interpreting results:

- The semi-synthetic Criteo experiment uses real covariates but synthetic outcomes and treatment assignments.
- The `simulate_from_X` calibration is based on a standard-normal reference population; a requested zero rate is not guaranteed to equal the realized rate for every supplied covariate sample.
- The current DR-learner simulation implementation does not include cross-fitting.
- Negative Binomial model support does not by itself constitute a dedicated integer-count DGP comparison.
- The Hurdle Gamma-versus-lognormal comparison remains outside the completed implementation.

These distinctions should be preserved when reporting experimental findings.

## Contribution Status

Member P's assigned work, P1–P15, has been implemented and committed to the project branch.

The contribution covers:

- Estimand definitions and theoretical documentation.
- Literature review.
- Simulation scenarios and transformation documentation.
- DGP and evaluation metrics.
- Outcome models and meta-learners.
- Hyperparameter tuning.
- Scale-equivariance analysis.
- Simulation and semi-synthetic experiment support.
- Oracle analysis.
- Back-transformation experiments.
- Sanity checks and manuscript contributions.

Member Q's additional work is outside the scope of this branch's completed Member P contribution.

## Research Documentation

Refer to the `report/` directory and the supporting Markdown documentation files for detailed methodology, theoretical definitions, experiment descriptions, analysis, and manuscript material.