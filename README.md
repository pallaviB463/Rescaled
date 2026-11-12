# Rescaled

## Overview

**Rescaled** is a research project investigating causal uplift estimation across different outcome scales. It studies how transformations of the outcome variable affect treatment-effect estimation, model behaviour, and the scale equivariance of uplift estimators.

The project implements and evaluates multiple meta-learners, outcome transformations, simulation scenarios, and evaluation metrics. It also includes oracle-based analysis and a Criteo semi-synthetic experiment to study estimator performance under different conditions.

The primary focus is to understand how modelling choices and outcome scales influence the reliability and behaviour of estimated treatment effects.

---

## Project Objectives

The main objectives of this project are:

- Study causal treatment-effect estimation under different outcome scales.
- Investigate the impact of outcome transformations on uplift estimation.
- Implement and evaluate different meta-learning approaches.
- Develop simulation scenarios to test estimator behaviour under controlled conditions.
- Examine scale equivariance and the effects of back-transforming predictions.
- Compare model performance using appropriate evaluation metrics.
- Conduct oracle-based analysis to understand estimator performance.
- Evaluate the proposed approaches using a Criteo semi-synthetic experiment.
- Perform sanity checks and robustness analyses to validate the implementation.

---

## Methodology

The project follows a simulation-based and experimental methodology.

### 1. Data Generation

Synthetic datasets are generated using configurable data-generating processes (DGPs). These allow controlled experimentation with different treatment assignments, outcome distributions, and treatment-effect structures.

### 2. Outcome Transformations

Different outcome transformations are implemented to investigate how changing the outcome scale affects model fitting and treatment-effect estimation.

### 3. Meta-Learners

Multiple meta-learning approaches are implemented to estimate treatment effects, including:

- S-Learner
- T-Learner
- X-Learner
- DR-Learner
- Transformed Outcome Learner

### 4. Evaluation

The estimators are evaluated using metrics designed to measure the quality of treatment-effect estimates and compare performance across different experimental settings.

### 5. Simulation and Oracle Analysis

Simulation experiments are used to compare estimators under controlled conditions. Oracle analysis provides an additional reference for evaluating estimator behaviour.

### 6. Criteo Semi-Synthetic Experiment

A Criteo semi-synthetic workflow is included to evaluate the methods using real covariates with synthetically generated treatment and outcome variables.

### 7. Back-Transformation Analysis

The project investigates back-transformation strategies and their effect on treatment-effect estimation, including comparisons with alternative approaches.

---

## Repository Structure

```text
Rescaled/
│
├── config/
│   └── Configuration files for experiments
│
├── data/
│   └── Datasets used in experiments
│
├── notebooks/
│   └── Jupyter notebooks for exploration and analysis
│
├── report/
│   ├── criteo_semisynthetic.md
│   ├── backtransform.md
│   └── scenarios_stress_grid.md
│
├── results/
│   ├── simulations/
│   │   └── Simulation outputs and summary metrics
│   │
│   ├── oracle_analysis/
│   │   └── Oracle rankings, policy values, robustness results and figures
│   │
│   ├── tuning/
│   │   └── Hyperparameter tuning experiments and comparisons
│   │
│   ├── backtransform/
│   │   └── Back-transformation baselines and ablation results
│   │
│   ├── null_audit/
│   │   └── Null-scenario audits and prediction magnitude analysis
│   │
│   └── model_comparisons/
│       └── Model and six-scenario comparison results
│
├── src/
│   ├── dgp.py
│   ├── transforms.py
│   ├── models.py
│   ├── metalearners.py
│   ├── metrics.py
│   ├── tuning.py
│   ├── run_sim.py
│   ├── oracle_analysis.py
│   └── backtransform.py
│
├── tests/
│   └── Automated tests for project modules
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** Intermediate smoke tests, pilot experiments, duplicate files, backups, and other temporary outputs are maintained locally under `results/_local_only/` and are excluded from version control.

---

## Technologies Used

- **Python 3.11**
- NumPy
- Pandas
- SciPy
- Scikit-learn
- LightGBM
- Statsmodels
- Matplotlib
- Joblib
- Pytest

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pallaviB463/Rescaled.git
```

### 2. Navigate to the Project Directory

```bash
cd Rescaled
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

Run the simulation script from the project root:

```bash
python -m src.run_sim
```

The exact experiment configuration and available options depend on the implementation in the corresponding source modules.

---

## Running Tests

The project uses **pytest** for automated testing.

To run the complete test suite:

```bash
python -m pytest tests/ -v
```

To run tests for an individual module:

```bash
python -m pytest tests/test_dgp.py -v
```

```bash
python -m pytest tests/test_transforms.py -v
```

```bash
python -m pytest tests/test_models.py -v
```

```bash
python -m pytest tests/test_metrics.py -v
```

```bash
python -m pytest tests/test_metalearners.py -v
```

---

## Experimental Results

The `results/` directory contains the selected experimental outputs generated during the project.

| Directory | Contents |
|---|---|
| `simulations/` | Simulation outputs and zero-rate grid experiments |
| `oracle_analysis/` | Oracle policy values, ranking results, robustness analysis, and figures |
| `tuning/` | Causal tuning comparisons, validation results, tuning histories, and summary analysis |
| `backtransform/` | Back-transformation baseline and ablation results |
| `null_audit/` | Null-scenario audit results and prediction magnitude analysis |
| `model_comparisons/` | Comparisons across outcome models, learners, and simulation scenarios |

The tuning directory also contains `summarize_causal_comparison.py`, which combines the L2, Tweedie, and Hurdle causal tuning results and calculates mean and standard deviation across seeds.

The committed results focus on selected experimental outputs rather than temporary smoke tests, pilot runs, or intermediate files.

---

## Documentation

The `report/` directory contains documentation describing important experimental components and research contributions.

| Document | Description |
|---|---|
| `criteo_semisynthetic.md` | Documentation of the Criteo semi-synthetic experiment and workflow |
| `backtransform.md` | Documentation of back-transformation methods and related analysis |
| `scenarios_stress_grid.md` | Documentation of simulation scenarios and the stress-test grid |

---

## Testing and Validation

Automated tests are included to validate the main project components, including:

- Data-generating processes
- Outcome transformations
- Outcome models
- Meta-learners
- Evaluation metrics

The test suite is used to check implementation correctness and help identify regressions when changes are made.

---

## Research Scope

The research focuses on the relationship between outcome scales, transformations, and causal uplift estimation.

Key areas of investigation include:

1. Estimator behaviour under different outcome distributions.
2. The effect of outcome transformations on treatment-effect estimation.
3. Scale equivariance of uplift estimators.
4. Performance comparisons across different meta-learners.
5. Oracle-based evaluation and analysis.
6. Back-transformation strategies and their limitations.
7. Semi-synthetic evaluation using real-world covariates.

---

## Project Status

The core implementation and research activities assigned to **Member P (P1–P15)** have been completed.

The repository contains the corresponding implementation, tests, simulation workflows, experimental results, analysis, and documentation.

Further project contributions and extensions can be developed independently without modifying or removing the completed work.

---

## License

This repository is intended for academic and research purposes.