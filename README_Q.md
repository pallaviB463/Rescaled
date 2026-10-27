# Member Q Workspace: Neurolytics ADA Project

Code, pipelines, benchmarks, and manuscript artifacts for **Member Q (Engine and Real Data)** on the *Scale-Dependence of CATE and Uplift Rankings for Zero-Heavy Outcomes* project.

---

## 1. Environment & Setup

```powershell
pip install numpy pandas scipy scikit-learn lightgbm statsmodels matplotlib joblib pyarrow pyyaml pytest scikit-uplift
python run_tests.py     # Runs all 16 tests with LightGBM backend (0 failures)
```

---

## 2. Task Status Overview (Playbook Section 8)

| Task ID | Description | Status | Output Files / Artifacts |
|---|---|---|---|
| **Q1** | Learner cards & literature notes | **DONE** | [`notes/learner_cards.md`](notes/learner_cards.md) (Table 2 for paper) |
| **Q2** | Data access (Hillstrom & Criteo) | **DONE** | [`data/Hillstrom.csv`](data/Hillstrom.csv), [`data/criteo_sample_200k.parquet`](data/criteo_sample_200k.parquet), [`notes/data_note.md`](notes/data_note.md) |
| **Q3** | Meta-learners (S, T, X, DR) + tests | **DONE** | [`src/metalearners.py`](src/metalearners.py), [`tests/test_metalearners.py`](tests/test_metalearners.py) |
| **Q4** | Grid runner with common random numbers | **DONE** | [`src/run_sim.py`](src/run_sim.py), [`config/grid.yaml`](config/grid.yaml) |
| **Q5** | Timing run & quality gates check | **DONE** | [`config/grid_time.yaml`](config/grid_time.yaml), [`src/summarize.py`](src/summarize.py) (2.6 min/rep, grid kept intact) |
| **Q6** | Hillstrom real-data pipelines | **DONE** | [`results/hillstrom_mens.csv`](results/hillstrom_mens.csv), [`results/hillstrom_womens.csv`](results/hillstrom_womens.csv) |
| **Q7** | Manuscript skeleton & notation | **DONE** | [`report/main.tex`](report/main.tex), [`report/references.bib`](report/references.bib) |
| **Q8** | Simulation production runs (Stage 1) | **IN PROGRESS**| `python src/run_sim.py config/grid.yaml results/main --jobs 4 --tier must` |
| **Q9** | Analysis module (bootstrap CIs & Figs 2, 4)| **DONE** | [`src/analysis.py`](src/analysis.py), `figures/` |
| **Q10**| Real-data Table 3 | **DONE** | [`src/summarize_real.py`](src/summarize_real.py) |
| **Q11**| Reproducibility script | **DONE** | [`src/rebuild_figures.py`](src/rebuild_figures.py) |
| **Q12**| Writing & discussion | Scheduled | In progress alongside manuscript draft |

---

## 3. Quickstart & Reproducibility Commands

### A. Run Unit Tests
```powershell
python run_tests.py
```

### B. Reproduce Timing Run & Quality Gates (Task Q5)
```powershell
python src/run_sim.py config/grid_time.yaml results/time --jobs 1
python src/summarize.py results/time
```

### C. Run Real-Data Hillstrom Benchmarks (Task Q6 & Q10)
```powershell
# Primary (Men's E-Mail vs Control):
python src/run_real.py data/Hillstrom.csv mens results/hillstrom_mens.csv

# Replicate (Women's E-Mail vs Control):
python src/run_real.py data/Hillstrom.csv womens results/hillstrom_womens.csv

# Format Table 3:
python src/summarize_real.py results/hillstrom_mens.csv results/hillstrom_womens.csv
```

### D. Rebuild All Figures and Analysis (Task Q9 & Q11)
```powershell
python src/rebuild_figures.py
```

---

## 4. Hand-Off Interfaces
- **From Member P (Week 3/4):** Replacement of stand-in modules (`dgp.py`, `metrics.py`, `transforms.py`, `models.py`) with tested versions; `simulate_from_X` for Criteo semi-synthetic benchmark.
- **To Member P:** All meta-learners in [`src/metalearners.py`](src/metalearners.py) accept parameter `e` (propensity) uniformly for plug-and-play execution.
