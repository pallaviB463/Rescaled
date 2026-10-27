# Hand-Off Note: Member Q to Member P (Week 4 Hand-Off)

**From:** Member Q (Engine and Real Data)  
**To:** Member P (Truth and Models)  
**Date:** October 2026  

---

### 1. Status of Deliverables from Member Q

As agreed in Part 5 of the Playbook, Member Q has finalized the experiment engine and real-data pipelines:

1. **Uniform Interface for Meta-Learners (`LEARNERS`):**
   - All learners in [`src/metalearners.py`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/src/metalearners.py) accept `e` (propensity score) uniformly:
     ```python
     LEARNERS[name](ModelClass, X, W, y, X_test, e=0.5) -> tau_hat
     ```
   - `e` is used for propensity weighting in X-learner ($(1-e)t_1 + e t_0$) and DR-learner pseudo-outcome inverse weighting.
   - For S- and T-learners, `e` is accepted in the signature and safely ignored.
   - All 16 unit tests pass with `backend == "lightgbm"`.

2. **Criteo Semi-Synthetic Covariate Sample (Task P12 Ready):**
   - We downloaded the official Criteo Uplift v2.1 benchmark (13,979,592 rows, 85.00% treated) and extracted a clean, 200,000-row stratified subsample preserved at:
     ```
     data/criteo_sample_200k.parquet
     ```
   - Contains features `f0, ..., f11` and binary treatment assignment $W$ ($e = 0.8500$).
   - Standardized features and ready for Member P's `simulate_from_X(X, W, scenario)` function.

3. **Empirical Benchmarks on Real Marketing Field Experiment (Task Q6 / Table 3):**
   - Both Men's E-Mail vs Control ($N=42{,}613$) and Women's E-Mail vs Control ($N=42{,}693$) have finished out-of-fold cross-validation.
   - Results are saved in [`results/hillstrom_mens.csv`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/results/hillstrom_mens.csv) and [`results/hillstrom_womens.csv`](file:///c:/Users/palla/Downloads/Member_Q_starter_kit/q_kit/results/hillstrom_womens.csv).
   - Key finding for Section 7: `log1p` drops to $\rho \approx 0.72\text{--}0.77$ and replaces $\sim 50\%$ of the top-20% customer list when units change from dollars to hundreds of dollars, while Tweedie and Hurdle maintain $\rho = 1.0000$ and $100\%$ overlap.

---

### 2. What Member Q Needs from Member P

To proceed with Hand-off 1 & 2:
1. **P's Tested & Frozen Modules (End of Week 3):**
   - Tested versions of `dgp.py`, `metrics.py`, `transforms.py`, and `models.py` to replace Q's stand-in copies.
2. **`simulate_from_X(X, W, scenario)` (Week 4 Hand-off 2):**
   - Signature:
     ```python
     simulate_from_X(X, W, scenario) -> X, W, Y, truth
     ```
     where `truth` is a dictionary with keys `('tau_level', 'p0', 'p1', 'mu0', 'mu1', 'sigma')`.
   - Once provided, Q will plug it directly into `run_sim.py` to produce the Criteo semi-synthetic results table.
3. **Oracle Scale Sensitivity Figure (Fig 1 & Oracle Table):**
   - True rank correlation and top-20% overlap vs. $\log_{10}(a)$ across all 6 scenarios.
