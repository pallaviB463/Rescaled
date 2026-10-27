# Member Q handoff brief: Neurolytics ADA project

> **What this is.** A context pack so a fresh Claude session (a new chat, a Project, or Claude Code) can pick up Member Q's work without re-explaining anything. **Attach with it:** `Member_Q_starter_kit.zip` (the code), this PDF, and ideally `Neurolytics_Two_Person_Playbook.pdf` and the original one-pager. Then paste the first message in section 10. Prepared 27 Oct 2026.


## 1. The project in one page

**Title:** Does the Way We Transform Outcomes Change Who We Target? Scale-dependence of CATE and uplift rankings for zero-heavy outcomes, with count-aware learners. Course project for Advanced Data Analytics (team Neurolytics).

**Idea.** Spend and purchase outcomes are mostly zero, so analysts apply log(1+y) or arcsinh before fitting uplift models. Chen and Roth (2024) showed that for the average effect such a transform gives a number that depends on the units of y (dollars vs cents) when treatment changes who buys at all. The project asks whether the same happens for individual-level effects: does the unit or transformation change which customers rank as most persuadable, and how much targeting payoff is lost? Count-aware models (Poisson, Tweedie, hurdle) are the proposed remedy.

**Estimands (never mix them).** `tau_level` = extra dollars or purchases, what a business wants; every targeting-quality metric (rank correlation, policy value, regret) is measured against it. `tau_m,a` = effect on the transformed outcome m(a*y), what a transformed-outcome analyst estimates; use it only to check how well a learner recovers its own target.

- **H1.** Log-like rankings change with the unit scale a when extensive and intensive effects are driven by different covariates.
- **H2.** As a goes to 0 the ranking approaches the level (dollar) ranking; as a goes to infinity it approaches the ranking by effect on purchase probability.
- **H3.** Log-link level models (Poisson, Tweedie, hurdle) are unit-equivariant, so their rankings do not change with units.
- **H4.** Count-aware learners give higher policy value than transformed-outcome learners when the two margins disagree.
- **Negative controls:** sqrt and rank transforms must not change rankings with units. If they do, there is a bug.

**Data.** (1) Own simulator with exact ground truth (six scenarios: null, homogeneous, extensive, intensive, aligned, opposed; zero rates 70/90/98%). (2) Hillstrom MineThatData e-mail test, 64,000 customers, three arms. (3) Criteo Uplift v2.1 (~14M rows), used as a semi-synthetic benchmark: real covariates, simulated outcomes with known effects.


## 2. Team situation and my role

The original plan had four members, but only two are doing the work. The work is split into **Member P (Truth and Models)**: dgp.py, metrics.py, transforms.py, models.py, Fig 1, Criteo semi-synthetic, theory/simulation/methods/related-work writing; and **Member Q (Engine and Real data), which is me**: metalearners.py, run_sim.py, run_real.py, data access, production runs, analysis and Figs 2-4, Table 3, manuscript skeleton, and the abstract, introduction, experiments, real data, discussion, decision rule and Appendix B. P's four modules currently exist in my repo only as **stand-in copies** (with bug fixes); P's tested versions replace them at hand-off.


## 3. Interface contract (frozen; do not change signatures)

```
simulate(n, scenario, zero_rate, d=10, sigma=0.8, e=0.5, seed=0) -> X, W, Y, truth
    truth = dict(p0, p1, mu0, mu1, sigma, tau_level)
true_transformed_cate(truth, m, a) -> array          # needs m(0)=0
apply_transform(name, y, a) -> array                  # identity, log1p, arcsinh, sqrt, rank
Model class: fit(X, y) -> self ; predict(X) -> array on the scale of y
    L2, Poisson, Tweedie, Hurdle (NegBin only if the integer-count generator is built)
LEARNERS[name](M, X, W, y, Xte, e=0.5) -> tau_hat     # name in S, T, X, DR ; e is known treatment prob.
metrics: rank_corr, topk_overlap, policy_value, regret, qini_auc, ipw_policy_value
```

**Results row** (one per run, saved per replication as parquet): scenario, zero_rate, n, seed, family, transform, scale, learner, backend, mean_tau_hat, rho_level, value, regret, rho_vs_a1, top20_overlap_vs_a1. `rho_vs_a1` is the rank correlation with the same pipeline at scale a = 1. Policy value and regret are top-20% and level-scale only.


## 4. Repository map (`q_kit/`)

| Path | Owner | What it does |
|---|---|---|
| src/metalearners.py | Q | S, T, X, DR learners; all take e. X second stage and DR final stage are plain L2 (pseudo-outcomes can be negative). DR is cross-fitted (5 folds). |
| src/run_sim.py | Q | Grid-driven runner: YAML grid, joblib parallel, one file per replication (resumable), common random numbers, aggregate(). CLI: config out_dir --jobs --tier. |
| src/run_real.py | Q | Hillstrom pipeline: Men's vs control (or Women's), e from data, repeated K-fold out-of-fold scores, transforms applied inside training folds only, IPW policy value at k=10/20/30/50%, Qini. |
| src/dgp.py, metrics.py, transforms.py, models.py | P (stand-ins) | Simulator, metrics, transforms, LightGBM models (sklearn fallback if LightGBM missing; backend column records which). |
| config/grid.yaml, grid_pilot.yaml | Q | Main grid (tiers must/should) and the 10-replication pilot. |
| tests/, run_tests.py | both | 16 tests; run_tests.py works without pytest. |
| notes/learner_cards.md | Q | DRAFT learner table for the paper (Table 2); formulas written from memory, need verifying against the papers. |


## 5. Status (what is verified, and by whom)

| Item | Status |
|---|---|
| Setup, repo, contract | Done. Windows, Python 3.13, LightGBM installed. |
| All 16 tests | PASS on my machine with backend=lightgbm (including DR no-leakage, null-scenario handling, resume). |
| Pilot run | `python src/run_sim.py config/grid_pilot.yaml results/pilot --jobs 4`: 10 replications finished in 2.4 min on 4 workers (n=5,000, 3 scales, T and DR). Results exist but the controls table has not yet been checked. |
| Compute estimate (extrapolated, not measured) | About 1 core-minute per replication at pilot size; about 6 core-minutes at n=20,000 with 7 scales. Must tier about 240 replications = roughly 24 core-hours, about 6 hours on 4 cores. Needs confirming with a single timing run (grid_time.yaml, section 8). |
| Hillstrom pipeline | Code done; tested only on a synthetic file with Hillstrom's column names. Not yet run on the real CSV. |
| Learner-equivariance at a=0.001 and a=1000 with LightGBM | NOT yet verified (tiny or huge targets can interact with LightGBM thresholds). |
| P's real modules | Not yet swapped in (confirm with P). |


## 6. Decisions already made (do not re-litigate)

- **Hillstrom:** Men's e-mail vs control is primary; Women's vs control is the replicate. Within two arms e is about 0.5; compute it from the data. No ground truth, so evaluate with repeated 5-fold CV, rank stability across scales, IPW policy value, Qini, paired bootstrap intervals.
- **Criteo:** semi-synthetic only (via P's simulate_from_X), subsample 200k to 1M rows stratified by treatment; state whether real treatment (with its share as e) or fresh random W with e=0.5 is used.
- **Learners:** T and DR are Must; S and X are Should. Count-aware models apply only in the outcome (nuisance) models.
- **Grid:** n=20,000; 90% zeros for all six scenarios (50 reps; 20 for null and homogeneous); 70% and 98% only for opposed and aligned (Should). Scales 0.001...1000 (7) for transformed pipelines; 0.01, 1, 100 for count-aware. Always include a=1.
- **Quality gates:** sqrt, rank and level-model rows must have rho_vs_a1 near 1 before any batch is trusted; null scenario shows nothing detectable (rho_level is NaN there because the truth is constant; use mean tau_hat and value instead); no regret below 0 or correlation above 1.
- **Never report rows with backend == sklearn-fallback.**
- **Novelty wording:** claim a systematic benchmark across meta-learner families under unit changes. Do not claim nobody has used hurdle or zero-inflated models in uplift; two-stage and zero-inflated uplift models already exist. The database sweep is still to be done by P.


## 7. Gotchas already hit or known

- **YAML:** a bare `null` is parsed as None. Scenario `"null"` must be quoted in configs (runner now also tolerates it).
- **Windows:** the joblib/loky `wmic` traceback about physical cores is a harmless warning. No `make` by default, so use a Python script for the figure-rebuild target instead of a Makefile.
- **Calibration** targets the control arm's zero rate (in the null scenario treated equals control).
- **Spearman with a constant vector is NaN** (null-scenario truth); aggregate with nanmean.
- **X-learner** combines as (1-e)*t1 + e*t0. **DR** pseudo-outcome: mu1 - mu0 + W(Y-mu1)/e - (1-W)(Y-mu0)/(1-e).
- **rank transform** is applied on training rows only in run_real.py to avoid leakage.
- Single-run estimated rankings are noisy (rho with truth about 0.5 to 0.6 at n=20,000 in smoke tests). Use 50+ replications, paired comparisons, bootstrap CIs.


## 8. Remaining work for Member Q (with acceptance tests)

| ID | Task | Done when |
|---|---|---|
| Q5 | Timing run: save `config/grid_time.yaml` (n=20000, 7 scales, count_scales [0.01,1,100], learners [T,DR], one cell: scenarios ["opposed"], zero_rates [0.9], reps 1) and run it with --jobs 1. Then check controls with a small summarize script (mean rho_vs_a1, rho_level, regret by family/transform/scale/learner for opposed and null). | Minutes per replication recorded; sqrt, rank, level models have rho_vs_a1 near 1 at all scales including 0.001 and 1000; log1p/arcsinh drop below 1 in opposed. Cut the grid if total wall time would exceed about 12 h. |
| Q1 | Verify and finish learner cards (Kunzel et al.; Kennedy; Nie and Wager; Curth and van der Schaar). | Table 2 text final, one line per learner on why it matters for zero-heavy Y. |
| Q2 | Download Hillstrom and Criteo (scikit-uplift or source pages). Data note: row counts, treatment shares, buyer rate. | Hillstrom 64,000 rows, three arms, buyer rate about 1% (verify); Criteo about 14M rows, heavily treated (verify). |
| Q6 | Run `python src/run_real.py data/Hillstrom.csv mens results/hillstrom_mens.csv`, then womens. | Two result tables; sanity-check column names and arm labels ("Mens E-Mail", "Womens E-Mail", "No E-Mail"). |
| Hand-off | Replace stand-in dgp/metrics/transforms/models with P's tested versions; rerun all tests. Ask P for simulate_from_X. | All tests still pass; backend=lightgbm. |
| Q7 | Manuscript skeleton (LaTeX or Word): fixed notation tau_level, tau_m,a, a, k; placeholder for every figure/table with owner. | Skeleton compiles. |
| Q8 | Production runs. Stage 1: `python src/run_sim.py config/grid.yaml results/main --jobs N --tier must`. Stage 2: `--tier should`. Gate every batch. | Parquet for every cell; gate report (controls near 1, null clean, no impossible values). |
| Q9 | Analysis: aggregate, bootstrap CIs over seeds, paired differences. Fig 2 rank stability vs log10(a) per family/learner; Fig 3 policy value vs top-k; Fig 4 regret heatmap per scenario. | Figures with CIs and a notebook/script that regenerates them from results/. |
| Q10 | Real-data runs and Table 3: S/T/X/DR (or T/DR) for log1p, arcsinh, sqrt and the best count-aware model at scales 0.01, 1, 100. | Table 3 and one figure; P reviews numbers. |
| Q11 | Reproducibility: README with exact commands and a script that rebuilds all figures from raw results. | P reproduces Fig 2 from a clean clone. |
| Q12 | Writing: abstract, introduction, experiments, real data, discussion and limitations, decision rule, Appendix B, 5-minute presentation outline. | Compiled manuscript. Decision rule example: with many zeros and a revenue or count goal, target the level-scale effect with a count-aware or hurdle model; use log-like transforms only with a sensitivity check across units. |


## 9. How I want the assistant to work

- Give runnable, tested code; say plainly what was verified and what was not, and never claim a result that was not computed.
- Keep estimands separate (level vs transformed) in every metric and every sentence of the paper.
- When I ask to understand a concept (CATE, meta-learners, Qini, hurdle models, Chen and Roth), explain it in this order: intuition, small example, definition, math, how it relates to this project, and one likely viva question.
- Respect the frozen interface contract; if a change is needed, say so and explain the impact on P's modules.
- I am on Windows with Python 3.13 and LightGBM; give PowerShell-friendly commands.

