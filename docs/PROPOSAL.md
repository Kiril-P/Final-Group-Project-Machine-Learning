# Project proposal: Behavioral anomaly detection in human chess

| Field | Detail |
|--------|--------|
| **Working title** | Unmasking the board: behavioral anomaly detection in human chess |
| **Course** | BCSAI — Machine Learning Foundations (group project) |
| **Dataset** | Lichess-style games ([Kaggle: datasnaek/chess](https://www.kaggle.com/datasets/datasnaek/chess)); ≥20k rated games |
| **Date** | 30 April 2026 |
| **Status** | Approved by Prof. Matteo Turilli (see `docs/Professor Feedback.md`) |

---

## 1. Problem statement and scope

We study **unsupervised anomaly detection** on **player-level aggregates** derived from online chess games. The scientific question is:

> Among players in a given rating context, whose **observed behavioral statistics** are **statistically inconsistent** with the bulk of that population?

This is **not** a project to prove “smurfing” or fair-play violations. There is **no reliable ground-truth label** for that outcome. Following faculty guidance, all reported results will be framed as **detectable deviations** or **systematic differences** relative to a rating-conditioned reference—not as definitive identification of wrongdoing.

**Intended use:** prioritize cases for **human review** (e.g. platform moderation workflows), with explicit caveats for benign explanations (see §7).

---

## 2. Learning objective (ML framing)

| Aspect | Choice |
|--------|--------|
| **Task** | Unsupervised anomaly detection (continuous anomaly scores + optional binary flags from contamination or top‑k thresholds). |
| **Unit of analysis** | **Player** (one vector per player after aggregating across many games). |
| **Reference group** | Peers in similar **Elo band** and **time control** (e.g. blitz vs rapid analyzed separately). |
| **Success** | (i) Flagged players differ from the rest on held-out features and scores; (ii) differences **cohere with an external engine signal** where available; (iii) **synthetic** known anomalies rank highly under controlled injection. |

This matches the evaluation progression recommended in faculty feedback: **population contrast → external grounding → synthetic recovery**.

---

## 3. Data

- **Source:** Public Kaggle extract of Lichess games (structured CSV: ratings, clocks, openings, outcomes, move lists).
- **Scale:** >10k games (course guideline); we use the full extract available in the repository pipeline.
- **Cleaning:** Drop incomplete rows; enforce minimum **games per player** to stabilize aggregates; parse **increment_code** into time-control categories; retain columns needed for features and optional engine analysis.

**Leakage discipline:** Any scaling or unsupervised fitting is applied **after** a proper train/validation/test split (or cross-validation) at the **player** level where applicable, so that the same player does not inform both normalization thresholds and final scoring in an invalid way. The final report will state the exact split policy.

---

## 4. Feature engineering (comparable across players)

Features are chosen to be **interpretable** and **comparable** across players, with explicit controls for confounders (per Prof. Turilli: game count, time control, opponent strength).

| Theme | Examples (player-level aggregates) |
|--------|-------------------------------------|
| **Results vs expectation** | Win rate vs **Elo-implied** expected score from mean opponent rating. |
| **Game shape** | Mean plies, dispersion of game length; opening theory depth (`opening_ply`) normalized by rating (“opening sophistication”). |
| **Schedule of play** | Dominant **time control** category from `increment_code`; analysis **stratified** by category when sample sizes allow. |
| **Opponent field** | Mean opponent rating; rating differential summaries (pool strength). |
| **Stability** | Rating volatility (e.g. standard deviation of own rating across games in window). |
| **Optional external signal** | **Average centipawn loss (ACPL)** from Stockfish on a **fixed random subsample** of games (cost-controlled); used mainly for **validation and reporting**, not as the only anomaly definition. |

We will document definitions, scaling (e.g. z-score within band), and any transformations fit **only on training** data in the pipeline notebook and report.

---

## 5. Models and baselines

**Principle:** Classical, interpretable detectors are **primary**; neural methods are **secondary comparisons**, not the main claim—consistent with faculty feedback on model complexity.

| Tier | Methods |
|------|---------|
| **Baselines** | Simple rules and reference scores (e.g. univariate extremes, distance to centroid / k‑NN style scores in scaled space) to show that non-trivial structure is not an artifact of a single exotic algorithm. |
| **Core detectors** | **Isolation Forest**; **One-Class SVM** (RBF). Hyperparameters tuned with a bounded search (e.g. `RandomizedSearchCV`) with budget justified in the report. |
| **Optional neural comparator** | Small **feedforward autoencoder** on the same feature vector: anomaly score = reconstruction error. **If it does not outperform simpler methods, that is reported as a finding.** |

**Agreement analysis:** Compare rankings or top‑k overlap across methods to argue conclusions are not sensitive to a single inductive bias.

---

## 6. Validation and evaluation (no ground-truth labels)

### 6.1 Layer A — Statistical separation

- Compare **flagged vs non-flagged** players (or high vs low anomaly quantiles) on the feature space: **Welch t-tests** (or nonparametric alternatives where assumptions fail), effect sizes, and score distributions.
- **Clustering metrics** (e.g. silhouette, Davies–Bouldin) will be used **only when** a clear partition is defined (e.g. model-based normal vs anomaly labels), with careful wording—no “silhouette theater” without a defined clustering task.

### 6.2 Layer B — External grounding (engine)

- Where ACPL (or related engine summaries) is available, report **correlation and calibration** between anomaly scores and engine-implied strength, with confidence intervals / multiple-testing awareness as appropriate.
- Interpretation: high anomaly score with **better-than-peer** engine metrics in a **low rating band** is **consistent with** the behavioral mismatch hypothesis; it is **not** proof of policy violation.

### 6.3 Layer C — Synthetic anomalies

- **Inject** a small number of **known** synthetic outliers (e.g. extreme feature vectors mimicking “engine-like” statistical profiles) into the scaled feature matrix.
- Report **precision@k**, **recall@k**, **ROC-AUC / AP** on the augmented ranking task to show the pipeline can recover **planted** deviations under controlled conditions.

### 6.4 Course alignment

The final submission will mirror course expectations: **EDA and preprocessing**, **pipelines** without leakage, **multiple models + baselines**, **systematic tuning**, **appropriate metrics**, **interpretability** (permutation importance; SHAP where applicable for tree models), **failure cases**, and **reflection** on limitations and ethics.

---

## 7. Interpretability, ethics, and limitations

- **Interpretability:** Feature importance (permutation; SHAP for tree-based components); qualitative review of top-flagged players’ raw statistics.
- **Benign high scores:** Rapid improvement, opening specialists, sandbagging concerns, pool mismatch, or data artifacts can all inflate anomaly scores. The report will include a **failure-mode** subsection with concrete examples.
- **Ethics:** No public naming of accounts as “cheaters”; results are **statistical screening** outputs for informed human follow-up only.
- **Negative results:** If signal is weak or methods disagree, we will report that honestly—per course emphasis, negative findings are valid outcomes when analyzed.

---

## 8. Deliverables and timeline (course)

Aligned with the Machine Learning Foundations group assignment:

| Deliverable | Role of this proposal |
|-------------|------------------------|
| **Code + notebook pipeline** | Implements §3–§6; reproducible `README` and scripts. |
| **Report (≤2000 words + figures)** | Condenses problem, data, methods, results, limitations, team reflection. |
| **Presentation & poster** | Same narrative: question → data → methods → results → what failed / surprised us. |

Internal milestones: lock feature definitions and split policy → implement baselines + core models → complete synthetic and engine validation blocks → freeze figures for poster/report.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Stockfish runtime dominates | Fixed subsample depth and game cap; ACPL as validation layer, not required for all rows. |
| Sparse players / bands | Minimum games per player; merge thin rating bands if needed; report sample sizes. |
| Overclaiming | Language audit: “consistent with”, “flags for review”, “detectable deviation”. |

---

## 10. References (initial)

- Course project brief (local): `docs/ML_group_project-2.pdf`
- Faculty feedback (local): `docs/Professor Feedback.md`
- Lichess open data policy: [database.lichess.org](https://database.lichess.org/)
- Kaggle dataset card: [datasnaek/chess](https://www.kaggle.com/datasets/datasnaek/chess)
- Pedregosa et al., *JMLR* 12 (2011) — scikit-learn  
- Liu et al. — Isolation Forest; Schölkopf et al. — one-class SVM (standard citations in final report bibliography)

---

*This document is the canonical project description for graders and collaborators; implementation details live in `README.md` and `src/`.*
