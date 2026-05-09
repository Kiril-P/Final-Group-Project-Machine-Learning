# Project Decisions Log

This file documents every significant methodological choice we made, why we made it,
what we considered but rejected, and what we'd do differently with more time or data.
Think of it as a lab notebook — the reasoning behind the code, not just the code itself.

---

## Core Limitation — Read This First

**We have no ground-truth labels. There are no confirmed cheaters in our dataset.**

This is the fundamental constraint of the entire project, and it shapes every decision
we made. It is not a gap in our methodology — it is a reality of the problem domain
that every serious chess integrity system faces.

Without confirmed cheater labels, we cannot report a real precision or recall rate.
Our AUC numbers are measured against synthetically injected anomalies that we created
ourselves — they tell us how well the model detects the *kind* of anomaly we designed,
not how well it would perform on actual confirmed cheaters.

**What this system is:**
A behavioral triage tool. It efficiently identifies the players most worth a human
reviewer's attention out of a large population. It ranks statistical outliers — players
whose aggregated behavior is unusual relative to peers at their rating level across
multiple independent signals.

**What this system is not:**
A verdict engine. Flagging a player does not mean they are cheating. A flagged player
could be a naturally talented outlier, someone on a hot streak, an intensive opener
who studied their lines deeply, or simply statistical noise. The system explicitly
labels which signals it is confident about and which are model-detected but ambiguous.

**Why unsupervised is the right approach here:**
The two alternatives — supervised learning and semi-supervised learning — both require
labeled data. Chess.com and Lichess treat their ban lists as confidential. Lichess's
own supervised model (Kaladin) is trained only on player confessions, which is a biased
and incomplete label set. Given the data available to us, unsupervised anomaly detection
with honest uncertainty reporting is not a compromise — it is the appropriate method.

**This limitation is acknowledged explicitly throughout the project** — in the
validation strategy (synthetic injection with two difficulty levels), in the ensemble
design (triage list vs. high-confidence shortlist), in the per-player explanation
output (confident vs. model-based signals), and in the report.

---

## 1. Dataset: Lichess July 2016 (6.25M games) over the small Kaggle CSV (20k games)

**What we chose:** The big Lichess dataset, sampled to 500k games → ~28k unique players.

**Why:** The small Kaggle dataset only gave us ~1,100 players after aggregation. That's
too few for robust anomaly detection — a 5% contamination rate means ~55 suspected
anomalies, which isn't enough signal. The Lichess export also has embedded Stockfish
eval annotations (`[%eval ...]`) which unlock move-quality features (ACPL, blunder rate,
best move rate) — the strongest behavioral signals we have.

**What we considered:** Running on the small dataset first to verify the pipeline, then
switching. We did this — the small dataset still works and is the default fallback.

**Tradeoff:** The Lichess dataset is a specific month (July 2016), so results may not
generalize perfectly to modern play. Acceptable for a research project of this scope.

---

## 2. Time control filter: rapid + classical only

**What we chose:** Filter to rapid (10-30 min) and classical (30+ min) games only.

**Why:** Cheating via engine consultation requires time to think. Bullet and blitz players
physically can't consult an engine between moves fast enough to matter — the anomaly
signal in those games is mostly noise. Restricting to slower time controls means our
behavioral features actually measure something meaningful.

**Tradeoff:** We lose a large chunk of the dataset (blitz is by far the most common
format on Lichess). Worth it for signal quality.

---

## 3. Features: 13 base + extended, but move_time_cv is missing

**What we have:** 8 base features (win rate, rating volatility, etc.) + 5 extended
features from Stockfish evals (ACPL, blunder rate, best move rate, comeback rate,
time pressure rate). 13 total.

**What we wanted but couldn't get:** `move_time_cv` — the coefficient of variation of
per-move think times. A player using an engine has suspiciously *uniform* think times:
every move takes roughly the same amount of time because they're waiting for the engine.
This is probably the single strongest behavioral cheating signal there is.

**Why it's missing:** The Lichess July 2016 export doesn't include `[%clk ...]` clock
annotations in the move text. Without move timestamps, we simply can't compute this.
The feature is implemented in `features.py` and will activate automatically if a dataset
with clock data is ever used.

**Future work:** Any Lichess export from ~2017 onwards includes clock annotations.
Rerunning on a more recent dataset would likely significantly improve model performance,
especially for subtle cheaters who look normal on eval features alone.

---

## 4. Validation strategy: synthetic anomaly injection (no real labels)

**The core problem:** We have no confirmed cheaters in our dataset. Lichess doesn't
publish its ban list. Without ground-truth labels, we can't compute a "real" ROC-AUC.

**What we did:** Inject synthetic anomalies in two flavors:
- `engine_perfect`: push features to the 99th percentile — obvious cheaters, easy test
- `subtle`: take real player rows and nudge a few features by ~1.5 standard deviations —
  harder, more realistic, the number we actually care about

**Why this is the right call:** Every unsupervised anomaly detection paper on behavioral
data does something like this. There's no better alternative when labels don't exist.

**Honest limitation:** Our "subtle" injection is designed to be detectable by density-
based models like LOF — so LOF's AUC 0.96 partly reflects that our test is somewhat
self-fulfilling. A truly realistic test would need confirmed cheater data.
We report this limitation explicitly.

---

## 5. Train/val/test split: 70/15/15 with scaler fit on train only

**What we chose:** 70% train, 15% val (hyperparameter tuning), 15% test (final numbers).
StandardScaler is fit exclusively on the training rows — the same learned mean/std is
then applied to val and test without refitting.

**Why:** If we scaled on all data before splitting, the scaler's mean and std would
include information from the val and test sets. That's data leakage — it makes metrics
look better than they'd be on truly new data. Fitting on train only is the correct way.

**Why 70/15/15 and not 80/20:** We wanted a proper validation set for hyperparameter
search separate from the test set. Using the test set for tuning would invalidate it.
15% gives ~4,300 players per held-out split which is large enough for stable metrics.

---

## 6. Hyperparameter search: random search on validation set with synthetic injection

**What we chose:** 20-iteration random search per model, evaluated on the val split
with 50 injected synthetic anomalies (subtle strategy).

**Why random search over grid search:** The search spaces are large enough that a full
grid would be computationally expensive. 20 random draws gives a good exploration of
the space for a dataset of this size (diminishing returns above ~20 for these models).

**Autoencoder exception:** We run a small exhaustive grid (encoding_dim × threshold
percentile) at reduced epochs (30 instead of 100) during search, then retrain the
winner at full epochs. This keeps search cost manageable without skipping the AE.

---

## 7. HDBSCAN: kept in results, removed from ensemble

**Why we added it:** HDBSCAN is conceptually appealing for anomaly detection — it finds
dense clusters of normal players and marks everything outside as noise. No fixed
contamination parameter needed for the clustering itself, unlike IF or LOF.

**What happened:** On this dataset, HDBSCAN flags exactly 0 players in the final
results. It can rank players by anomaly score (AUC ~0.64-0.67 on synthetic tests), but
the threshold never gets crossed — likely because the 13D feature space is sparse enough
that HDBSCAN forms loose clusters with low membership probabilities, pushing the 95th
percentile threshold up to 1.0. Strict `> threshold` then flags nobody.

**Note:** sklearn's HDBSCAN is transductive (can't score new data). We work around this
with a KNN approximation: store training membership probabilities, then for any new
point average the probabilities of its k nearest training neighbors. This adds noise
and likely hurts performance vs. the native approach.

**Decision:** Keep HDBSCAN in the pipeline for completeness and reporting. Remove it
from the ensemble — contributing zero votes serves no purpose. If `move_time_cv` were
available, tighter clusters might form and HDBSCAN could become useful.

---

## 8. IsolationForest and ZScoreBaseline: kept in results, removed from ensemble

**Overlap analysis results:**
- IsolationForest: AUC 0.72, flags 1,612 players, 882 "unique" (not caught by LOF/AE)
- ZScoreBaseline: AUC 0.76, flags 1,008 players, 491 "unique"

**Decision:** At AUC 0.72-0.76, unique catches from these models are more likely
false positives than real anomalies the stronger models missed. Adding them to the
ensemble would pollute the flag list with noise. They're retained in `model_results.csv`
and reported in the paper — we did the work, it just tells us they're the wrong tool
for this feature space.

**Why IsolationForest underperforms:** Surprising given it's the go-to model for anomaly
detection. Most likely reason: random feature splits in high(ish)-dimensional space don't
cleanly separate anomalies when normal players form a complex, non-uniform distribution.
LOF's local density approach handles this better.

---

## 9. Final ensemble: LOF + Autoencoder + OneClassSVM, majority vote (≥2/3)

**The three voters and why:**
- **LOF** (AUC 0.957): density-based, best at subtle outliers in dense regions
- **Autoencoder** (AUC 0.959): reconstruction-based, different failure modes from LOF
  (LOF and AE only agree ~26-35% of the time — genuinely complementary)
- **OneClassSVM** (AUC 0.842): margin-based, adds 159 unique catches not found by
  either LOF or AE — a meaningful third perspective at reasonable accuracy

**Flag tiers:**
- `ensemble_flag` (≥2/3 agree): triage list, high recall — players worth a second look
- `ensemble_confident` (3/3 agree): shortlist, high precision — players most likely
  to be genuinely anomalous

**What this system is and isn't:**
This is a first-pass filter, not a verdict. The output is a ranked list of behaviorally
unusual players that warrants human review. We make no claim that flagged players are
cheaters — only that their aggregate behavior is statistically unusual relative to the
rest of the population.

---

## 10. Feature importance: IsolationForest used for permutation importance

**Why IF for importance given it's a weak model:** Permutation importance measures how
much a model's performance drops when a feature is shuffled. Using LOF for this would
be better in principle, but LOF is a transductive method — its O(n²) fitting cost makes
permutation testing over many features and repetitions slow. IF is fast and gives a
reasonable importance ranking even if its absolute anomaly scores are weaker.

**Future improvement:** SHAP values per-player using LOF or AE scores would give both
global importance *and* per-player explanations ("this player was flagged because their
ACPL is 3 standard deviations below average for their rating band").

---

## 11. Rating-band normalization: only avg_acpl and best_move_rate, not all eval features

**The question:** Should we normalize eval-based features within rating bands before
feeding them into the model?

**The answer:** Only for avg_acpl and best_move_rate — and the reasoning matters.

A key insight: win rate doesn't need normalization because the Elo system already handles
it. Both a 1200 and a 2400 player should converge to ~50% win rate over enough games —
that's literally what Elo does. The suspicious signal is deviation from 50%, not absolute
win rate, and we already capture that with `win_rate_vs_expected`.

For avg_acpl and best_move_rate it's different. These directly measure move quality,
which scales with Elo. A 1200 naturally has ACPL ~60-80; a 2000 naturally has ACPL ~20-30.
Without normalization, a 1200 with ACPL 40 looks suspicious against the whole population,
but is actually normal-to-suspicious for their band — we can't tell. Normalizing within
band makes the comparison fair.

For blunder_rate, comeback_rate, time_pressure_rate: we decided NOT to normalize.
Blunders are blunders regardless of Elo — yes, lower-rated players blunder more, but
the suspicious signal is someone who blunders far less than their peers, which the
global comparison captures fine. Same logic for the others.

**Implementation:** New columns `avg_acpl_band_z` and `best_move_rate_band_z` computed
in `add_engineered_features`. The model uses these instead of the raw values. Raw columns
are kept in `player_features.csv` for reporting.

---

## 12. Per-player explainability: z-score breakdown with honest confidence labels

**Why:** Flagging a player and saying nothing else is a black box. A human reviewer
needs to know *why* to make a sensible decision. Without explanation, the system
is just a list of names.

**Why NOT just say why for everything:** Some features have a clear, intuitive cheating
interpretation (low ACPL → plays like an engine). Others are statistically anomalous but
ambiguous (unusual game length could mean anything). Making up confident explanations
for unclear signals would be dishonest and could mislead reviewers into wrong conclusions.

**What we did:** For each ensemble-flagged player, compute within-band z-scores for all
model features and find the top-3 most deviant. Each feature has a pre-defined
"suspicious direction" (high or low) and an explanation text. Features where we're
confident in the interpretation are marked `confident=True`; features that are just
statistically anomalous are marked `confident=False` with text that says
"model-detected, not directly interpretable."

**Confident signals** (we know what they mean):
avg_acpl_band_z, best_move_rate_band_z, win_rate_vs_expected, performance_vs_actual,
underdog_win_rate, comeback_rate, time_pressure_rate, blunder_rate, rating_volatility

**Model-detected only** (anomalous but ambiguous):
avg_turns, opening_ply_ratio, victory_efficiency, win_rate (alone, without context)

**Output:** `results/player_explanations.csv` — one row per flagged player with the
top 3 signals, their z-scores, explanations, and a plain-English summary.

---

## 13. ACPL consistency across games (STDCPL), band-normalized

**What it is:** The standard deviation of `avg_acpl_game` across all of a player's
games — how much their move quality varies from game to game.

**The signal:** An engine is unnaturally consistent. It plays at roughly the same
level every game regardless of the position, the opponent, or how tired the human
behind the keyboard is. A genuine human has good days and bad days — their ACPL
varies noticeably across games. Unusually low STDCPL for a player's rating band is
therefore suspicious.

**Why band normalization is necessary here too (user's insight):** Higher-rated
players are naturally more consistent than lower-rated ones — their skill floor is
higher so there's less room to fluctuate. A 2200 player might vary between ACPL 20-35
across games; a 1200 player might vary between 50-120. Without band normalization
we'd flag strong players as suspicious just for being consistently good, which is
wrong. Comparing within band makes the signal meaningful.

**One caveat:** The std estimate needs enough games to be reliable. With our minimum
of 5 games per player it can be noisy. We include it and let the ensemble absorb
the noise — a single noisy feature doesn't break LOF or the Autoencoder.

**Feature name:** `acpl_consistency_band_z` (suspicious direction: unusually LOW).

---

## 15. Per-game-phase ACPL: opening / middlegame / endgame split

**Why:** Both Chess.com and Lichess look at accuracy broken down by game phase, not
just as a flat average. A flat avg_acpl can miss the most common cheating pattern:
a player who uses theory in the opening (so their opening accuracy looks normal)
and then turns on the engine once the position gets complicated.

Splitting into three phases gives us two new signals:
- **Middlegame ACPL** (moves 11-30): where engine assistance matters most. Positions
  are complex and unique — a human makes mistakes here, an engine doesn't.
- **Phase gap** (opening ACPL − middlegame ACPL): if this is unusually large for
  a player's rating band, they're playing at human level in theory and engine level
  once theory runs out. That's the clearest behavioral signature of mid-game cheating.

**Why opening ACPL is a weaker signal:** Opening moves are heavily memorised. A
dedicated player who has studied their opening lines can have very low opening ACPL
without any engine help. We include it in the feature set but mark it as lower
confidence in the explainability output.

**Phase boundaries:** moves 1-10 (opening), 11-30 (middlegame), 31+ (endgame).
Standard thresholds used in academic chess analysis literature.
NaN for any phase with fewer than 3 player moves — avoids misleading averages
from games that ended very early.

**Band normalization:** all three phase ACPL values and the phase gap are normalized
within rating bands, for the same reasons as overall avg_acpl (see decision #11).

---

## Future Work — What We Would Do With More Time or Data

These are things we identified, thought through, and consciously decided not to
implement — not because they aren't valuable, but because of data or time constraints.
Documenting them here rather than just leaving them out.

---

### F1. Move-time features (per-move clock data)

**What it is:** The coefficient of variation of per-move think times (std / mean).
An engine responds almost instantly to every position, so its "think time" per move
is just the time spent waiting + clicking — nearly constant. A human varies a lot:
fast on forcing lines, long on critical decisions. Low move-time CV = suspiciously
robotic pacing.

**Why we don't have it:** The Lichess July 2016 export has no `[%clk ...]` clock
annotations in the move text. Clock annotations only appear in Lichess exports from
roughly mid-2017 onwards. The feature is fully implemented in `lichess_loader.py`
and `features.py` and will activate automatically if a dataset with clock data is used.

**Impact if available:** Probably the single strongest cheating signal. Chess.com
explicitly uses move timing as a primary detection factor.

---

### F2. Move sequence pattern modeling (LSTM / transformer)

**What it is:** Rather than aggregating features at the player level, model the
*sequence* of moves directly. Engines don't just make good moves — they make moves
in a recognisably algorithmic order. Lichess's Irwin system uses a neural network on
move sequences for exactly this reason.

**Why we didn't implement it:** Requires per-game sequence data going into a model,
not per-player aggregated statistics. This is a fundamentally different architecture
— a sequence model on top of our current pipeline, not an extension of it. Significant
scope increase with real risk of not finishing before the deadline.

**What would be needed:** Parse each game into a sequence of (move quality, position
complexity, eval delta) tuples, feed into an LSTM or transformer, train on labeled
data (or use contrastive self-supervised learning). Ground-truth labels remain the
hard problem.

---

### F3. Opening deviation analysis

**What it is:** Track where each player deviates from known opening theory (e.g.
from an ECO database), and measure how their accuracy *drops off* after the deviation
point. A strong player deviates from theory into known good alternatives — their
accuracy stays high. An engine user's accuracy stays high indefinitely because the
engine handles any position. A genuine beginner's accuracy drops sharply after
deviation because they're improvising.

**Why we didn't implement it:** Requires an opening database (e.g. Lichess opening
explorer) to look up move novelty points. Not in our current dataset.

---

### F4. Mirrored game / cheating ring detection

**What it is:** Lichess catches cases where two accounts play identical or near-identical
move sequences in the same time window — one player is consulting an engine and
mirroring moves against a different opponent. Purely network-based detection.

**Why we didn't implement it:** We're doing individual behavioral profiling, not
cross-player network analysis. Detecting cheating rings would require building a
game-similarity graph across all players, which is a separate project.

---

### F5. Account-level and device signals

**What it is:** Chess.com uses over 100 factors including device fingerprinting, IP
patterns, session behavior, tab-switching frequency, and account creation patterns.
A brand-new account that immediately plays at a high level with suspicious behavioral
features is far more likely to be cheating than an established account.

**Why we don't have it:** None of this is in any public dataset. It's proprietary
platform data. We could approximate it with account age (first game date) from Lichess
data if we had the full history, but the July 2016 export doesn't include account
creation dates.

---

---
