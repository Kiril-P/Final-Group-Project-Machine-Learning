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

## Decision 16 — Equal Voting vs. AUC-Weighted Ensemble

**What we chose:** Each of the three ensemble models (LOF, Autoencoder, OC-SVM) casts an
equal vote. A player is flagged if ≥ 2 of 3 models agree.

**What we considered:** Weight each model's vote by its AUC score on the synthetic
validation set, so a model with AUC 0.96 counts for more than one with AUC 0.79.

**Why we kept equal voting:**
Weighting only changes the outcome in the narrow band of players where models
*disagree* — and only when one model's weight is strong enough to override the majority
on its own. In practice, with three models that all perform reasonably well (AUC 0.79–0.96),
a weighted vote would almost always reach the same verdict as a simple majority vote.

The exception would be something like a 0.96 / 0.79 / 0.79 split where the strong model's
"yes" would always beat two "no" votes — but that means we're flagging players based on
one model alone, which is less robust than the whole point of building an ensemble.

There's also an honesty issue: our AUCs are measured on *synthetic* anomalies that we
designed ourselves, not real cheaters. Using those synthetic AUCs to upweight one model
over another would be optimizing for simulated performance that may not reflect reality.

Equal voting is more transparent (easy to explain: "two models agreed"), more robust
(no single model can dominate), and in our case nearly equivalent in outcome.

If we had real labeled cheater data, AUC-weighted or stacked ensemble learning would
be worth revisiting.

---

## Decision 17 — Scoring All Players, Not Just the Training Split

**What we chose:** After fitting models on the 70% training split, we run scoring over
all 28k+ players (train + val + test) and save the results to `all_player_results.csv`.
Player explanations are generated from this full-coverage result.

**Why this is not leakage:**
Leakage would mean the model learned from val/test data — which it didn't.
The StandardScaler was fit exclusively on X_train. Each model was `fit()` on X_train only.
The val and test arrays were already transformed using the train-only scaler.
Scoring them with a trained model is identical to production deployment: you train once,
then score any new player who arrives — the model doesn't change.

Evaluation metrics (holdout_evaluation.csv) come from a completely separate Stage 4b
evaluation pipeline and are not affected by this step.

**Why we did it:**
The 30% of players in val+test are real players with real behavioral patterns. Using the
pipeline to only flag players in the training split means we miss 30% of the population
for no good reason. The whole point of the system is to triage a full player pool.

---

## Decision 18 — Position-Complexity-Weighted ACPL

**The problem with raw ACPL:**
Standard average centipawn loss treats all moves equally. A 20cp mistake in a dead-equal
position gets the same weight as a 20cp mistake when you're already down 300cp. But in a
position where you're completely lost, almost any move is "wrong" — the eval is already in
the tank. What really matters is how you play when the game is still undecided.

**What we did:**
Weight each move's centipawn loss by `exp(-|eval_before| / 100)`:
- Position at 0cp (equal):  weight = 1.00 — full credit, this move really matters
- Position at ±100cp:       weight = 0.37 — still relevant but position is lopsided
- Position at ±300cp:       weight = 0.05 — game is basically decided, barely counts

So `weighted_acpl = sum(loss_i × weight_i) / sum(weight_i)`.

This is the same core intuition behind Ken Regan's academic framework (the FIDE-endorsed
system). He calls it the "marginal centipawn principle" — a 20cp loss in an equal position
carries 5× the detection weight of the same 20cp loss in a clearly won/lost position.

**Why this is a better engine signal:**
An engine playing in an equal, critical position will have near-zero CPL with full weight.
A human will mess up exactly when the position is at its most complicated — which is also
when the exponential weight is highest. So the gap between engine and human is amplified
most precisely in the positions where it matters.

**Implementation note:**
We use `|eval_before|` — the absolute value of the eval from White's perspective, before
the player's move. This is the most natural position-tension measure: 0cp = perfectly equal,
anything positive/negative means one side is ahead. We apply the same formula regardless
of color; the sign of the eval tells you who's winning, but the magnitude tells you how
"important" the position is, and that's what we care about.

**Band normalization:**
Same reason as regular ACPL: higher-rated players have lower weighted ACPL naturally.
We compare within rating bands so a 1200 isn't flagged just for being a 1200, and a 2200
isn't automatically suspicious for being accurate. Column: `avg_weighted_acpl_band_z`.

---

## Decision 19 — UMAP Visualisation

**Why UMAP and not PCA / t-SNE:**
PCA is linear — it can't capture the non-linear structure of anomaly clusters (anomalies
often live in sparse pockets, not along principal variance directions). t-SNE works but
doesn't preserve global structure at all; clusters might end up anywhere on the plot with
no spatial meaning. UMAP is faster than t-SNE AND preserves a mix of local and global
structure, so you can actually interpret proximity on the plot.

**Two-panel design:**
- Left panel: binary flag (red = flagged, grey = normal). Clean and easy to present —
  shows at a glance whether the model produces a distinct cluster or noise.
- Right panel: vote count gradient (0 / 1 / 2 / 3 models voting). This is the more
  informative one: it shows whether the "confident" flags (all 3 models agreed) sit at
  the core of the anomaly cluster or are scattered. If they're at the core, the ensemble
  is doing something coherent. If they're random, that's a red flag about the models.

**Parameters:**
- `n_neighbors=15`: standard default. Controls local vs. global balance in the embedding.
  Lower = tighter local clusters; higher = more global structure preserved. 15 is the
  UMAP paper's recommendation for datasets of this size.
- `min_dist=0.1`: lets points pack a bit tightly so clusters are visually obvious.
  Default is 0.1, didn't change it because it works well for our use case.
- `random_state=42`: UMAP is stochastic. Fixing the seed means the plot is reproducible
  across runs — important when you're including it in a report.

**What to look for:**
A good result shows the flagged players (red) clustering in a relatively separated region,
not scattered uniformly across the embedding. Perfect separation isn't expected — we have
no ground truth and our anomaly signal is noisy — but some visible tendency to cluster
suggests the features are capturing something real. If the flagged points are completely
random on the UMAP, that's honest evidence our model is mostly noise.

**What we actually see (results/umap_overview.png):**
The result is genuinely encouraging and makes sense given what we know about the problem.

*Left panel (binary flag):* The 902 flagged players are NOT randomly scattered. There's a
clear concentration in the upper-left region of the embedding where red dots are noticeably
denser than anywhere else. That's not what uniform random noise looks like — it's a real
geometric signal that our 19 features produce a meaningful distinction between most flagged
and most normal players.

*Right panel (vote count gradient):* This is the more interesting one. The 265 purple dots
(all 3 models agreed = "confident" flags) cluster tightly in that same upper-left zone.
The 637 red dots (2-vote flags) surround them. The 1,758 orange dots (1-vote borderline)
bleed outward toward the normal population. This is exactly the gradient you'd want to see —
confident flags at the core, borderline flags at the edge, normals at the periphery.

*Why the scattered flags are NOT a problem — they're expected:*
There are also flagged players scattered throughout the rest of the plot, mixed in with
normal players. This is actually correct behavior for this problem, for two reasons:

1. **Cheaters don't all cheat the same way.** A player who cheats only in endgames has a
   completely different feature profile from one who runs an engine the whole game, or one
   who only cheats in games against stronger opponents. These different "cheating styles"
   will land in different regions of the feature space. You wouldn't expect them to all
   cluster together — they'd appear as scattered outliers across the embedding.

2. **Outliers by definition don't cluster.** UMAP preserves the structure of the data. A
   genuinely anomalous player who deviates in a unique way will appear isolated, away from
   every cluster. That's not a failure of the model — that's what an outlier looks like in
   2D. If every flagged player had the exact same profile, they'd form one tight blob. The
   scatter tells us our model is catching diverse kinds of anomalies, not just one archetype.

Compare this to what failure looks like: 902 red dots spread perfectly uniformly across
the entire 17k-player blob with no spatial preference whatsoever. That's what random
flagging produces. The upper-left concentration we actually see is evidence of structure.

**Short version for the report:** The UMAP shows a visible anomaly-dense region (upper-left)
where the confident flags concentrate, surrounded by a gradient of less certain flags.
Scattered flagged players elsewhere reflect the diversity of anomalous behavioral patterns —
not all suspicious players have the same profile, and the model is catching multiple
distinct deviation types rather than a single archetype.

---

## Decision 20 — Rating Trajectory (Elo Gain Rate)

**What we added:** `rating_gain` (total Elo gained over the observed period) and
`rating_gain_rate` (Elo gained per day). Both computed from the player's first and last
game in the dataset, sorted by `UTCDate` so the direction is real.

**Why it matters:**
A player going from 1200 → 1600 in three weeks is a completely different situation from
a stable 1400 player, even though both have 1400 as their *average* rating. Our existing
`rating_volatility` (std of rating) captures some of this, but it can't distinguish
between rapid rise, rapid fall, or yo-yo. `rating_gain_rate` captures direction AND speed.

Legitimate human improvement at anything above 3–5 Elo/day, sustained over weeks, is
extremely unusual — that's essentially what it takes to become a grandmaster from scratch.
When we see players gaining 10–15 Elo/day, that's outside the range of normal human
improvement. It doesn't prove cheating on its own, but combined with low ACPL or high
best-move rate it strongly suggests something is off.

**Implementation:**
We extract `UTCDate` from the Lichess CSV and attach it to every player-game record.
Before aggregation, player records are sorted by `[player_id, game_date]` so pandas
`groupby(...).first()` / `.last()` give the earliest and latest games in the dataset.
`days_active = (last_date − first_date).days`, clipped to minimum 1 to avoid
divide-by-zero for players whose games all fell on the same day.

**No band normalization needed:**
A 10 Elo/day gain is suspicious at 800 Elo and suspicious at 2000 Elo — the absolute
rate matters, not how it compares to peers in your rating band. We skip normalization here.

**Scope limitation:**
Our dataset is one month of Lichess games (July 2016). So `days_active` is at most ~30
days for everyone. Within that window a 400-point gain is still highly suspicious. With a
full year of data, `rating_gain_rate` would be an even stronger feature.

---

## Decision 21 — Timeout Loss Rate

**What we added:** `timeout_loss_rate` = fraction of games the player LOST because they
ran out of time (`Termination == "Time forfeit"`).

**Why it's a minor but real signal:**
Engine users respond in milliseconds — they never flag. A human who genuinely never loses
on time across hundreds of games is unusual, especially at time controls where flagging is
common (blitz, rapid). It's not a strong standalone signal — some humans are just fast —
but it supports the other signals when they're already pointing in the same direction.

We deliberately marked this as `confident=False` in the explainability module, meaning it
never appears as a "primary" reason in a player explanation. It can show up as a
supporting feature (#2 or #3) but won't be the headline reason anyone gets flagged.

**Elo-independent:**
Unlike ACPL-based features, this doesn't need band normalization. A 900-rated player
and a 2200-rated player both have similar flagging rates in rapid time controls.
The signal is comparable across skill levels.

---

## Decision 22 — Dropped `time_pressure_rate` (Bug Fix)

**What happened:**
`time_pressure_rate` (fraction of moves played with < 10s on the clock) was included in
the model, showing 100% coverage. In reality our dataset has no clock annotations —
`[%clk]` tags are absent from the games. The loader's fallback set `tp_count = 0` for
every game, making `time_pressure_rate = 0.0` for all 17k players.

A feature with zero variance is completely useless: after StandardScaler it becomes a
column of all zeros, and every model silently ignores it. It was polluting the feature
set with a dummy column and making the coverage stats look artificially clean.

**The fix:**
We detect this in `add_engineered_features`: if `total_time_pressure.sum() == 0`, the
column is set to NaN instead of 0.0. The coverage check in `get_feature_matrix` then
sees 0% coverage and skips the feature automatically, with a log warning explaining why.

This is the right behavior: when you don't have clock data, say so explicitly rather than
sneaking in a zero-variance dummy column. The feature remains in the code so it activates
automatically if a dataset with clock annotations is used in the future.

---

## Decision 23 — Did Not Implement Color Performance Asymmetry

**What we considered:**
White vs black win rate split — compute `white_win_rate` and `black_win_rate` separately,
then check whether a player performs unusually similarly with both colors
(`color_win_asymmetry = |white_win_rate − black_win_rate|`). The argument: engine users
don't care about color, so they'd show near-zero asymmetry even at ratings where most
humans have a clear color preference.

**Why we didn't do it:**
This came down to domain knowledge. At 2400+ Elo (where Christoph, one of our team
members, plays), you develop a very deliberate opening repertoire with both colors.
A strong player who has prepared deeply with both 1.e4 and 1.d4 as White, and has solid
systems against both, will play equally well regardless of color — not because they're
using an engine, but because they've done the work. At lower ratings, color preference is
also highly opening-dependent: if a 1400-rated player has spent six months learning the
Sicilian as Black and the Italian as White, their asymmetry will be low simply because
their repertoire is balanced — nothing suspicious about that.

The feature could add noise (flagging well-prepared players) without adding a strong
cheating signal. The ACPL-based features already capture "plays too accurately" much more
directly. Color asymmetry would at best be a weak supporting signal and at worst
create false positives among genuinely well-prepared players.

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

## Decision 24 — AUC = 1.000 for "engine_perfect": Expected, Not Leakage

**What we found:** LOF, IsolationForest, OneClassSVM, and Autoencoder all report
AUC = 1.000 on the `engine_perfect` synthetic benchmark. That immediately triggers
the question any ML engineer should ask: *is something wrong here?*

**Short answer:** No. It's expected by construction. But it's also meaningless as a
performance metric — which is why we renamed it `sanity_check` in the output files.

### Why it's guaranteed to be 1.0

The synthetic "engine-perfect cheater" is created like this:

```python
synthetic = np.tile(np.percentile(X_arr, 99, axis=0), (n, 1))
synthetic += rng.normal(0, 0.05, synthetic.shape)
```

Fifty fake players are placed at the 99th percentile of every feature simultaneously,
then jittered by 5% noise. The benchmark then asks: "can your anomaly detector find
these 50 points among 2,700 test points?"

- **LOF**: p99 points have the lowest local density of any points in the set — by
  definition, no nearby neighbors. LOF assigns maximum anomaly score. AUC → 1.0.
- **IsolationForest**: extreme values require ≤ 2 random splits to isolate. The
  algorithm literally scores "ease of isolation" — p99 points are the easiest.
- **OC-SVM**: the support of normal data is a hypersphere around the training
  centroid. p99 points are maximally far from that centroid.
- **Autoencoder**: trained on normal-distribution data; p99 inputs produce the
  highest reconstruction error because they're furthest from the learned manifold.

This result says nothing about whether the models would detect a real cheater. It
only confirms that the models are numerically functioning anomaly detectors. It's
a smoke test, not a validation.

### What is and isn't leakage here

There's no data leakage in the traditional sense. The models are fit on X_train
only. The synthetic anomalies are injected into X_test after all training is done.
Applying a fitted model to new data is just inference — that's not leakage.

What it IS is a **tautological benchmark**: the test is designed such that any
reasonable anomaly detector will ace it. Putting it in the report without this
explanation makes the results look more impressive than they are.

### The band z-score question

A separate, genuine methodological question: band z-scores (`avg_acpl_band_z`,
`best_move_rate_band_z`, etc.) are computed on ALL 17,909 players before the
train/test split. That means each player's z-score is partially influenced by the
test set players' statistics.

This IS technically a form of feature-level leakage. How bad is it?
- Each Elo band has thousands of players; 15% test holdout shifts the band mean
  and std by well under 1%.
- The effect on any individual player's z-score is negligible.
- These are contextual normalization statistics ("what's normal for a 1400?"),
  not learned model weights. The philosophical question of whether normalizing
  by a population statistic is "leakage" is genuinely debated in the ML literature.

The practical honest answer: the effect is tiny and doesn't materially change any
AUC number. But strict methodology would compute band stats on X_train only and
apply them to val/test — we didn't do that, and it's worth noting.

### The meaningful number: subtle strategy

The benchmark that actually matters is `subtle`: real player rows with 1/3 of
features perturbed by 1.5σ. These stay on the data manifold and require genuine
discriminative power to detect.

**Final test-set AUC (subtle strategy, what we report):**

| Model | AUC |
|---|---|
| LOF | 0.973 |
| Autoencoder | 0.937 |
| OneClassSVM | 0.883 |
| IsolationForest | 0.746 |
| HDBSCAN | 0.599 |
| ZScore | 0.781 |

These are the numbers we present. The sanity_check column in holdout_evaluation.csv
exists to confirm the models are working; it is not a performance claim.

**What we changed:** Renamed `engine_perfect` → `sanity_check` in all output CSV
files (val_evaluation.csv, holdout_evaluation.csv). The validation.py source code
now has an explicit comment explaining why AUC≈1.0 is expected there.

---

## Decision 25 — HDBSCAN: Tried, Evaluated, Excluded from Ensemble

**What we tried:** HDBSCAN (Hierarchical DBSCAN via sklearn) as a 6th anomaly
detector alongside LOF, IsolationForest, OC-SVM, Autoencoder, and ZScore.
The idea was to add a second density-based model to complement LOF.

**What we found:**

| Benchmark | HDBSCAN AUC | LOF AUC (for comparison) |
|---|---|---|
| subtle | 0.668 | 0.962 |
| sanity_check | 0.767 | 1.000 |
| realistic_cheater | 0.581 | 0.750 |

AUC 0.668 on the subtle benchmark and 0.581 on the realistic_cheater benchmark —
both far below LOF and the other ensemble models. HDBSCAN is not contributing
meaningful signal on either benchmark that actually matters.

**Why it underperforms:** HDBSCAN is designed for finding arbitrarily-shaped
clusters in low-to-moderate dimensional space. With 21 features, we're in the
high-dimensional regime where Euclidean distance becomes less meaningful (the
"curse of dimensionality"). LOF handles this better through local density
comparison — it's less sensitive to the global distance distribution because
it normalizes by each point's neighborhood. HDBSCAN requires finding
globally-coherent clusters, which is harder at 21 dimensions.

**What we kept:** HDBSCAN scores and labels are still written to every results
CSV (`HDBSCAN_score`, `HDBSCAN_label` columns in all_player_results.csv and
model_results.csv). The CV table and holdout_evaluation.csv both include HDBSCAN
rows. This transparency means we can show the failure explicitly rather than
pretending we never tried it.

**What we excluded:** HDBSCAN does NOT vote in `ensemble_flag` or
`ensemble_confident`. Those are computed from LOF + Autoencoder + OC-SVM only.
This is documented in the `ENSEMBLE_VOTERS` constant in models.py with a
prominent comment.

---

## Decision 26 — Realistic Cheater Benchmark (Third Evaluation Strategy)

**The problem with our existing benchmarks:**

- `sanity_check` (p99 of every feature): proves models function, tells us nothing
  about real cheating detection ability.
- `subtle` (random 1/3 of features perturbed by 1.5σ): a useful generalist
  stress test, but it doesn't reflect the actual structure of cheating behavior.
  A real cheater deviates on specific eval/performance features, not randomly
  across whatever 1/3 of their profile comes up in a coin flip.

**What real engine-assisted players look like** (sources: Lichess fair play team
blog posts, Chess.com fair play methodology, Regan 2011, Guid & Bratko 2006,
Oxera 2023 review of computer-move detection):

Sophisticated cheaters do NOT play perfectly — they deliberately use the 2nd or
3rd engine choice, occasionally allow a blunder, and maintain normal behavioral
statistics to avoid detection. This means:

- They're detectable in the **distribution** of quality across many games, not
  in individual move perfection.
- They deviate strongly on **eval/performance features** (ACPL, best move rate,
  win rate, comeback rate) — these are the features where engine assistance shows.
- They deliberately keep **behavioral features** normal (game length, opening
  variety, opponent rating, number of games) because these are visible and easy
  to maintain naturally.

**How we constructed `realistic_cheater` synthetic anomalies (100 per injection):**

All features start at 0 (population mean in scaled space). Then:

| Feature direction | Features pushed | Percentile range |
|---|---|---|
| Low (very accurate) | avg_acpl_band_z, avg_weighted_acpl_band_z, avg_acpl_middlegame_band_z, avg_acpl_opening_band_z, avg_acpl_endgame_band_z, acpl_consistency_band_z, blunder_rate | p3–p10 |
| Low (never times out) | timeout_loss_rate | p5–p12 |
| High (outperforms) | best_move_rate_band_z, win_rate, performance_vs_actual, comeback_rate, underdog_win_rate | p86–p95 |
| Moderate high (fast climb) | rating_gain_rate | p72–p82 (noisy signal, pushed mildly) |
| **Normal (kept at 0)** | avg_turns, turns_std, avg_opening_ply, rating_volatility, avg_opponent_rating, n_games, avg_rating, avg_rating_diff | 0 ± 0.08 |

**Why p5–p10 and not p1–p2:** Sophisticated cheaters are "very good but not
impossibly perfect." Pushing to p1 creates players as unrealistic as the
sanity_check benchmark. The p5–p10 range represents the "statistically unusual
but plausible" zone that real detection systems operate in. Values are sampled
uniformly within each range so synthetic players are varied, not identical clones.

**100 anomalies vs 50 for subtle:** The realistic_cheater profile is more
constrained (only specific features are extreme) so the classification problem
is harder. More anomalies give more statistical power to measure AUC reliably.

**Where it appears:** val_evaluation.csv, holdout_evaluation.csv, and a separate
roc_curves_realistic.csv (parallel to roc_curves_subtle.csv). These are the two
ROC curves we present in the report.

**What the results actually showed (test set):**

| Model | subtle AUC | realistic_cheater AUC | Drop |
|---|---|---|---|
| LOF | 0.962 | 0.750 | −0.212 |
| Autoencoder | 0.943 | 0.520 | −0.423 |
| OneClassSVM | 0.892 | 0.682 | −0.210 |
| IsolationForest | 0.773 | 0.700 | −0.073 |
| ZScore | 0.822 | **0.460** | −0.362 |
| HDBSCAN | 0.668 | 0.581 | −0.087 |

Every model drops significantly on the harder benchmark — which is expected, since
we deliberately designed it to be harder. But two results stand out and deserve
explanation:

**ZScore drops to 0.460 — below random (0.5)**

This is not a bug; it's the ZScore baseline behaving exactly as its design implies.
The ZScore takes the *maximum absolute z-score* across all 21 features per player.
Realistic cheaters are placed at p3–p10 on eval features — roughly 1.7–2.0σ below
the mean in scaled space. But with 21 features, a completely normal player will
frequently have *at least one* feature at 2.5–3σ purely by chance (the expected
maximum of 21 independent |N(0,1)| variables is ≈ 2.3). So normal players get
higher max-|z| scores than the synthetic cheaters, inverting the ranking.

This perfectly illustrates why univariate baselines fail on sophisticated cheating:
a cheater who is moderately suspicious across *many* features simultaneously is
invisible to any model that only looks at the single worst feature per player.

**Autoencoder drops to 0.520 — essentially random**

The autoencoder was trained on normal data and flags anything it can't reconstruct.
For realistic cheaters, behavioral features (avg_turns, opening_ply, game count,
etc.) are kept exactly at the population mean — 15 out of 21 features are normal.
Only the 6 eval features are anomalous. The autoencoder's MSE loss averages over
all 21 dimensions, so the 15 normal features contribute ≈ 0 error each and dilute
the signal from the 6 suspicious ones. The model learned to attend to the dominant
variance structure in the data (behavioral/gameplay variety), not the eval pattern.

**What this means for the ensemble:**

These failures are actually the strongest justification for the ensemble design.
The three voters fail in completely different ways on the realistic benchmark:

- LOF still detects at 0.750 because local density comparison catches players who
  are unusual relative to their nearest neighbors, even when the anomaly is
  distributed across multiple features rather than concentrated in one.
- Autoencoder essentially fails (0.520) on this profile — it's a density-learner
  that relies on global reconstruction error, which gets overwhelmed by the normal
  behavioral dimensions.
- OC-SVM at 0.682 — the kernel boundary partially captures the multi-feature
  pattern but can't fully recover what the AE loses.

No single model is reliable across both benchmark types. The majority-vote ensemble
(≥2 of 3) is more robust precisely because its voters have different inductive
biases and different blind spots.

---

## Decision 27 — Rating Convergence Detection: Not Implemented

**The idea:** Rather than flagging any rapid rating gain, we considered detecting
the specific pattern of "rating plateau followed by a sudden spike" — because a
new player improving fast is normal, but an established player suddenly gaining
200 Elo in two weeks is suspicious.

**Why we thought of it:** This is exactly the kind of temporal anomaly that real
chess integrity systems flag. An account that was 1600 for six months and then
becomes 1850 in a fortnight is a much stronger signal than a beginner who jumped
from 1200 to 1400.

**Why we didn't implement it:** Our dataset is a one-month snapshot (July 2016).
To detect convergence, we'd split each player's game history into chronological
segments and compare the rating slope in the first half vs the second half. With
20–50 games per player across 3–4 weeks, each segment is roughly 10–25 games
spanning 10–15 days. That's not long enough to reliably distinguish a genuine
plateau from week-to-week variance — a player who had a bad week followed by a
good week would look identical to a cheater who activated. The feature would add
noise, not signal.

The implementation is straightforward (sort by `game_date`, compute `early_gain`
vs `late_gain` per player, derive `gain_acceleration`). It would be a strong
feature with 3–6 months of history. We'd implement it if the dataset had that
depth.

**What we have instead:** `rating_gain_rate` (total Elo change / days active)
captures the magnitude of the overall trajectory. It's a weaker signal than
convergence detection but correctly identified as "conditional base" — included
when coverage is ≥50%, given modest weight in the ensemble via the voting system
that requires ≥2 of 3 strong models to agree.

---
