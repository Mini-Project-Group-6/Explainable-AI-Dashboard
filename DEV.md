# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this is

An Explainable AI Co-Teaching Dashboard for Ghana's Colleges of Education. It scores
student-teacher lesson plans against rubric criteria and surfaces SHAP-based explanations
so tutors and student teachers can see *why* a plan was scored the way it was.

This is an academic research project. The research question is whether explainable AI
feedback builds trust and improves lesson-plan quality, measured with TAM instruments in a
pre/post design. Target journal: *Teaching and Teacher Education*.

## Stack

- **Scoring:** XGBoost over structural/tabular features
- **Text layer:** DistilBERT fine-tuned on rubric criteria
- **Explanations:** SHAP (TreeExplainer for the tabular model)
- **Dashboard:** Streamlit, four tabs — scoring, revision history, trust survey, transparency
- **Storage:** PostgreSQL (`python -m app.database setup` creates it). SQLite only for the test suite
- **Testing:** Streamlit AppTest

## Rubric scope — important

The rubric is **ten plan-assessable criteria**, each tagged with its NTS indicator code,
derived from the official STS School Placement Handbook lesson observation checklist.

The official checklist has 25 items, but 15 of them assess **live teaching delivery** and
cannot be judged from a written plan. Those are deliberately excluded. This is a documented
scope limitation, not an oversight. Do not add criteria that require observing a lesson
being taught.

Regulatory naming: the current body is **GTEC** (NCTE merged into it under Act 1023, 2020).
Older material in this repo may say "GES/NCTE" — that phrasing is outdated and should be
corrected wherever it appears in user-facing text or manuscript material.

## Team roles

Five-student split. Know which layer you're touching:

| Role | Owns |
|------|------|
| S1 | XGBoost/DistilBERT training, SHAP pipeline |
| S2 | SHAP integration, explanation visualisation, XAI UI components |
| S3 | Streamlit dashboard, PostgreSQL, deployment |
| S4 | TAM survey instruments, data collection, IRB |
| S5 | Statistical validation, manuscript |

## Current work

### Environment

Python **3.12** venv at `.venv/`. `model/requirements.txt` records the working versions and
the two documented deviations from proposal D5 (`shap` and `spacy` have no cp312 wheels at
their pinned versions). Install torch from the CPU index, not plain PyPI — the default
Windows wheel bundles CUDA and is ~2.4GB for a project that never touches a GPU.

### Latency — why the text channel is split in two

Measured on the deployment-class CPU: **one DistilBERT forward pass at 512 tokens is
~1.2s**, and that is the floor. Raising `max_evals` 100 → 300 changed nothing; larger
batches were *slower* per item (1159 → 1358 ms); doubling torch threads did nothing. SHAP's
Partition explainer needs ~2 coalitions per text segment, so a 40-segment plan costs ~80
passes ≈ 93s. Not a tuning problem — the cost is the model.

So `RubricScorer` splits them:

| | Cost | Default |
|---|---|---|
| `with_text` — DistilBERT **score**, feeds the fitted blend | ~1.2s | on |
| `with_text_attributions` — **where in the prose** it looked | ~93s | off |

Default end-to-end: **1.76s** (features 0.37, structural 0.16, text 1.22) against the ≤15s
budget. Attribution is for S3 to precompute once per submission and cache, never to block an
upload. Withholding it costs little: v0.3 gave every criterion a structural feature, so
tabular SHAP explains all ten alone — text attribution existed for the four that had none.

Even aggressive optimisation (int8 quantisation + ONNX Runtime, ~4× at best) lands at
20–30s, so precompute is the answer rather than a faster path.

### Unrecognised plan formats

`SECTION_PATTERNS` is still a draft (FEATURES.md open question 2). A plan in an
unanticipated layout has most features default to 0 and would be scored as a very poor
lesson — a parsing failure and a bad lesson looked identical. Below
`SECTION_RECOGNITION_FLOOR` (4 of 9 expected sections) `LessonPlanText.format_recognised`
is False and `predict.py` puts `format_warning` at the head of the explanation's caveats.
The wording blames the tool, not the student teacher. See `tests/test_format_guard.py`.

### Regenerating artifacts

`model/artifacts/` and `model/data/` are committed (about 20 MB), so a fresh
clone scores plans without retraining. They are also regenerable. After a
retrain, commit every artifact together:

```
cd model
python -m data.synthetic -n 200 --out data/synthetic_v1
python -m data.synthetic --revisions 40 --out data/synthetic_revisions
python -m scoring.train_xgboost  --labels data/synthetic_v1/labels.csv --plans data/synthetic_v1/plans
python -m evaluation.cross_validate --labels data/synthetic_v1/labels.csv --plans data/synthetic_v1/plans
python -m explainability.shap_tree --model artifacts/rubric_model.joblib --labels data/synthetic_v1/labels.csv --plans data/synthetic_v1/plans --out artifacts/shap
python -m scoring.train_bert_lora --labels data/synthetic_v1/labels.csv --plans data/synthetic_v1/plans
```

### Validation — 5-fold CV, 200 synthetic plans, 22 features (contract v2.0.0)

Mean QWK **0.783** (target ≥ 0.70). Every criterion ≥ 0.638.

| Criterion | QWK | | Criterion | QWK |
|---|---|---|---|---|
| C01 Learning outcomes | 0.896 | | C10 Lesson closure | 0.831 |
| C05 Assessment strategies | 0.854 | | C07 Lesson sequencing | 0.826 |
| C09 Concept explanation | 0.848 | | C02 Pedagogical content knowledge | 0.790 |
| C08 Attention to all learners | 0.738 | | C06 Introduction/RPK | 0.722 |
| C04 Resources/ICT | 0.691 | | C03 Teaching & learning strategies | 0.638 |

**Same-corpus ablation** — the only variable is the feature set, so the gain is
attributable to F19–F22 rather than to the generator changes made alongside them:

| | 18 features | 22 features | Δ |
|---|---|---|---|
| C10 Lesson closure | 0.520 | 0.831 | +0.310 |
| C06 Introduction/RPK | 0.614 | 0.722 | +0.108 |
| C08 Attention to all learners | 0.650 | 0.738 | +0.088 |
| C04 Resources/ICT | 0.655 | 0.691 | +0.036 |
| other six criteria | — | — | −0.034 … +0.032 |
| **mean QWK** | **0.729** | **0.783** | **+0.055** |

> **Quote the mean, not the per-criterion deltas.** The same ablation was run on two
> independently generated 200-plan corpora. The mean gain was stable (+0.051, +0.055) but
> individual criteria moved by a factor of 2–3 between them: C10 +0.139 → +0.310, C08
> +0.184 → +0.088, C04 +0.082 → +0.036. Same code, same settings, only the random corpus
> differed. 200 synthetic plans under 5-fold CV simply do not pin a single criterion down.
> **The defensible claim is "adding F19–F22 raised mean QWK by roughly 0.05, concentrated on
> the four criteria they target."** Reporting a per-criterion figure to three decimals
> implies a precision this design does not have. Fold SD is in the table above; it
> understates the true variance, which the cross-corpus comparison exposes.

> **Corrected after code review.** An earlier version of this table reported mean +0.071.
> Two lexicons in `feature_engineer.py` matched by bare substring: `"sen"` matched inside
> "pre**sen**tation" — the canonical content-section header — so F21 scored ≥1 for
> essentially every plan, and F19 counted lesson subject-matter ("leaf", "seed", "stone")
> as teaching resources. Both features were therefore partly constants and the gains
> attributed to them were inflated. Matching is now whole-word via `_lexicon_pattern`, and
> `tests/test_format_guard.py` sweeps every lexicon term against common lesson-plan
> vocabulary so a new short cue cannot reintroduce this. The finding survives at a smaller
> size; see the variance note above before quoting any single figure.

### Corpus provenance — both channels must be fitted to the same data

Each trainer records a content hash of its training corpus (`model_contract.corpus_fingerprint`,
over plan ids and rubric scores), and `load_artifacts` compares them. This exists because
regenerating the synthetic data while reusing a text checkpoint left the two channels fitted
to different plans, and `predict` blended them into a score **with no warning at all** —
every existing check passed, because the feature names still matched.

| State | Response |
|---|---|
| fingerprints differ | blending disabled, warning, user-facing caveat |
| only one recorded | fitted weights discarded, warning, "provisional" caveat |
| both match | silent — the normal case |
| neither recorded | silent (pre-fingerprint artifacts; refusing would be worse) |

**Retraining either channel invalidates the blend weights.** Refit afterwards:
`python -m evaluation.blend_weights --labels ... --plans ...`

**These remain synthetic-data numbers and are optimistic.** Quality profiles are
correlated by design, so a criterion can be partly predicted from features measuring other
criteria. They are not evidence the model holds up on real CoE plans.

### Text channel — DistilBERT + LoRA, held-out (46 plans), v0.3 corpus

745,738 of 67,706,900 parameters trainable (1.10%). `epochs=20`, early-stopped at 19 with
the best checkpoint at epoch 16 (val MSE 0.4489).

**Epoch count was worth measuring, not guessing.** At the original 8 epochs the model was
still improving and reached val MSE 0.5453 / mean QWK 0.636. Allowed to run properly it
reached 0.4489 / **0.712** — biggest single gains C07 +0.254, C05 +0.156, C04 +0.116. Note
this is a *schedule* parameter: `get_linear_schedule_with_warmup` spreads LR decay over
`steps × epochs`, so raising it changes the whole trajectory rather than appending epochs.
The superseded 8-epoch checkpoint is kept at `artifacts/bert_rubric_e8` for comparison.

| Criterion | Tabular (5-fold CV) | Text (held-out) |
|---|---|---|
| C01 Learning outcomes | **0.896** | 0.719 |
| C05 Assessment strategies | **0.854** | 0.746 |
| C09 Concept explanation | **0.848** | 0.802 |
| C10 Lesson closure | **0.831** | 0.703 |
| C07 Lesson sequencing | **0.826** | 0.786 |
| C02 Pedagogical content knowledge | **0.790** | 0.737 |
| C08 Attention to all learners | **0.738** | 0.682 |
| C06 Introduction/RPK | 0.722 | **0.726** |
| C04 Resources/ICT | 0.691 | **0.740** |
| C03 Teaching & learning strategies | **0.638** | 0.627 |
| **MEAN** | **0.783** | **0.727** |

Best val MSE 0.3865 at epoch 8, early-stopped at 11. Both channels are fitted to corpus
`963d4c91`, recorded in each artifact — see the provenance guard below.

> **Correction — an earlier version of this file claimed the text model beat the tabular
> model on the four criteria structure could not measure, and called that the
> channel-weighting design "confirmed by measurement". That was measured on the pre-v0.3
> corpus and does not hold.** On that corpus the generator emitted one fixed sentence per
> score for exactly those blocks ("Slower learners will be helped." = score 2), so the text
> model was matching fixed strings — the same artifact class as the F21 bijection fixed at
> the same time. Once the generator sampled from phrase pools with jitter, the advantage
> disappeared: the tabular model now leads on **all ten** criteria.
>
> What survived as clean evidence is the same-corpus ablation above: adding F19–F22 raised
> mean QWK 0.708 → 0.779 with gains concentrated on the four target criteria.
>
> **Second correction (after proper training).** With `epochs=8` the text channel looked
> like it might not earn its place at all. It was undertrained. At 20 epochs it beats the
> structural channel outright on C02 and C04, and the fitted blend beats both channels alone
> on 8 of 10 criteria. The right claim is narrower than the original one and better
> supported: *the two channels make partly uncorrelated errors, so blending them at
> per-criterion weights fitted to held-out error beats either alone* — not "text rescues the
> criteria structure cannot measure", which was a generator artifact.

Held-out QWK is written to `artifacts/bert_rubric/holdout_report.json` and loaded by
`TextRubricScorer.reliability`, which `predict.py` feeds to `reconcile` as per-criterion
`text_reliability`. Regenerate with `python -m scoring.train_bert_lora ... --eval-only`.

### Channel blending — fitted, not assumed

`channel_weights` has three fallbacks, in descending order of trustworthiness:

1. **Fitted** (`evaluation/blend_weights.py`) — the weight minimising held-out error on
   plans *both* channels were validated against. This is what runs.
2. **Measured** — both channels' held-out QWK, compared like for like.
3. **Heuristic** — coverage vs a flat prior, when nothing has been measured.

Fitted structural weights on 46 shared held-out plans:

| Criterion | w_struct | blended RMSE | struct only | text only |
|---|---|---|---|---|
| C01 Learning outcomes | 0.92 | **0.383** | 0.386 | 0.599 |
| C02 Pedagogical content knowledge | 0.69 | **0.465** | 0.491 | 0.585 |
| C03 Teaching & learning strategies | 0.59 | **0.685** | 0.713 | 0.742 |
| C04 Resources/ICT | 0.15 | **0.604** | 0.676 | 0.607 |
| C05 Assessment strategies | 0.61 | **0.460** | 0.518 | 0.588 |
| C06 Introduction/RPK | 0.48 | **0.600** | 0.652 | 0.645 |
| C07 Lesson sequencing | 0.81 | **0.496** | 0.502 | 0.601 |
| C08 Attention to all learners | 0.60 | **0.660** | 0.677 | 0.696 |
| C09 Concept explanation | 0.53 | **0.395** | 0.436 | 0.444 |
| C10 Lesson closure | 0.80 | **0.470** | 0.484 | 0.664 |

**The blend beats both channels alone on all ten criteria**, and no fitted weight is 0 or 1
— both channels contribute everywhere. That is the strongest form of the ensemble argument
and the only real justification for running two models: their errors are partly
uncorrelated on every criterion.

Earlier fits produced weights of exactly 0.00 (C04) and 1.00 (C06), dropping a channel
outright. Those were fitted across a corpus mismatch — the text checkpoint had been trained
on data the structural model never saw. The provenance guard now makes that state
detectable; see below.

Refit after retraining either channel — the weights are only valid for the checkpoints they
were fitted against:
`python -m evaluation.blend_weights --labels ... --plans ...`

Both trainers share `train_xgboost.validation_mask`, so the two channels are validated on
identical plans; `blend_weights._assert_shared_holdout` checks this rather than trusting it.
Refit after retraining either channel:
`python -m evaluation.blend_weights --labels ... --plans ...`

> **Corrected defect, kept for the record.** The weighting originally compared
> `tabular_confidence` — a *coverage* heuristic where 1.0 just means "has ≥2 features" —
> against the text channel's *measured* QWK. Different kinds of quantity, and the result was
> perverse: the text model got measurably worse on every criterion yet its weight rose. The
> like-for-like fix alone was not enough (proportional-to-accuracy still gave the weaker
> channel 45%), which is why the weights are now fitted to held-out error.

Held-out QWK is written to `artifacts/bert_rubric/holdout_report.json` and loaded by
`TextRubricScorer.reliability`, which `predict.py` feeds to `reconcile` as per-criterion
`text_reliability` — replacing the flat `TEXT_CHANNEL_PRIOR` guess. Regenerate with
`python -m scoring.train_bert_lora ... --eval-only`.

> **Windows: torch must be imported before xgboost.** xgboost ships its own OpenMP runtime
> and once it is loaded, torch's `c10.dll` cannot initialise (`OSError: [WinError 1114]`).
> spaCy pulls torch in through `thinc.compat`, so the traceback surfaces at the spaCy
> import and looks like a broken spaCy install. `model/compat.py` fixes the order and is
> called from `model_contract`; do not remove that call. Reproduce with
> `python -c "import xgboost; import torch"`.

> **Fixed label leak — re-read before quoting any earlier metric.** The first CV run
> reported QWK 1.000 with zero variance for assessment strategies. `_assessment_block`
> rendered exactly `score` numbered items, and F14 counts numbered items, so the feature
> *was* the label (80/80 exact match). Activity counts had the same shape, and
> `_apply_language_quality` derived writing quality from the mean of all ten scores, leaking
> the whole profile into F17/F18. All three are fixed; the 0.724 above is post-fix. Any
> metric recorded before this is invalid.

### Brief objectives → where they live

| # | Objective | Status |
|---|-----------|--------|
| 21 | Train XGBoost/BERT to score plans against the rubric | Done: `scoring/`; artifacts trained and loading under contract 2.0.0 (synthetic data only) |
| 22 | SHAP feature-importance for *every* feedback item | `suggestions.py` — each item carries its Shapley value, influence share and `attribution_text`; non-SHAP rules declare themselves |
| 23 | Streamlit dashboard: scores, SHAP waterfall, suggestions | S2 components in `render.py` (`st_waterfall`, `st_force_plot`, `st_beeswarm`, `st_suggestions_panel`, `st_transparency_panel`); the dashboard shell itself is S3 |
| 24 | Measure trust + plan quality before/after | S4 built: instruments, consent and pre/post collection in `app/study/`, export via `python -m app.study export`. **Materials are draft until ethics approval** (`docs/irb/README.md`). Synthetic before/after pairs: `data/synthetic.py --revisions N`. Analysis is S5 |

Method & Tools calls for waterfall, **force plot** and summary visualisations — all four
forms are in `explainability/shap_tree.py` (`save_waterfall`, `save_force_plot` /
`force_plot_html`, `save_beeswarm`, `save_summary_bar`).

S2 is implemented:

1. ✅ `model/model_contract.py` — artifact paths + version, the frozen 22-feature list in
   training order, the 10 criterion IDs with NTS tags, expected output shapes from XGBoost,
   SHAP and DistilBERT, and `load_artifacts()`. Validates on load *and* at import: a
   reorder or rename in `feature_engineer.py` now raises immediately. Currently v2.0.0 —
   the v0.3 feature additions invalidated every 1.x bundle, and the contract rejected the
   stale one with an exact diff rather than letting SHAP index the wrong columns.
2. ✅ `model/explainability/feature_labels.py` — plain-English label, criterion and NTS
   indicator per feature, plus `coverage_report()` / `uncovered_criteria()`.
3. ✅ `model/explainability/reconcile.py` — folds tabular SHAP and text attributions into
   one per-criterion explanation. The two are never summed; scores are blended by channel
   weight, evidence is ranked by within-channel influence share.
4. ✅ `model/explainability/suggestions.py` — rule-based revisions. A suggestion needs both
   a negative contribution *and* a measured value outside its healthy range.
5. ✅ `model/explainability/render.py` — XAI UI components. Framework-free view models plus
   Streamlit widgets for S3 to drop in: SHAP waterfall/force/beeswarm embeds, the
   text-attribution highlighter, the suggestions panel and the transparency panel. S3 never
   needs to import `shap` or manage a matplotlib figure.

Tests: `cd model && python -m unittest discover -s tests` — 121 tests, stdlib only. They run
on a bare checkout with no ML stack installed, which is the point: the contract's rules are
checkable without loading the models.

### Naming: the brief says "GES and NCTE", this repo says GTEC

Objective 21 in the compendium predates Act 1023 (2020), under which NCTE merged into
**GTEC**. Per the rubric-scope section above, user-facing text and manuscript material use
GTEC. The ten criteria themselves are unchanged — this is a naming correction, not a scope
change. Worth a line in the methods section so the supervisor sees it is deliberate.

### The coverage gap is closed — keep the machinery that handled it

Four criteria (resources/ICT, introduction/RPK, attention to all learners, closure) used to
have no structural feature measuring them. **v0.3 added F19–F22 and closed that.**

`uncovered_criteria()`, `coverage_report()` and `tabular_confidence()` still compute from
the live feature map, and `reconcile.py` still withholds the structural channel below
`STRUCTURAL_EVIDENCE_FLOOR`. They now return "no gaps" — that is a result, not dead code.
If a criterion is ever added without a feature, the dashboard degrades honestly instead of
presenting a decomposition that isn't evidence. `test_the_gap_machinery_still_works_if_a_gap_reappears`
reintroduces a gap and asserts exactly that; don't delete it.

### S3 — dashboard, storage, accounts

- **PostgreSQL is required.** With no `DATABASE_URL` or `DB_HOST`+`DB_NAME`,
  nothing is stored and sign-in stays closed; the server log says why. The old
  silent fallback to `data/coteach.db` is gone: it let a machine without its
  `.env` write study data outside the study database.
  - `python -m app.database setup`: creates the `coteach` role and a database
    it owns, and writes `.env`. It asks for the superuser's password once, or
    reads `PGPASSWORD`. Without `--port`, it uses the newest *running* server
    the Windows installer registered (`HKLM\SOFTWARE\PostgreSQL\Services`). On
    the dev machine that is PostgreSQL 18 on **5433**, because the installer
    avoided 17's 5432. The port goes into `.env`, so the app stays on that
    server.
  - `python -m app.database check`: connects and shows the server version and
    row counts. If the connection fails, it names the local server that is
    running.
  - `python -m app.database copy-sqlite data/coteach.db`: brings old SQLite
    data across. It never changes the source, and a second run copies nothing.
- **Backups.** `python -m app.database backup` writes a custom-format
  `pg_dump` to `data/backups/` (gitignored) and keeps the newest 14.
  - The dump only counts once `pg_restore --list` has read it back and found
    every table.
  - `pg_dump` must be at least as new as the server, so it is taken from the
    installer's own `bin` folders, newest first. `COTEACH_PG_BIN` overrides
    that.
  - `schedule-backup` registers the Windows task "CoTeach database backup":
    daily at 18:00, runs as the signed-in user under `pythonw`, and catches up
    after a missed day. Results go to `data/backups/backup.log`. It uses an
    inline PowerShell command because the dev machine's execution policy is
    AllSigned. The task is registered on the dev machine.
  - To restore: `pg_restore --clean --if-exists --no-owner -d coteach <file>`.
  - The dumps sit on the same disk as the database, so an off-machine copy is
    still to be arranged (data-management plan section 8).
- **Tests** run on a scratch SQLite file:
  `.venv\Scripts\python -m unittest discover -s tests -t .`. To run the same
  suite on PostgreSQL, set `COTEACH_TEST_DATABASE_URL` to a database whose name
  ends in `_test`. Every app table in that database is dropped first. The
  harness blanks the `DB_*` variables rather than deleting them: `db.py` calls
  `load_dotenv()`, which refills absent variables, and deleting them once let a
  developer's `.env` point the tests at their dev database.
- **Accounts.** There is no self-registration.
  `python -m app.accounts import class.csv --passwords <new file>` creates a
  cohort all at once. Existing accounts are skipped and never reset. The
  password sheet is refused inside the repository, except under `data/`. Reset
  a forgotten password with `passwd <email> --generate`.
- **Failure handling.** `get_engine()` only proves the database was up at
  startup. Reads and writes after that catch `SQLAlchemyError`: saving and
  history degrade to session memory, the study reads as unavailable, and
  sign-in says "unavailable" without counting the attempt towards lockout.
  Anything shown to people uses `db.display_url()`, which masks the password.
- **`.streamlit/config.toml`** turns off browser usage statistics (the ethics
  documents promise no third-party collection) and hides tracebacks and the
  developer menu. It also caps uploads at 25 MB.
- **Schema changes.** `create_all` creates missing tables but never alters
  existing ones. Adding a column to a deployed table needs a migration.
- **Deployment is not decided yet.** Only the database container exists. Still
  needed: an app service, HTTPS (passwords currently cross the network
  unencrypted), and backups.

### S4 — trust study (`app/study/`, `docs/irb/`)

**Instruments.** TAM — perceived usefulness and ease of use (Davis 1989),
behavioural intention (Venkatesh & Davis 2000) — plus Hoffman et al. (2018)'s
XAI trust scale, all on one 5-point agreement scale. Hoffman's explanation
satisfaction scale is post-only. 22 items pre, 30 post; the 22 are identical in
both waves. TR6 ("I am wary") is reverse-keyed. The adaptation table
(`docs/irb/survey_instruments.md`) is generated from `instruments.py`, and a test
fails if the two differ.

**The gate** (`flow.py`, pure). A student teacher must accept or decline before
their first upload. If they accept, the pre-survey must come before any plan is
scored, so the baseline precedes AI feedback. The post-survey opens after
`COTEACH_POST_AFTER_SUBMISSIONS` (default 2) plans scored *since* the
pre-survey. Declining unblocks immediately and is never asked again. Tutors and
researchers are never asked. Withdrawal deletes survey answers and is final in
the app.

**Approval is pinned to content.** `approval.fingerprint()` hashes the items,
the two participant documents and the procedure (roles, post threshold).
`is_approved()` needs `IRB_PROTOCOL` set, `APPROVED_FINGERPRINT` matching, no
`{{PLACEHOLDERS}}` left and every source in `_VERIFIED_SOURCES`. Until then,
responses are stored with `irb_protocol = NULL` (pilot), and the export drops
them unless `--include-pilot` is given. Editing approved wording drops the
study back to pilot on its own. That is intended.

**Export** (`python -m app.study export --out data/export`, or the researcher
view in the Trust survey tab): `participants.csv`, `responses.csv`,
`submissions.csv`, keyed by random `P-XXXXXXXX` codes. Item columns are raw;
`*_mean` columns are reverse-keyed. No statistics here — that is S5.

**Status: draft.** `python -m app.study status` lists what is outstanding:
placeholders in the participant documents, and item wording that has not been
checked against the papers (the `source_text` was transcribed, not copied).
`docs/irb/README.md` lists the claims the colleges must confirm before
submission.

### Pending verification

NTS indicator codes in `model_contract.CRITERIA` are matched by **descriptor wording**
against the National Teachers' Standards; the letter suffixes still need one pass against
the printed STS handbook. Add each confirmed code to `_VERIFIED_NTS_CODES`;
`unverified_criteria()` reports the rest and the UI captions itself accordingly.

## Conventions

- **Participant-facing wording lives in `docs/irb/`, never in code.** The app
  reads the consent form and information sheet from there, so the committee
  and the participants see the same text.
- **Nothing identifying leaves the database.** Exports carry participant codes
  only — no emails, names or plan file names (students name files after
  themselves).

- **Feature order is a contract.** Structural features must be defined identically at
  training and inference time. This has broken before. Never reorder or rename a feature
  without updating `model_contract.py` and retraining. `model_contract` now holds an
  independent copy of the list and compares it to the extractor at import, so drift fails
  loudly instead of silently misattributing SHAP values.
- **Do not modify training code** while working on S2 tasks without asking first.
- **Never sum a tabular SHAP value and a text attribution.** They come from different
  models over different input spaces; the sum decomposes nothing. `reconcile.py` blends
  scores and ranks evidence by within-channel influence share instead.
- **A suggestion needs a measured weakness, not just a negative SHAP value.** Telling a
  student teacher to fix something they already did well is the failure mode this project
  exists to measure.
- Everything downstream imports artifacts through `load_artifacts()`, never by reaching for
  model files directly.
- Improvement suggestions are rule-based, not generated text — it's more defensible to
  reviewers.

## Environment

Windows + PowerShell. Use the **official Python** installed via winget. MSYS2 Python is
incompatible with torch and will produce import errors that look like code bugs. Keep the
repo path short and outside OneDrive (`C:\dev\coteach`, not a synced Documents subfolder) —
long paths and sync interference break virtualenvs.
