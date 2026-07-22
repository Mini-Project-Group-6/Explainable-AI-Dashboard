# Feature Specification — 18 Rubric-Aligned Features (v0.1 DRAFT)

**Owner:** S1 (Gbadegbe) · **Status:** draft for group/supervisor review
**Consumed by:** `ingestion/feature_engineer.py` → XGBoost scorer → SHAP TreeExplainer

Each lesson plan (PDF/DOCX) is converted to one 18-value feature vector. The
XGBoost model maps this vector to the 6-dimension GES/NCTE rubric score
(0–4 per dimension). Feature names below are the exact column names used in
code and shown in SHAP waterfall plots, so they must be readable by tutors.

The proposal (Part D3) explicitly names: SMART objective count, Bloom's
taxonomy verb distribution (6 levels), content-to-activity ratio, assessment
alignment score, and sentence complexity index — that is 10 of the 18. The
remaining 8 are proposed here and mapped so **every rubric dimension is
covered by at least 2 features**.

## Rubric dimensions (GES/NCTE)

| # | Dimension | Features covering it |
|---|-----------|----------------------|
| D1 | Objectives | F1, F2, F3 |
| D2 | Content | F10 (+ F4–F9 as cognitive-demand signal) |
| D3 | Methods / delivery | F11, F12 (+ F10) |
| D4 | Assessment | F13, F14 |
| D5 | Language | F17, F18 |
| D6 | Time management | F15, F16 |

## Feature table

### Objectives (D1)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F1 | `objective_count` | int | Number of learning objectives detected. Objectives section located by header keywords ("objectives", "learning outcomes", "by the end of the lesson"); each bullet/sentence containing an instructional verb counts as one objective. |
| F2 | `smart_objective_count` | int | Objectives that are SMART-like: contain (a) a measurable Bloom verb (not "know"/"understand"/"appreciate"/"learn"), AND (b) a condition or criterion (e.g. "at least three…", "correctly…", "using the diagram…"). Detected via verb lexicon + dependency pattern for numeric/adverbial modifiers. |
| F3 | `objective_measurability_ratio` | float 0–1 | `smart_objective_count / max(objective_count, 1)`. Separates "many vague objectives" from "few but well-formed". |

### Cognitive demand — Bloom's taxonomy distribution (signals D1 + D2)

Proportions of all instructional verbs (in objectives + activities sections)
falling at each Bloom level, matched by lemma against the verb lexicons in
Appendix A. The six values sum to 1 (all zero if no instructional verbs found).

| # | Feature name |
|---|--------------|
| F4 | `bloom_remember_prop` |
| F5 | `bloom_understand_prop` |
| F6 | `bloom_apply_prop` |
| F7 | `bloom_analyze_prop` |
| F8 | `bloom_evaluate_prop` |
| F9 | `bloom_create_prop` |

### Content & methods (D2, D3)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F10 | `content_activity_ratio` | float | Token count of content/core-points sections ÷ token count of learner-activity sections. Balanced plans sit near 1; very high = lecture-heavy, very low = thin content. Capped at 5.0. |
| F11 | `learner_activity_verb_density` | float | Learner-centred activity verbs (discuss, demonstrate, present, solve, practise, group, role-play, …) per 100 tokens of the activities section. Proxy for learner engagement required by the GES participatory-methods criterion. |
| F12 | `activity_variety_count` | int | Number of distinct activity types detected from a keyword taxonomy: {discussion, group work, demonstration, practice/exercise, questioning, ICT/TLM use, role play, field/observation}. Range 0–8. |

### Assessment (D4)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F13 | `assessment_alignment_score` | float 0–1 | Cosine similarity between the mean spaCy vector of the objectives section and the mean vector of the assessment/evaluation section. Measures whether what is assessed matches what was promised. (Note: `en_core_web_sm` has no word vectors — computed from tf-idf overlap instead, or upgrade to `en_core_web_md`; see Open Questions.) |
| F14 | `assessment_item_count` | int | Number of assessment items detected in the evaluation section: interrogative sentences, numbered items, or imperative instructions ("list…", "explain…", "calculate…"). |

### Time management (D6)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F15 | `time_allocation_coverage` | float 0–1 | Share of the plan's main stages (introduction/RPK, presentation, activities, evaluation, closure) that carry an explicit time allocation (regex: `\d+\s*(min|minutes|mins)`). |
| F16 | `time_total_consistency` | float 0–1 | 1 − normalised absolute difference between the sum of stage minutes and the stated lesson duration (from the plan header, e.g. "Duration: 60 minutes"). 1 = perfectly consistent; 0 = off by ≥50% or no stated duration. |

### Language (D5)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F17 | `sentence_complexity_index` | float | Mean sentence length (tokens) × mean parse-tree depth, z-scored against the training corpus. Flags both fragmentary and run-on writing. |
| F18 | `readability_flesch` | float | Flesch Reading Ease over the full plan text (syllable counts via a rule-based counter — no extra dependency). Professional clarity signal. |

## Normalisation & missing-section policy

- Counts (F1, F2, F12, F14) are left raw — XGBoost handles scale; SHAP values stay interpretable ("+0.4 because 5 objectives").
- If a section is not found, its features default to 0 and a section-missing condition is logged by the extractor (feeds S2's "improvement suggestions" panel).
- All features are deterministic — same document always yields the same vector (needed for SHAP faithfulness review).

## Open questions for the group / supervisor

1. **F13 vectors:** `en_core_web_sm` (proposal-pinned) ships no word vectors. Options: (a) tf-idf lexical overlap — fully offline, simplest; (b) switch to `en_core_web_md` (~40 MB, still offline). Recommend (a) for v1.
2. **Section header taxonomy:** need 2–3 real GES-format lesson plans from the tutors to finalise the header keyword lists (e.g. is "RPK" always spelled out? "Core points" vs "Content"?).
3. **Bloom verb lexicons (Appendix A)** should be sanity-checked by a pedagogy teammate (S4/S5) before training.

## Appendix A — Bloom's taxonomy starter verb lexicons

- **Remember:** define, list, name, state, recall, identify, label, match, recognise, select
- **Understand:** explain, describe, summarise, classify, discuss, interpret, paraphrase, illustrate, compare (basic), outline
- **Apply:** apply, use, solve, demonstrate, calculate, complete, show, implement, practise, sketch
- **Analyze:** analyse, differentiate, distinguish, examine, organise, contrast, categorise, investigate, deconstruct
- **Evaluate:** evaluate, justify, critique, judge, defend, argue, assess, appraise, recommend
- **Create:** create, design, construct, compose, develop, formulate, plan, produce, invent, generate

Vague (non-measurable, excluded from SMART): know, understand, learn, appreciate, be aware of, be familiar with, grasp
