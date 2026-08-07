# Feature Specification — 22 Rubric-Aligned Features (v0.3)

**Owner:** S1 (Gbadegbe) · **Status:** implemented and validated
**Consumed by:** `ingestion/feature_engineer.py` → XGBoost scorer → SHAP TreeExplainer
**Contract:** `model_contract.py` v2.0.0 — the frozen feature order lives there

Each lesson plan (PDF/DOCX) is converted to one 22-value feature vector. The
XGBoost model maps this vector to the rubric score (1–4 per criterion). Feature
names below are the exact column names used in code; the tutor-readable labels
shown in SHAP plots live in `explainability/feature_labels.py`.

**v0.3 added F19–F22**, one for each criterion that previously had no feature
measuring it. Same-corpus ablation, 5-fold CV over 200 synthetic plans:

| | 18 features | 22 features | Δ |
|---|---|---|---|
| C04 Resources/ICT | 0.548 | 0.743 | **+0.194** |
| C08 Attention to all learners | 0.568 | 0.761 | **+0.193** |
| C10 Lesson closure | 0.605 | 0.763 | **+0.159** |
| C06 Introduction/RPK | 0.674 | 0.747 | **+0.073** |
| other six criteria | — | — | −0.007 … +0.033 |
| **mean QWK** | **0.708** | **0.779** | **+0.071** |

The gain is concentrated on the four targeted criteria with no measurable cost
elsewhere, which is the evidence that the features — not the corpus — caused it.

The proposal (Part D3) explicitly names: SMART objective count, Bloom's
taxonomy verb distribution (6 levels), content-to-activity ratio, assessment
alignment score, and sentence complexity index — that is 10 of the 18. The
remaining 8 were proposed here.

## Rubric mapping — revised

> **v0.1 of this document mapped the features onto a six-dimension "GES/NCTE"
> rubric (D1–D6). That mapping is superseded and must not be used.** The live
> rubric is the **ten plan-assessable criteria** in `rubric_schema.py`, and the
> regulator is **GTEC** (NCTE merged into it under Act 1023, 2020).
>
> The authoritative feature → criterion → NTS mapping is
> `explainability/feature_labels.py`, which is validated against
> `model_contract.py` at import. This table is a summary of it; the code wins.

| Criterion | Direct features | Indirect |
|-----------|-----------------|----------|
| C01 Learning outcomes | F1, F2, F3 | F4, F5, F13 |
| C02 Pedagogical content knowledge | F4–F10 | — |
| C03 Teaching and learning strategies | F11, F12 | F6–F10, F19 |
| C04 Resources including ICT | **F19** | F12 |
| C05 Assessment strategies in the plan | F13, F14 | — |
| C06 Lesson introduction and RPK | **F20** | — |
| C07 Lesson sequencing and timing | F15, F16 | F20 |
| C08 Attention to all learners | **F21** | — |
| C09 Concept explanation and examples | F17, F18 | — |
| C10 Lesson closure | **F22** | F15 |

### The coverage gap — closed in v0.3

The v0.1 claim that "every rubric dimension is covered by at least 2 features"
held against the old six-dimension rubric. Against the ten criteria it did not:
four had no structural feature measuring them. A regressor was still trained for
each, so each still emitted a score, but its SHAP decomposition was over
features measuring something else and was not evidence about the criterion.

F19–F22 close that. **Keep the machinery that handled the gap.**
`uncovered_criteria()`, `coverage_report()` and `tabular_confidence()` compute
from the live map, and `reconcile.py` still withholds the structural channel for
any criterion below `STRUCTURAL_EVIDENCE_FLOOR`. Those returning "no gaps" is a
result, not dead code: if a criterion is ever added without a feature, the
dashboard degrades honestly instead of silently presenting a decomposition it
should not. There is a test that reintroduces a gap and asserts exactly that.

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
| F13 | `assessment_alignment_score` | float 0–1 | Cosine similarity between the objectives section and the assessment/evaluation section. Measures whether what is assessed matches what was promised. **Resolved:** implemented as lemma-overlap cosine (open question 1, option (a)) because `en_core_web_sm` ships no word vectors. |
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

### Criterion coverage (D-block additions, v0.3)

| # | Feature name | Type | Definition & extraction method |
|---|--------------|------|-------------------------------|
| F19 | `resource_specificity_count` | int 0–10 | Distinct concrete resources named in the resources section (falling back to the whole plan if there is no such section), matched against a noun taxonomy — flashcard, chart, real object, projector, … — plus 1 if any ICT resource appears. Counts *named* things on purpose: "TLMs will be provided" names nothing and scores 0, which is the judgement a tutor makes. |
| F20 | `rpk_link_score` | float 0–1 | Three equal parts: (a) an RPK/introduction section exists, (b) it carries a prior-knowledge cue ("previous knowledge", "already know", "recall", …), (c) its content words overlap the objectives. Part (c) is what separates a real link from the boilerplate "Teacher revises previous knowledge" that appears in every weak plan. |
| F21 | `differentiation_strategy_count` | int 0–6 | Distinct kinds of provision named, one count per bucket: support/scaffolding, extension, SEN, ability grouping, gender equity, language support. |
| F22 | `closure_quality_score` | float 0–1 | Four equal parts: the closure exists, it consolidates key points, the *learners* do the summarising, and attainment is checked. A closure that is only "Teacher summarises" scores partially — the distinction the rubric draws. |

Both F20 and F22 saturate near 1.0 for scores 3–4 on synthetic data, so they
separate weak-from-adequate better than adequate-from-excellent. Worth
re-checking against real plans, which will vary more.

Sections: F19–F22 need `resources` and `differentiation` headers, added to
`ingestion/extract_text.SECTION_PATTERNS` in v0.3, plus the existing `rpk` and
`closure`. `EXPECTED_SECTIONS` (wider than `MAIN_STAGES`) drives which absences
are reported to the suggestions panel. **Do not add to `MAIN_STAGES`** to
introduce a section — F15 is a share *of those stages*, so extending it silently
rescales a trained feature.

## Normalisation & missing-section policy

- Counts (F1, F2, F12, F14) are left raw — XGBoost handles scale; SHAP values stay interpretable ("+0.4 because 5 objectives").
- If a section is not found, its features default to 0 and a section-missing condition is logged by the extractor (feeds S2's "improvement suggestions" panel).
- All features are deterministic — same document always yields the same vector (needed for SHAP faithfulness review).

## Open questions for the group / supervisor

1. ~~**F13 vectors**~~ — resolved: lemma-overlap cosine, fully offline. See F13 above.
2. **Section header taxonomy:** still open. Need 2–3 real lesson plans from the
   tutors to finalise the header keyword lists (e.g. is "RPK" always spelled
   out? "Core points" vs "Content"?). `ingestion/extract_text.SECTION_PATTERNS`
   is still a draft, and a header the segmenter misses silently zeroes the
   features for that section.
3. **Bloom verb lexicons (Appendix A)** should be sanity-checked by a pedagogy
   teammate (S4/S5) before training on real data.
4. **The four uncovered criteria** (see "The coverage gap") — decide whether to
   add structural features in v0.3 or rely on the text channel. This changes
   the feature contract and needs retraining either way.
5. **`healthy_range` thresholds** in `explainability/feature_labels.py` gate
   which revision suggestions fire. They are pedagogical heuristics, not learned
   values, and want a review by S4/S5 against what tutors actually expect of a
   JHS-length lesson.

## Appendix A — Bloom's taxonomy starter verb lexicons

- **Remember:** define, list, name, state, recall, identify, label, match, recognise, select
- **Understand:** explain, describe, summarise, classify, discuss, interpret, paraphrase, illustrate, compare (basic), outline
- **Apply:** apply, use, solve, demonstrate, calculate, complete, show, implement, practise, sketch
- **Analyze:** analyse, differentiate, distinguish, examine, organise, contrast, categorise, investigate, deconstruct
- **Evaluate:** evaluate, justify, critique, judge, defend, argue, assess, appraise, recommend
- **Create:** create, design, construct, compose, develop, formulate, plan, produce, invent, generate

Vague (non-measurable, excluded from SMART): know, understand, learn, appreciate, be aware of, be familiar with, grasp
