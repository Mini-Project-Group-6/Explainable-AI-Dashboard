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
- **Storage:** SQLite by default; PostgreSQL via docker-compose
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

S1 is complete — models are trained and the rubric is finalised. Active work is **S2**:

1. `model_contract.py` — declare artifact paths + version, the frozen feature-name list in
   training order, criterion IDs with NTS tags, and expected output shapes from XGBoost,
   SHAP, and DistilBERT. One `load_artifacts()` that validates feature names on load and
   raises loudly on mismatch.
2. Feature-label map — plain-English label per feature, its criterion, its NTS indicator.
   SHAP plots are only explainable if the axis labels are readable by a teacher educator.
3. Reconciling tabular SHAP values with DistilBERT token attributions into one
   per-criterion explanation.
4. Mapping negative feature contributions to concrete revision suggestions (rule-based).

## Conventions

- **Feature order is a contract.** Structural features must be defined identically at
  training and inference time. This has broken before. Never reorder or rename a feature
  without updating `model_contract.py` and retraining.
- **Do not modify training code** while working on S2 tasks without asking first.
- Everything downstream imports artifacts through `load_artifacts()`, never by reaching for
  model files directly.
- Improvement suggestions are rule-based, not generated text — it's more defensible to
  reviewers.

## Environment

Windows + PowerShell. Use the **official Python** installed via winget. MSYS2 Python is
incompatible with torch and will produce import errors that look like code bugs. Keep the
repo path short and outside OneDrive (`C:\dev\coteach`, not a synced Documents subfolder) —
long paths and sync interference break virtualenvs.
