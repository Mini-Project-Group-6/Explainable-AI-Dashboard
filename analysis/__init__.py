"""S5: statistical analysis of the trust study.

Reads the anonymised export written by ``python -m app.study export`` and
answers the three research questions — see ``analysis/report.py``.

    load        read the export, screen it, recompute scale scores
    stats       alpha, paired tests, effect sizes, Holm, Spearman
    report      the planned analyses, tables, figures, summary.md
    synthetic   an export with known effects, for testing the pipeline

    python -m analysis run --export data/export --out data/analysis
    python -m analysis synthetic --out data/analysis-synthetic

Everything is written under data/ by default, which is gitignored: even an
anonymised export is study data, and it does not belong in the repository.
"""
