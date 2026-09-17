# analysis/report.py
"""Run the planned analyses and write tables, figures and a summary.

    RQ1  Do trust and acceptance change between the surveys?
         PU, PEOU, BI and TR, paired pre → post, Holm across the four.
    RQ2  Does plan quality improve across drafts?
         First vs last scored plan, overall and per criterion (Holm across ten).
    RQ3  Does the change in trust go with the change in quality?
         Spearman, change in each scale vs change in overall score
         (Holm across four; TR is the primary one).
    Descriptives: reliability of every scale in each survey, explanation
    satisfaction (post only), background questions.

"First vs last" uses each participant's first and last scored plan,
whatever the phase: with the upload gate, the first plan always follows the
pre-survey anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis import stats
from analysis.load import StudyData, keyed_items
from app.study import instruments

PAIRED_SCALES = [s.key for s in instruments.SCALES if set(s.waves) == {"pre", "post"}]
PRIMARY_SCALE = "TR"

# Reference palette (dataviz skill): slot 1 blue, slot 2 orange; chrome and ink.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


@dataclass
class Results:
    reliability: pd.DataFrame
    prepost: pd.DataFrame
    satisfaction: pd.DataFrame
    quality_overall: pd.DataFrame
    quality_criteria: pd.DataFrame
    link: pd.DataFrame
    background: pd.DataFrame
    change: pd.DataFrame          # per participant: Δscales, first/last score
    trajectories: pd.DataFrame    # participant_code, sequence, overall_score
    figures: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def _wave(data: StudyData, wave: str) -> pd.DataFrame:
    return data.responses[data.responses["wave"] == wave].set_index("participant_code")


def reliability(data: StudyData) -> pd.DataFrame:
    rows = []
    for scale in instruments.SCALES:
        for wave in scale.waves:
            frame = _wave(data, wave)
            items = keyed_items(frame, scale.key)
            complete = items.dropna()
            means = frame[scale.key].dropna()
            rows.append({
                "scale": scale.key, "construct": scale.name, "wave": wave,
                "n": len(complete), "items": items.shape[1],
                "alpha": stats.cronbach_alpha(complete.to_numpy()),
                "mean": means.mean() if len(means) else math.nan,
                "sd": means.std(ddof=1) if len(means) > 1 else math.nan,
            })
    return pd.DataFrame(rows)


def prepost(data: StudyData) -> pd.DataFrame:
    pre, post = _wave(data, "pre"), _wave(data, "post")
    codes = sorted((set(pre.index) & set(post.index)) - data.late_baseline)
    rows = []
    for key in PAIRED_SCALES:
        result = stats.paired(pre.loc[codes, key], post.loc[codes, key])
        rows.append({"scale": key, "construct": instruments.scale(key).name,
                     "primary": key == PRIMARY_SCALE, **result.as_row()})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = stats.holm(frame["p"]) if len(frame) else []
    return frame


def satisfaction(data: StudyData) -> pd.DataFrame:
    post = _wave(data, "post")
    rows = []
    for scale in instruments.SCALES:
        if set(scale.waves) != {"post"}:
            continue
        values = post[scale.key].dropna()
        rows.append({"scale": scale.key, "construct": scale.name, "n": len(values),
                     "mean": values.mean() if len(values) else math.nan,
                     "sd": values.std(ddof=1) if len(values) > 1 else math.nan,
                     "median": values.median() if len(values) else math.nan})
    return pd.DataFrame(rows)


def _first_last(data: StudyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each participant's first and last scored plan (those with at least two)."""
    plans = data.submissions.copy()
    plans["overall_score"] = pd.to_numeric(plans["overall_score"], errors="coerce")
    plans = plans.dropna(subset=["overall_score"]).sort_values(
        ["participant_code", "sequence"])
    counts = plans.groupby("participant_code")["sequence"].transform("size")
    plans = plans[counts >= 2]
    first = plans.groupby("participant_code").head(1).set_index("participant_code")
    last = plans.groupby("participant_code").tail(1).set_index("participant_code")
    return first, last


def criterion_columns(data: StudyData) -> list[str]:
    from app.study.export import SUBMISSION_BASE_COLUMNS
    return [c for c in data.submissions.columns if c not in SUBMISSION_BASE_COLUMNS]


def quality(data: StudyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    first, last = _first_last(data)
    overall = stats.paired(first["overall_score"], last["overall_score"])
    overall_frame = pd.DataFrame([{"measure": "Overall score (0–100)",
                                   **overall.as_row()}])
    rows = []
    for column in criterion_columns(data):
        result = stats.paired(pd.to_numeric(first[column], errors="coerce"),
                              pd.to_numeric(last[column], errors="coerce"))
        rows.append({"criterion": column, **result.as_row()})
    criteria = pd.DataFrame(rows)
    if len(criteria):
        criteria["p_holm"] = stats.holm(criteria["p"])
    return overall_frame, criteria


def change_scores(data: StudyData) -> pd.DataFrame:
    pre, post = _wave(data, "pre"), _wave(data, "post")
    codes = sorted((set(pre.index) & set(post.index)) - data.late_baseline)
    frame = pd.DataFrame(index=pd.Index(codes, name="participant_code"))
    for key in PAIRED_SCALES:
        frame[f"delta_{key}"] = post.loc[codes, key] - pre.loc[codes, key]
    first, last = _first_last(data)
    frame["first_score"] = first["overall_score"].reindex(frame.index)
    frame["last_score"] = last["overall_score"].reindex(frame.index)
    frame["delta_score"] = frame["last_score"] - frame["first_score"]
    return frame.reset_index()


def link(change: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in PAIRED_SCALES:
        result = stats.spearman(change[f"delta_{key}"], change["delta_score"])
        rows.append({"scale": key, "primary": key == PRIMARY_SCALE,
                     "n": result.n, "rho": result.rho, "p": result.p})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = stats.holm(frame["p"]) if len(frame) else []
    return frame


def background(data: StudyData) -> pd.DataFrame:
    pre = _wave(data, "pre")
    rows = []
    for question in instruments.BACKGROUND:
        answers = pre[question.id] if question.id in pre else pd.Series(dtype=object)
        total = len(pre)
        for option in question.options:
            count = int((answers == option).sum())
            rows.append({"item": question.id, "question": question.text,
                         "answer": option, "n": count,
                         "percent": 100 * count / total if total else math.nan})
        skipped = int(answers.isna().sum()) if total else 0
        rows.append({"item": question.id, "question": question.text,
                     "answer": "(not answered)", "n": skipped,
                     "percent": 100 * skipped / total if total else math.nan})
    return pd.DataFrame(rows)


def analyse(data: StudyData) -> Results:
    change = change_scores(data)
    overall, criteria = quality(data)
    plans = data.submissions.copy()
    plans["overall_score"] = pd.to_numeric(plans["overall_score"], errors="coerce")
    return Results(
        reliability=reliability(data),
        prepost=prepost(data),
        satisfaction=satisfaction(data),
        quality_overall=overall,
        quality_criteria=criteria,
        link=link(change),
        background=background(data),
        change=change,
        trajectories=plans[["participant_code", "sequence", "overall_score"]]
        .dropna().sort_values(["participant_code", "sequence"]),
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=9)
    ax.grid(color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="semibold",
                 pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9,
            va="bottom")


def _figure(width: float, height: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Helvetica", "Arial",
                                       "DejaVu Sans"]
    figure, ax = plt.subplots(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    return plt, figure, ax


def _save(plt, figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def figure_prepost(data: StudyData, results: Results, path: Path) -> Path | None:
    """Dot plot: each scale's mean with its 95% CI, first survey vs follow-up."""
    pre, post = _wave(data, "pre"), _wave(data, "post")
    codes = sorted((set(pre.index) & set(post.index)) - data.late_baseline)
    if len(codes) < 2:
        return None
    scales = PAIRED_SCALES + [s.key for s in instruments.SCALES
                              if set(s.waves) == {"post"}]
    plt, figure, ax = _figure(7.2, 4.2)
    from scipy import stats as sps

    def mean_ci(values):
        values = values.dropna()
        if len(values) < 2:
            return math.nan, math.nan
        half = sps.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))
        return values.mean(), half

    labels = []
    for row, key in enumerate(scales):
        y = len(scales) - 1 - row
        only_post = set(instruments.scale(key).waves) == {"post"}
        labels.append((y, f"{key} · {instruments.scale(key).name}"
                          + ("\n(follow-up only)" if only_post else "")))
        for offset, (wave, frame, color) in zip(
                (0.14, -0.14), (("pre", pre, SERIES[0]), ("post", post, SERIES[1]))):
            if wave not in instruments.scale(key).waves:
                continue
            mean, half = mean_ci(frame.loc[codes, key])
            if math.isnan(mean):
                continue
            ax.plot([mean - half, mean + half], [y + offset] * 2, color=color,
                    linewidth=2, solid_capstyle="round", zorder=2)
            ax.scatter([mean], [y + offset], s=64, color=color, edgecolors=SURFACE,
                       linewidths=2, zorder=3,
                       label={"pre": "First survey", "post": "Follow-up"}[wave]
                       if row == 0 else None)
    ax.set_yticks([y for y, _ in labels], [label for _, label in labels])
    ax.set_xlim(1, 5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Scale mean (1 = strongly disagree, 5 = strongly agree)",
                  color=INK_2, fontsize=9)
    _style(ax, "Trust and acceptance, before and after using the dashboard",
           f"Means with 95% CIs · n = {len(codes)} answered both surveys")
    # Category rows need no horizontal rules; they would run between the
    # paired dots. Only the value gridlines stay.
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=9, labelcolor=INK_2)
    return _save(plt, figure, path)


def figure_link(results: Results, path: Path) -> Path | None:
    change = results.change.dropna(subset=[f"delta_{PRIMARY_SCALE}", "delta_score"])
    if len(change) < 3:
        return None
    plt, figure, ax = _figure(6.4, 4.4)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.scatter(change[f"delta_{PRIMARY_SCALE}"], change["delta_score"], s=56,
               color=SERIES[0], edgecolors=SURFACE, linewidths=2, alpha=0.9,
               zorder=3)
    row = results.link.set_index("scale").loc[PRIMARY_SCALE]
    ax.set_xlabel("Change in trust (follow-up − first survey, scale points)",
                  color=INK_2, fontsize=9)
    ax.set_ylabel("Change in overall plan score (last − first draft)",
                  color=INK_2, fontsize=9)
    _style(ax, "Change in trust against change in plan quality",
           f"One dot per participant · Spearman ρ = {stats.fmt(row['rho'])}, "
           f"{stats.p_clause(row['p'])}, n = {int(row['n'])}")
    return _save(plt, figure, path)


def figure_trajectories(results: Results, path: Path) -> Path | None:
    plans = results.trajectories
    if plans.empty or plans["sequence"].max() < 2:
        return None
    plt, figure, ax = _figure(6.4, 4.2)
    for index, (_, person) in enumerate(plans.groupby("participant_code")):
        ax.plot(person["sequence"], person["overall_score"], color=MUTED,
                linewidth=1, alpha=0.35, zorder=1,
                label="Each participant" if index == 0 else None)
    by_draft = plans.groupby("sequence")["overall_score"].agg(["mean", "size"])
    by_draft = by_draft[by_draft["size"] >= 3]
    ax.plot(by_draft.index, by_draft["mean"], color=SERIES[0], linewidth=2,
            solid_capstyle="round", zorder=2, label="Mean (drafts with n ≥ 3)")
    ax.scatter(by_draft.index, by_draft["mean"], s=56, color=SERIES[0],
               edgecolors=SURFACE, linewidths=2, zorder=3)
    last = by_draft.index.max()
    ax.annotate(f"{by_draft.loc[last, 'mean']:.1f}", (last, by_draft.loc[last, "mean"]),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=INK, fontsize=9)
    ax.set_xticks(range(1, int(plans["sequence"].max()) + 1))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Draft", color=INK_2, fontsize=9)
    ax.set_ylabel("Overall plan score (0–100)", color=INK_2, fontsize=9)
    _style(ax, "Plan scores across drafts",
           f"{plans['participant_code'].nunique()} participants · "
           f"{len(plans)} scored plans")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2)
    return _save(plt, figure, path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _summary(data: StudyData, results: Results, figures: dict[str, Path]) -> str:
    f, fp, fs = stats.fmt, stats.fmt_p, stats.fmt_signed
    out: list[str] = ["# Trust study — analysis report", ""]
    if data.synthetic:
        out += ["> **SYNTHETIC DATA.** This export was generated by "
                "`python -m analysis synthetic` to exercise the analysis. None of "
                "these numbers describe real participants.", ""]
    if data.include_pilot:
        out += ["> **Includes pilot data** collected before ethics approval. "
                "Not for publication.", ""]
    out += [f"Source: `{data.source}` · instrument version "
            f"{data.instrument_version}", ""]
    for note in data.notes:
        out += [f"- Note: {note}"]
    if data.notes:
        out.append("")

    flow = data.flow
    out += ["## Sample", "",
            _markdown_table(["Step", "n"], [
                ["Participants in the export", str(flow["participants_in_export"])],
                *([["Excluded: consented to unapproved materials",
                    str(flow.get("excluded_pilot_consent", 0))]]
                  if not data.include_pilot else []),
                ["Analysed participants", str(flow["participants_analysed"])],
                ["Answered the first survey", str(flow["pre_completed"])],
                ["Answered the follow-up", str(flow["post_completed"])],
                ["Answered both", str(flow["both_waves"])],
                [("Excluded from pre/post: first survey after AI feedback"
                  if not data.include_late_baseline else
                  "First survey after AI feedback (kept)"),
                 str(flow["late_baseline"])],
                ["**Pre/post comparisons**", f"**{flow['paired_analysed']}**"],
            ]), ""]

    out += ["## Reliability", "",
            "Cronbach's α on reverse-keyed items, complete responses only.", "",
            _markdown_table(["Scale", "Survey", "Items", "n", "α", "M", "SD"], [
                [f"{r.scale} {r.construct}", r.wave, str(r.items), str(r.n),
                 f(r.alpha), f(r.mean), f(r.sd)]
                for r in results.reliability.itertuples()]), ""]

    out += ["## RQ1 — Change between the surveys", "",
            "Paired t-test on scale means, Holm-corrected across the four "
            "scales, with the Wilcoxon signed-rank test as a robustness check. "
            f"{PRIMARY_SCALE} is the primary outcome.", "",
            _markdown_table(
                ["Scale", "n", "Before M (SD)", "After M (SD)", "Δ [95% CI]",
                 "t (df)", "p", "p Holm", "d_z", "Wilcoxon p", "r_rb"],
                [[f"{'**' if r.primary else ''}{r.scale}{'**' if r.primary else ''} "
                  f"{r.construct}", str(r.n),
                  f"{f(r.mean_pre)} ({f(r.sd_pre)})",
                  f"{f(r.mean_post)} ({f(r.sd_post)})",
                  f"{fs(r.mean_diff)} [{fs(r.ci_low)}, {fs(r.ci_high)}]",
                  f"{f(r.t)} ({r.df})", fp(r.p), fp(r.p_holm), f(r.d_z),
                  fp(r.p_wilcoxon), f(r.r_rank_biserial)]
                 for r in results.prepost.itertuples()]), ""]

    if len(results.satisfaction):
        out += ["### Explanation satisfaction (follow-up only)", "",
                _markdown_table(["Scale", "n", "M", "SD", "Median"], [
                    [f"{r.scale} {r.construct}", str(r.n), f(r.mean), f(r.sd),
                     f(r.median)] for r in results.satisfaction.itertuples()]),
                ""]

    q = results.quality_overall.iloc[0]
    out += ["## RQ2 — Plan quality across drafts", "",
            "First vs last scored plan of each participant with at least two.", "",
            _markdown_table(
                ["Measure", "n", "First M (SD)", "Last M (SD)", "Δ [95% CI]",
                 "t (df)", "p", "d_z", "Wilcoxon p"],
                [[q["measure"], str(q["n"]), f"{f(q['mean_pre'], 1)} ({f(q['sd_pre'], 1)})",
                  f"{f(q['mean_post'], 1)} ({f(q['sd_post'], 1)})",
                  f"{fs(q['mean_diff'], 1)} [{fs(q['ci_low'], 1)}, {fs(q['ci_high'], 1)}]",
                  f"{f(q['t'])} ({q['df']})", fp(q["p"]), f(q["d_z"]),
                  fp(q["p_wilcoxon"])]]), ""]
    if len(results.quality_criteria):
        out += ["Per criterion (rubric scores 1–4), Holm-corrected across the ten:", "",
                _markdown_table(["Criterion", "n", "First M", "Last M", "Δ",
                                 "p Holm", "d_z"], [
                    [r.criterion.replace("_", " "), str(r.n), f(r.mean_pre),
                     f(r.mean_post), fs(r.mean_diff), fp(r.p_holm), f(r.d_z)]
                    for r in results.quality_criteria.itertuples()]), ""]

    out += ["## RQ3 — Trust change and quality change", "",
            "Spearman's ρ between each scale's change and the change in overall "
            "plan score, Holm-corrected across the four.", "",
            _markdown_table(["Scale", "n", "ρ", "p", "p Holm"], [
                [f"{'**' if r.primary else ''}{r.scale}{'**' if r.primary else ''}",
                 str(r.n), f(r.rho), fp(r.p), fp(r.p_holm)]
                for r in results.link.itertuples()]), ""]

    if len(results.background):
        out += ["## Background (first survey)", "",
                _markdown_table(["Question", "Answer", "n", "%"], [
                    [r.item, r.answer, str(r.n), f(r.percent, 0)]
                    for r in results.background.itertuples()]), ""]

    if figures:
        out += ["## Figures", ""]
        for name, path in figures.items():
            out += [f"![{name}](figures/{path.name})", ""]

    out += ["## Methods notes", "",
            "- Scale scores are item means after reverse-keying (TR6). A scale "
            "score is missing unless every item was answered.",
            "- Pilot data (collected before ethics approval) is excluded unless "
            "`--include-pilot` is given.",
            "- A first survey answered after the participant had already seen AI "
            "feedback is not a baseline and is left out of the pre/post and "
            "change analyses unless `--include-late-baseline` is given.",
            "- The dashboard's scores come from models trained on synthetic "
            "plans; RQ2 measures change on the dashboard's own scale.",
            ""]
    return "\n".join(out)


def write(data: StudyData, results: Results, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in ("reliability", "prepost", "satisfaction", "quality_overall",
                 "quality_criteria", "link", "background", "change"):
        path = tables_dir / f"{name}.csv"
        getattr(results, name).to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)

    figures: dict[str, Path] = {}
    for slug, name, draw in (
            ("prepost", "Trust and acceptance before and after",
             lambda path: figure_prepost(data, results, path)),
            ("trust_vs_quality", "Trust change against quality change",
             lambda path: figure_link(results, path)),
            ("score_trajectories", "Plan scores across drafts",
             lambda path: figure_trajectories(results, path))):
        drawn = draw(figures_dir / f"{slug}.png")
        if drawn is not None:
            figures[name] = drawn
            written.append(drawn)
    results.figures = list(figures.values())

    summary = out_dir / "summary.md"
    summary.write_text(_summary(data, results, figures), encoding="utf-8")
    written.append(summary)
    return written
