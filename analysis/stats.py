# analysis/stats.py
"""The statistics, as plain functions over arrays. S5 owns this file.

Kept free of pandas and plotting so every number in the report can be checked
against a hand calculation in the tests.

Choices, fixed before seeing any data:

* **Pre/post** — paired t-test on scale means (Likert scale means are treated
  as interval, the usual practice for multi-item TAM scales), with the
  Wilcoxon signed-rank test reported beside it as the robustness check.
  Effect sizes: Cohen's d_z (mean difference / SD of differences) and the
  matched-pairs rank-biserial correlation.
* **Multiplicity** — Holm's step-down correction within each family of tests
  (the four TAM/trust scales; the ten rubric criteria).
* **Association** — Spearman's rho, because change scores on short scales are
  coarse and rarely normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps

NAN = float("nan")


def cronbach_alpha(items: np.ndarray) -> float:
    """Cronbach's alpha for a respondents × items matrix (complete rows only).

    ``NaN`` when it is undefined: fewer than two items, fewer than two
    complete respondents, or no variance in the total score.
    """
    matrix = np.asarray(items, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return NAN
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if matrix.shape[0] < 2:
        return NAN
    k = matrix.shape[1]
    total_variance = matrix.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return NAN
    item_variance = matrix.var(axis=0, ddof=1).sum()
    return float(k / (k - 1) * (1 - item_variance / total_variance))


def rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014).

    (sum of ranks of positive differences − sum of ranks of negative ones) /
    total rank sum, with zero differences dropped, as the Wilcoxon test drops
    them. +1 means every non-zero change was an increase.
    """
    diff = np.asarray(differences, dtype=float)
    diff = diff[(diff != 0) & ~np.isnan(diff)]
    if diff.size == 0:
        return NAN
    ranks = sps.rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


@dataclass(frozen=True)
class Paired:
    n: int
    mean_pre: float
    sd_pre: float
    mean_post: float
    sd_post: float
    mean_diff: float
    ci_low: float
    ci_high: float
    t: float
    df: int
    p: float
    d_z: float
    p_wilcoxon: float
    r_rank_biserial: float

    def as_row(self) -> dict:
        return dict(self.__dict__)


def paired(pre: Sequence[float], post: Sequence[float]) -> Paired:
    """Paired comparison over the complete pairs of *pre* and *post*."""
    a = np.asarray(pre, dtype=float)
    b = np.asarray(post, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = int(a.size)
    diff = b - a

    def sd(x):
        return float(x.std(ddof=1)) if x.size > 1 else NAN

    mean_diff = float(diff.mean()) if n else NAN
    sd_diff = sd(diff)
    t = p = d_z = ci_low = ci_high = NAN
    if n > 1 and sd_diff > 0:
        t_result = sps.ttest_rel(b, a)
        t, p = float(t_result.statistic), float(t_result.pvalue)
        d_z = mean_diff / sd_diff
        half = sps.t.ppf(0.975, n - 1) * sd_diff / math.sqrt(n)
        ci_low, ci_high = mean_diff - half, mean_diff + half
    elif n > 1:
        # Every participant changed by exactly the same amount: the interval
        # is that amount and the test statistic is undefined.
        ci_low = ci_high = mean_diff

    p_w = NAN
    if np.count_nonzero(diff) > 0:
        try:
            p_w = float(sps.wilcoxon(diff, zero_method="wilcox").pvalue)
        except ValueError:
            p_w = NAN

    return Paired(
        n=n,
        mean_pre=float(a.mean()) if n else NAN, sd_pre=sd(a),
        mean_post=float(b.mean()) if n else NAN, sd_post=sd(b),
        mean_diff=mean_diff, ci_low=ci_low, ci_high=ci_high,
        t=t, df=max(n - 1, 0), p=p, d_z=d_z,
        p_wilcoxon=p_w, r_rank_biserial=rank_biserial(diff),
    )


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order. NaNs are left out of the family."""
    values = [float(p) for p in pvalues]
    present = [(p, i) for i, p in enumerate(values) if not math.isnan(p)]
    m = len(present)
    adjusted = [NAN] * len(values)
    running = 0.0
    for rank, (p, index) in enumerate(sorted(present)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


@dataclass(frozen=True)
class Correlation:
    n: int
    rho: float
    p: float


def spearman(x: Sequence[float], y: Sequence[float]) -> Correlation:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return Correlation(int(a.size), NAN, NAN)
    result = sps.spearmanr(a, b)
    return Correlation(int(a.size), float(result.statistic), float(result.pvalue))


# ---------------------------------------------------------------------------
# APA-style formatting
# ---------------------------------------------------------------------------

def fmt_p(p: Optional[float]) -> str:
    """"< .001", ".043", "1.000"; "—" when undefined."""
    if p is None or math.isnan(p):
        return "—"
    if p < 0.001:
        return "< .001"
    text = f"{p:.3f}"
    return text[1:] if text.startswith("0") else text


def p_clause(p: Optional[float]) -> str:
    """"p < .001" or "p = .043" — for running text, where "p = < .001" reads wrong."""
    text = fmt_p(p)
    return f"p {text}" if text.startswith("<") else f"p = {text}"


def fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def fmt_signed(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.{digits}f}"
