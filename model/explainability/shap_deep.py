"""SHAP attributions for the DistilBERT text layer.

Produces the text half of the reconciled explanation: which spans of a lesson
plan's prose pushed each rubric criterion up or down. Output is a list of
``reconcile.SpanAttribution`` per criterion, carrying the quoted wording, its
character offsets in the plan, and the section it came from — so the dashboard
can highlight the actual sentence a tutor should look at.

Why the Partition explainer and not DeepExplainer
-------------------------------------------------
The proposal names SHAP generally; the file name is historical. ``shap.Deep-
Explainer`` implements DeepLIFT-style attribution against a framework graph and
does not support a peft/LoRA-wrapped HuggingFace model — and where it can be
forced to run, its rescale rule on a transformer's attention and LayerNorm
blocks has no correctness guarantee. SHAP's Partition explainer with a
``Text`` masker is the supported and standard path for transformer models: it
computes Owen values over a hierarchical clustering of the text, which are
proper Shapley values under that grouping structure. This is a defensible
choice to write up, not a workaround.

Cost
----
Every coalition is a forward pass. On CPU, DistilBERT at 512 tokens is roughly
50-80ms, so ``max_evals`` is the latency dial: the default 300 puts a single
plan around 15-25s, which is over the interactive budget (proposal E3 criterion
5, <=15s end to end). Explanations are therefore meant to be computed once per
submission and cached by S3, not recomputed per page view. ``granularity=
"section"`` is the fast path — under ten segments, a couple of seconds — when a
coarse "which part of the plan" answer is enough.

Run from the ``model/`` directory:
    python -m explainability.shap_deep --text-model artifacts/bert_rubric \\
        --plan data/synthetic_v1/plans/synth_0000.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from model_contract import CRITERION_KEYS, TEXT_MAX_TOKENS, criterion
from explainability.reconcile import SpanAttribution
from ingestion.extract_text import section_at, section_spans

logger = logging.getLogger(__name__)

#: Split patterns fed to ``shap.maskers.Text``. The masker treats each match as
#: a separator and attributes the text between separators.
GRANULARITY_PATTERNS: dict[str, str] = {
    # Lesson plans are line-structured (one bullet or stage per line), so a
    # line is usually also the unit a tutor would highlight.
    "sentence": r"(?<=[.?!])\s+|\n+",
    "line": r"\n+",
    "token": r"\W+",
}

DEFAULT_GRANULARITY = "sentence"
DEFAULT_MAX_EVALS = 300


class TextRubricExplainer:
    """SHAP Partition explainer over the DistilBERT rubric scorer."""

    def __init__(self, scorer=None, *, model_dir: str | Path | None = None,
                 granularity: str = DEFAULT_GRANULARITY,
                 max_evals: int = DEFAULT_MAX_EVALS,
                 fixed_context: Optional[int] = 1):
        """
        Args:
            scorer: a loaded ``TextRubricScorer``; loaded from *model_dir* if omitted.
            granularity: one of ``GRANULARITY_PATTERNS``.
            max_evals: coalition budget — the latency dial (see module docstring).
            fixed_context: passed to the Partition explainer. 1 keeps one side
                of each split fixed, roughly halving evaluations at a small cost
                in attribution sharpness; None is the exact-ish setting.
        """
        import shap

        if granularity not in GRANULARITY_PATTERNS:
            raise ValueError(
                f"granularity must be one of {sorted(GRANULARITY_PATTERNS)}, "
                f"got {granularity!r}")

        if scorer is None:
            from scoring.train_bert_lora import load_text_scorer
            scorer = load_text_scorer(model_dir)

        self.scorer = scorer
        self.granularity = granularity
        self.max_evals = max_evals
        self.fixed_context = fixed_context
        self.criterion_keys = tuple(scorer.criterion_keys)
        # Single-entry, keyed by plan text: the dashboard explains one
        # submission at a time, and holding more than one plan's spans would
        # grow without bound in a long-lived Streamlit session.
        self._cache: dict[str, dict[str, list[SpanAttribution]]] = {}

        self._masker = shap.maskers.Text(GRANULARITY_PATTERNS[granularity])
        self._explainer = shap.Explainer(
            self._predict,
            self._masker,
            output_names=[criterion(k).label for k in self.criterion_keys],
        )

    # -- model wrapper ------------------------------------------------------

    def _predict(self, texts):
        """(n_texts,) array of strings -> (n_texts, 10) scores, for SHAP."""
        return self.scorer.predict([str(t) for t in texts])

    # -- explanation --------------------------------------------------------

    def explain(self, texts: Sequence[str]):
        """Raw ``shap.Explanation``; ``.values`` is (n_texts, n_spans, 10)."""
        return self._explainer(list(texts), max_evals=self.max_evals,
                               fixed_context=self.fixed_context, silent=True)

    def explain_one(self, text: str,
                    min_abs_value: float = 0.0
                    ) -> dict[str, list[SpanAttribution]]:
        """criterion key -> attributed spans, ranked by absolute value.

        One explainer run produces all ten criteria at once, and the result is
        cached against *text*: a caller iterating criteria (see ``top_spans``)
        would otherwise pay ~93s ten times over for data a single run already
        computed.

        Args:
            min_abs_value: drop spans whose attribution is below this, to keep
                the panel readable. 0 keeps everything.
        """
        cached = self._cache.get(text)
        if cached is not None and min_abs_value == 0.0:
            return cached

        explanation = self.explain([text])[0]
        segments = [str(s) for s in explanation.data]
        offsets = _locate(text, segments)
        sections = section_spans(text)

        by_criterion: dict[str, list[SpanAttribution]] = {}
        for column, key in enumerate(self.criterion_keys):
            spans = []
            for index, segment in enumerate(segments):
                value = float(explanation.values[index][column])
                if abs(value) < min_abs_value or not segment.strip():
                    continue
                start, end = offsets[index]
                spans.append(SpanAttribution(
                    text=segment,
                    value=value,
                    start=start,
                    end=end,
                    section=section_at(sections, start) if start >= 0 else "",
                ))
            spans.sort(key=lambda s: abs(s.value), reverse=True)
            by_criterion[key] = spans

        if min_abs_value == 0.0:
            # Only the unfiltered result is cacheable; a filtered one is not a
            # substitute for a later call with a lower threshold.
            self._cache = {text: by_criterion}
        return by_criterion

    def top_spans(self, text: str, criterion_key: str, k: int = 5
                  ) -> list[SpanAttribution]:
        """The k most influential spans for one criterion.

        Cheap after the first call for a given text — see ``explain_one``.
        """
        return self.explain_one(text)[criterion_key][:k]

    def token_count(self, text: str) -> int:
        """Tokens the model would see before truncation — feeds the reconciler's
        'only the first N tokens were read' caveat."""
        return len(self.scorer.tokenizer(text, truncation=False)["input_ids"])

    def exceeds_context(self, text: str) -> bool:
        return self.token_count(text) > TEXT_MAX_TOKENS


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _locate(text: str, segments: Sequence[str]) -> list[tuple[int, int]]:
    """Character offsets of each segment, matched left to right.

    The Text masker returns segments with separators attached and does not
    hand back an offset mapping for regex maskers, so they are recovered by a
    forward scan. A segment that cannot be found (whitespace-only, or mangled)
    gets (-1, -1) and is reported without a location rather than mislocated.
    """
    offsets, cursor = [], 0
    for segment in segments:
        stripped = segment.strip()
        if not stripped:
            offsets.append((-1, -1))
            continue
        found = text.find(stripped, cursor)
        if found < 0:
            offsets.append((-1, -1))
            continue
        offsets.append((found, found + len(stripped)))
        cursor = found + len(stripped)
    return offsets


def aggregate_by_section(spans: Sequence[SpanAttribution]) -> dict[str, float]:
    """Sum span attributions per plan section — the coarse 'where' view."""
    totals: dict[str, float] = {}
    for span in spans:
        totals[span.section or "unlocated"] = (
            totals.get(span.section or "unlocated", 0.0) + span.value)
    return dict(sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True))


def save_text_plot(explanation, out_path: str | Path) -> Path:
    """SHAP's interactive text plot as standalone HTML (transparency tab)."""
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = shap.plots.text(explanation, display=False)
    out_path.write_text(
        f"<!doctype html><meta charset='utf-8'>{shap.getjs()}{html}",
        encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP text attributions for the DistilBERT rubric scorer")
    parser.add_argument("--plan", required=True, help="path to a .txt lesson plan")
    parser.add_argument("--text-model", default=None,
                        help="text model directory (default: contract path)")
    parser.add_argument("--granularity", default=DEFAULT_GRANULARITY,
                        choices=sorted(GRANULARITY_PATTERNS))
    parser.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    parser.add_argument("--criterion", default=None,
                        help=f"one of {', '.join(CRITERION_KEYS)} (default: all)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--html", default=None, help="write the SHAP text plot here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    text = Path(args.plan).read_text(encoding="utf-8")
    explainer = TextRubricExplainer(model_dir=args.text_model,
                                    granularity=args.granularity,
                                    max_evals=args.max_evals)
    if explainer.exceeds_context(text):
        logger.warning("Plan is %d tokens; only the first %d are read.",
                       explainer.token_count(text), TEXT_MAX_TOKENS)

    attributions = explainer.explain_one(text)
    keys = [args.criterion] if args.criterion else list(CRITERION_KEYS)
    for key in keys:
        target = criterion(key)
        print(f"\n{target.id} {target.label}  (NTS {target.nts_indicator})")
        print("-" * 78)
        for span in attributions[key][:args.top_k]:
            arrow = "+" if span.value > 0 else "-"
            where = f"[{span.section or 'unlocated'}]"
            print(f" {arrow} {abs(span.value):.4f} {where:16} {span.quote}")
        print(f"   by section: {aggregate_by_section(attributions[key])}")

    if args.html:
        save_text_plot(explainer.explain([text]), args.html)
        print(f"\nWrote {args.html}")


if __name__ == "__main__":
    main()
