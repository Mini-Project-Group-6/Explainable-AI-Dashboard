"""Train the DistilBERT text layer for rubric scoring (LoRA, CPU-only).

The structural features in ``ingestion/feature_engineer.py`` cover six of the
ten rubric criteria well and four barely at all (see
``explainability/feature_labels.coverage_report``): resources/ICT, the RPK
introduction, attention to all learners, and closure are things a plan *says*
rather than things its shape reveals. This model reads the prose so those four
criteria have an evidence channel at all.

Architecture
    distilbert-base-uncased
      + LoRA adapters on the attention query/value projections (r=8)
      + a 10-output regression head, one output per criterion in
        ``model_contract.CRITERION_KEYS`` order, targets on the raw 1-4 scale.

Only the adapters and the head train — ~1% of parameters — which is what makes
a 66M-parameter encoder fine-tunable on the deployment laptop with no GPU.

A plain PyTorch loop is used rather than ``transformers.Trainer`` on purpose:
the Trainer's argument names churn between releases (``evaluation_strategy`` ->
``eval_strategy``), and this loop is short enough to read and defend in the
methods section.

Run from the ``model/`` directory:
    python -m scoring.train_bert_lora --labels data/synthetic_v1/labels.csv \\
        --plans data/synthetic_v1/plans --out artifacts/bert_rubric
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from model_contract import (
    CONTRACT_VERSION,
    CRITERION_KEYS,
    SCORE_MAX,
    SCORE_MIN,
    TEXT_MAX_TOKENS,
    ArtifactContractError,
)

logger = logging.getLogger(__name__)

BASE_MODEL: str = "distilbert-base-uncased"

#: LoRA config. r=8 on q/v is the standard low-budget setting; the classifier
#: head is saved in full because it is randomly initialised and must train.
LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.10,
    target_modules=["q_lin", "v_lin"],
    modules_to_save=["pre_classifier", "classifier"],
    bias="none",
)

TRAIN_CONFIG = dict(
    # 20, not 8. Measured: at 8 the model was still improving and reached val
    # MSE 0.5453; allowed to run it early-stopped at 19 (best epoch 16) with
    # 0.4489, and held-out mean QWK rose 0.636 -> 0.712.
    #
    # Note this is a *schedule* parameter, not just a stopping point:
    # get_linear_schedule_with_warmup spreads the LR decay over
    # steps x epochs, so raising it changes the whole trajectory rather than
    # appending epochs to the same run. Early stopping decides where to stop.
    epochs=20,
    batch_size=8,
    learning_rate=5e-4,      # LoRA tolerates a much higher LR than full FT
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_length=TEXT_MAX_TOKENS,
    patience=3,              # epochs without val improvement before stopping
    seed=42,
)

#: Written next to the adapter so the explainer and loader can validate that a
#: checkpoint was trained against this contract.
MANIFEST_NAME = "text_bundle.json"

#: Held-out agreement per criterion, written beside the checkpoint and used by
#: the reconciler to weight the text channel. See write_holdout_report.
HOLDOUT_REPORT_NAME = "holdout_report.json"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class TextDataset:
    """Plan texts with their 10 criterion scores, in contract column order."""

    ids: list[str]
    texts: list[str]
    labels: list[list[float]]

    def __len__(self) -> int:
        return len(self.ids)


def load_text_dataset(labels_csv: str | Path, plans_dir: str | Path) -> TextDataset:
    """Read the same labels.csv manifest the XGBoost scorer trains on.

    Keeping one manifest for both models is what lets the two channels be
    reconciled per criterion later — they are fit to identical targets.
    """
    labels_csv, plans_dir = Path(labels_csv), Path(plans_dir)
    ids, texts, labels = [], [], []

    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        absent = [k for k in CRITERION_KEYS if k not in (reader.fieldnames or [])]
        if absent:
            raise ArtifactContractError(
                f"{labels_csv}: manifest is missing criterion columns {absent}; "
                f"expected all of {CRITERION_KEYS}")
        for record in reader:
            plan_id = record["plan_id"]
            ids.append(plan_id)
            texts.append((plans_dir / f"{plan_id}.txt").read_text(encoding="utf-8"))
            labels.append([float(record[k]) for k in CRITERION_KEYS])

    return TextDataset(ids=ids, texts=texts, labels=labels)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(base_model: str = BASE_MODEL):
    """DistilBERT + LoRA adapters + a 10-output regression head."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CRITERION_KEYS),
        problem_type="regression",   # -> MSE over the 10 outputs
    )
    peft_model = get_peft_model(
        model, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CONFIG))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info("Trainable parameters: %d / %d (%.2f%%)",
                trainable, total, 100 * trainable / total)
    return peft_model


def _encode(tokenizer, texts: Sequence[str], max_length: int,
            pad_to_max: bool = True):
    """Tokenise a batch.

    Training pads to ``max_length`` so every batch is the same shape. Inference
    pads only to the longest item in the batch: attention masking makes the two
    numerically equivalent, and a short plan then costs a fraction of a full
    512-token forward pass, which matters against the <=15s end-to-end budget
    (proposal E3 criterion 5) and against the SHAP explainer, which re-runs the
    model hundreds of times per plan.
    """
    return tokenizer(list(texts), truncation=True, max_length=max_length,
                     padding="max_length" if pad_to_max else True,
                     return_tensors="pt")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(dataset: TextDataset, base_model: str = BASE_MODEL,
          validation_fraction: float = 0.2, config: Optional[dict] = None):
    """Fit the text scorer. Returns (peft_model, tokenizer, history)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from scoring.train_xgboost import validation_mask

    cfg = {**TRAIN_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    device = torch.device("cpu")   # proposal D5: CPU-only deployment

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoded = _encode(tokenizer, dataset.texts, cfg["max_length"])
    y = torch.tensor(dataset.labels, dtype=torch.float32)

    # The identical split the tabular scorer uses — see validation_mask.
    val_mask = validation_mask(len(dataset), validation_fraction, cfg["seed"])
    train_idx = torch.tensor(np.where(~val_mask)[0])
    val_idx = torch.tensor(np.where(val_mask)[0])

    def subset(idx):
        return TensorDataset(encoded["input_ids"][idx],
                             encoded["attention_mask"][idx], y[idx])

    train_loader = DataLoader(subset(train_idx), batch_size=cfg["batch_size"],
                              shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=cfg["batch_size"])

    model = build_model(base_model).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    total_steps = max(1, len(train_loader) * cfg["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_ratio"] * total_steps), total_steps)
    loss_fn = torch.nn.MSELoss()

    history, best_val, best_state, stale = [], float("inf"), None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for input_ids, attention_mask, targets in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(targets)
        train_loss /= max(1, len(train_idx))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, targets in val_loader:
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask).logits
                val_loss += loss_fn(logits, targets).item() * len(targets)
        val_loss /= max(1, len(val_idx))

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        logger.info("epoch %d/%d  train MSE %.4f  val MSE %.4f",
                    epoch, cfg["epochs"], train_loss, val_loss)

        if val_loss < best_val - 1e-4:
            best_val, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                logger.info("Early stopping at epoch %d (best val MSE %.4f)",
                            epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, {"history": history, "best_val_mse": best_val,
                              "n_train": int(len(train_idx)),
                              "n_val": int(len(val_idx)),
                              "val_indices": [int(i) for i in val_idx],
                              "config": cfg}


def holdout_indices(n: int, validation_fraction: float = 0.2,
                    seed: int = TRAIN_CONFIG["seed"]) -> list[int]:
    """Recreate the validation split without retraining.

    Same deterministic scheme as ``train``; kept in one place so an evaluation
    run cannot accidentally score the model on rows it was fitted to.
    """
    import numpy as np

    from scoring.train_xgboost import validation_mask

    # Deliberately the *same* function the tabular scorer uses, so the two
    # channels are validated on identical plans and their predictions can be
    # blended on a common held-out set (evaluation/blend_weights.py).
    return [int(i) for i in np.where(validation_mask(n, validation_fraction, seed))[0]]


def evaluate_holdout(scorer: "TextRubricScorer", dataset: TextDataset,
                     validation_fraction: float = 0.2,
                     seed: int = TRAIN_CONFIG["seed"]):
    """Per-criterion QWK on the held-out split only.

    In-sample QWK over the whole corpus flatters a model that has seen 80% of
    it and says nothing about generalisation — the one number worth quoting is
    this one.
    """
    import numpy as np
    import pandas as pd

    from evaluation.metrics import full_report

    val_idx = holdout_indices(len(dataset), validation_fraction, seed)
    texts = [dataset.texts[i] for i in val_idx]
    truth = [dataset.labels[i] for i in val_idx]

    preds = scorer.predict(texts)
    Y_true = pd.DataFrame(truth, columns=list(CRITERION_KEYS))
    Y_pred = pd.DataFrame(np.clip(np.rint(preds), SCORE_MIN, SCORE_MAX),
                          columns=list(CRITERION_KEYS))
    return full_report(Y_true, Y_pred), len(val_idx)


def write_holdout_report(model_dir: str | Path, report, n_val: int) -> Path:
    """Persist held-out agreement next to the model.

    ``explainability/reconcile.py`` weights the text channel per criterion. Left
    to itself it falls back to a flat guess (``TEXT_CHANNEL_PRIOR``); this file
    replaces the guess with measured agreement, and travels with the checkpoint
    so the weights can never drift from the model they describe. Regenerated by
    ``--eval-only``.
    """
    model_dir = Path(model_dir)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "n_holdout": n_val,
        "metric": "quadratic_weighted_kappa",
        "note": ("Measured on synthetic plans. Re-run against the real annotated "
                 "CoE plans before quoting or relying on these weights."),
        "per_criterion": {
            key: {"qwk": round(float(report.loc[key, "qwk"]), 4),
                  "rmse": round(float(report.loc[key, "rmse"]), 4)}
            for key in CRITERION_KEYS
        },
        "mean_qwk": round(float(report.loc["MEAN", "qwk"]), 4),
    }
    path = model_dir / HOLDOUT_REPORT_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save(model, tokenizer, out_dir: str | Path, base_model: str,
         training_summary: dict) -> Path:
    """Persist the adapter, tokenizer and the contract manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)          # LoRA adapter + saved head modules
    tokenizer.save_pretrained(out_dir)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "base_model": base_model,
        "criterion_keys": list(CRITERION_KEYS),
        "score_range": [SCORE_MIN, SCORE_MAX],
        "max_length": training_summary["config"]["max_length"],
        "lora": {k: v for k, v in LORA_CONFIG.items()},
        "training": {k: v for k, v in training_summary.items() if k != "config"},
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                         encoding="utf-8")
    return out_dir


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

class TextRubricScorer:
    """Loaded text scorer: raw plan text -> 10 criterion scores in [1, 4].

    This is the callable that ``explainability/shap_deep.py`` explains and that
    ``explainability/reconcile.py`` treats as the text channel.
    """

    def __init__(self, model_dir: str | Path):
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.manifest = self._load_manifest(self.model_dir)
        self.criterion_keys = tuple(self.manifest["criterion_keys"])
        self.max_length = int(self.manifest["max_length"])
        self.reliability = self._load_reliability(self.model_dir)

        base = self.manifest.get("base_model") or PeftConfig.from_pretrained(
            self.model_dir).base_model_name_or_path
        backbone = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=len(self.criterion_keys), problem_type="regression")
        self.model = PeftModel.from_pretrained(backbone, self.model_dir)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._torch = torch

    @staticmethod
    def _load_reliability(model_dir: Path) -> dict[str, float]:
        """criterion -> held-out QWK, or {} if it has not been evaluated yet.

        Negative kappa means worse than chance; those are floored at 0 so a
        criterion the model cannot do is given no weight rather than negative
        weight.
        """
        path = model_dir / HOLDOUT_REPORT_NAME
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {key: max(0.0, float(entry["qwk"]))
                for key, entry in payload.get("per_criterion", {}).items()}

    @staticmethod
    def _load_manifest(model_dir: Path) -> dict:
        path = model_dir / MANIFEST_NAME
        if not path.is_file():
            raise ArtifactContractError(
                f"{model_dir} has no {MANIFEST_NAME}; it was not written by "
                f"scoring/train_bert_lora.save() and cannot be validated.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        actual = tuple(manifest.get("criterion_keys", ()))
        if actual != CRITERION_KEYS:
            raise ArtifactContractError(
                f"{path}: text model criterion order {actual} does not match "
                f"contract {CONTRACT_VERSION} order {CRITERION_KEYS}. Its "
                f"output columns would be misread. Retrain.")
        return manifest

    def predict(self, texts: Sequence[str], batch_size: int = 8):
        """(n_texts, 10) float32 scores, clipped to the rubric's [1, 4]."""
        import numpy as np

        torch = self._torch
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = _encode(self.tokenizer, batch, self.max_length,
                              pad_to_max=False)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            outputs.append(logits.cpu().numpy())
        scores = np.concatenate(outputs, axis=0) if outputs else np.zeros(
            (0, len(self.criterion_keys)), dtype="float32")
        return np.clip(scores, SCORE_MIN, SCORE_MAX).astype("float32")

    def predict_one(self, text: str) -> dict[str, float]:
        """criterion key -> score, for a single plan."""
        row = self.predict([text])[0]
        return {key: float(row[i]) for i, key in enumerate(self.criterion_keys)}


def load_text_scorer(model_dir: str | Path | None = None) -> TextRubricScorer:
    """Load the text scorer, defaulting to the contract's artifact path."""
    from model_contract import TEXT_MODEL_DIR

    model_dir = Path(model_dir) if model_dir else TEXT_MODEL_DIR
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"No text model at {model_dir}. Train it with:\n"
            f"    python -m scoring.train_bert_lora --labels data/synthetic_v1/labels.csv "
            f"--plans data/synthetic_v1/plans --out {model_dir}")
    return TextRubricScorer(model_dir)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT with LoRA for rubric scoring (CPU)")
    parser.add_argument("--labels", required=True, help="labels.csv manifest")
    parser.add_argument("--plans", required=True, help="directory of <plan_id>.txt files")
    parser.add_argument("--out", default="artifacts/bert_rubric")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; evaluate the saved model in --out")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dataset = load_text_dataset(args.labels, args.plans)
    logger.info("Loaded %d plans", len(dataset))

    if args.eval_only:
        out_dir = Path(args.out)
    else:
        model, tokenizer, summary = train(
            dataset, base_model=args.base_model,
            config={"epochs": args.epochs, "batch_size": args.batch_size})
        out_dir = save(model, tokenizer, args.out, args.base_model, summary)
        logger.info("Saved text model -> %s (best val MSE %.4f)",
                    out_dir, summary["best_val_mse"])

    # Held-out only. Scoring the training rows would flatter the model and tell
    # us nothing about generalisation.
    report, n_val = evaluate_holdout(TextRubricScorer(out_dir), dataset)
    logger.info("Held-out evaluation on %d plans:", n_val)
    for key in list(CRITERION_KEYS) + ["MEAN"]:
        logger.info("  QWK %-32s %.3f  (RMSE %.3f)",
                    key, report.loc[key, "qwk"], report.loc[key, "rmse"])
    logger.info("Wrote channel weights -> %s",
                write_holdout_report(out_dir, report, n_val))


if __name__ == "__main__":
    main()
