"""Smoke test: segment + featurise one synthetic plan. Run from model/."""
import json
import time

from ingestion.feature_engineer import FeatureEngineer
from scoring.train_xgboost import plan_from_text

text = open("data/synthetic_v1/plans/synth_0000.txt", encoding="utf-8").read()
plan = plan_from_text(text, "synth_0000")
print("sections found:", sorted(k for k, v in plan.sections.items() if v))
print("missing:", plan.missing_sections)
print("stated duration:", plan.stated_duration_minutes)

t0 = time.perf_counter()
engineer = FeatureEngineer()
print(f"spaCy load: {time.perf_counter() - t0:.2f}s")

t0 = time.perf_counter()
features = engineer.extract_features(plan)
print(f"feature extraction: {time.perf_counter() - t0:.2f}s")
print(json.dumps({k: round(v, 3) for k, v in features.items()}, indent=1))
