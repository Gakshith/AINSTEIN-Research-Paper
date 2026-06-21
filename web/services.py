"""Pure functions that read existing pipeline artifacts into JSON-friendly dicts.

Every path comes from ``EvaluationConfig`` via ``ConfigurationManager`` so the web
layer never duplicates artifact locations. All readers degrade gracefully (return
empty/None) when an artifact does not exist yet, so a fresh clone never crashes.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.AINSTEIN.config.configuration import ConfigurationManager
from src.AINSTEIN.entity.config_entity import EvaluationConfig

# Bundled demo data, surfaced only when AINSTEIN_SAMPLE is truthy and no real
# artifacts exist. Lets the dashboard look populated out-of-the-box.
SAMPLE_DIR = Path(__file__).resolve().parent / "sample" / "evaluation"


def _sample_enabled() -> bool:
    return os.environ.get("AINSTEIN_SAMPLE", "").lower() in ("1", "true", "yes", "on")


def _resolve(real_path: Path, sample_name: str) -> Path:
    """Prefer the real artifact; fall back to bundled sample when enabled."""
    real = Path(real_path)
    if real.exists() and real.stat().st_size > 0:
        return real
    if _sample_enabled():
        sample = SAMPLE_DIR / sample_name
        if sample.exists():
            return sample
    return real

SUMMARY_METRICS = [
    "success_rate_relaxed",
    "success_rate_strict",
    "rediscovery_relaxed",
    "rediscovery_strict",
    "novel_and_valid_relaxed",
    "novel_and_valid_strict",
    "judge_agreement",
    "token_jaccard",
    "token_f1",
    "keyword_overlap",
    "length_ratio",
    "sequence_similarity",
]


@lru_cache(maxsize=1)
def eval_config() -> EvaluationConfig:
    """Load the evaluation config once (cached). Paths only — cheap to build."""
    return ConfigurationManager().get_evaluation_config()


def _clean(value: Any) -> Any:
    """Make a value JSON-safe: NaN/NaT -> None, numpy scalars -> python scalars."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _read_csv(path: Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, OSError):
        return None
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def artifacts_available() -> bool:
    """True if the overall summary exists (from a real run or bundled sample)."""
    df = _read_csv(_resolve(eval_config().summary_csv, "evaluation_summary.csv"))
    return df is not None and not df.empty


def get_summary() -> dict | None:
    df = _read_csv(_resolve(eval_config().summary_csv, "evaluation_summary.csv"))
    if df is None or df.empty:
        return None
    return {k: _clean(v) for k, v in df.iloc[0].items()}


def get_tiers() -> list[dict]:
    df = _read_csv(_resolve(eval_config().tier_summary_csv, "evaluation_tier_summary.csv"))
    if df is None or df.empty:
        return []
    return _records(df)


def get_baselines() -> list[dict]:
    df = _read_csv(_resolve(eval_config().baseline_summary_csv, "baseline_summary.csv"))
    if df is None or df.empty or "method_name" not in df.columns:
        return []
    return _records(df)


def _main_results() -> pd.DataFrame | None:
    df = _read_csv(_resolve(eval_config().results_csv, "evaluation_results.csv"))
    if df is None or df.empty:
        return None
    if "method_name" in df.columns:
        df = df[df["method_name"] == "main_model"].copy()
    return df


def get_papers(tier: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    df = _main_results()
    if df is None or df.empty:
        return {"total": 0, "items": []}
    if tier and "tier" in df.columns:
        df = df[df["tier"].astype(str) == tier]
    total = len(df)
    cols = [c for c in ["paper_id", "title", "tier", "row_index", "success_rate_relaxed",
                        "success_rate_strict", "rediscovery_relaxed", "novel_and_valid_relaxed",
                        "judge_agreement", "token_f1", "error"] if c in df.columns]
    page = df[cols].iloc[offset: offset + limit]
    return {"total": int(total), "items": _records(page)}


def get_paper(paper_id: str) -> dict | None:
    df = _main_results()
    if df is None or df.empty or "paper_id" not in df.columns:
        return None
    match = df[df["paper_id"].astype(str) == str(paper_id)]
    if match.empty:
        return None
    return {k: _clean(v) for k, v in match.iloc[0].items()}


def get_report() -> str | None:
    p = _resolve(eval_config().experiment_report_md, "experiment_report.md")
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


PLOT_NAMES = {"tier_metrics.png", "baseline_comparison.png", "judge_agreement.png"}


def plot_path(name: str) -> Path | None:
    """Return the path to a known plot if it exists; None otherwise (prevents traversal)."""
    if name not in PLOT_NAMES:
        return None
    p = Path(eval_config().plots_dir) / name
    return p if p.exists() else None
