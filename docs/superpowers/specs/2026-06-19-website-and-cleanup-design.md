# Design: AINSTEIN cleanup, hardening & website (2026-06-19)

## Problem
The AINSTEIN pipeline core was solid (7-stage DAG + baselines + reporting, ~25 metrics/paper),
but the repo had accumulated correctness bugs, broken/dead scaffolding, no runtime guardrails,
and no web layer despite Flask being listed in requirements. Goal: fix every real error, harden
for a clean first run, get tests green + expanded, and add a polished website.

## Decisions
- **Website scope:** results dashboard + live single-paper demo.
- **Backend:** FastAPI (replaced empty Flask `app.py`; dropped Flask/Flask-Cors).
- **Frontend:** server-rendered Jinja2 + CSS-variable light/dark themes + Chart.js (no build step).
- **Critique loop:** when no candidate passes both critiques, keep the last candidate as a
  best-effort result and expose `critique_passed: bool` (instead of silently returning None).
- **Sample data:** bundled under `web/sample/`, surfaced only when `AINSTEIN_SAMPLE=1` and no real
  artifacts exist — so the dashboard looks great out-of-the-box without a costly run, while the
  honest empty-state still shows by default.

## Architecture
- `web/services.py` — pure artifact readers; all paths come from `EvaluationConfig` via
  `ConfigurationManager` (no duplicated paths). JSON-safe; degrade to empty/None when missing.
- `web/runner.py` — background, concurrency-limited live-demo job runner wrapping
  `AINSTEINController`; token-guarded; paper selection via `AINSTEIN_ROW_INDEX/PAPER_ID` env.
- `app.py` — FastAPI: JSON API (`/api/*`, auto OpenAPI at `/docs`) + server-rendered pages
  (`/`, `/paper/{id}`, `/demo`).
- `templates/` + `static/` — base/index/paper/demo, theme.css, app.js.

## Guards / fixes
- `require_hf_token()` / `hf_token_available()` in `utils/common.py`, called at both pipeline
  entry points; live demo degrades gracefully without a token.
- Existence guard in `data_validation`; missing internal-status treated as not-passed in stage 06.
- Removed stray `models/__init__.pyconfig/config.yaml`; deduped imports; fixed `temperature`
  type hints; added repo `.gitignore`; pinned `requirements.txt` (+ `requirements-dev.txt`).

## Out of scope
- Real batch run (needs HF token + cost). Live demo covers single-paper on demand.
- Persisting jobs across restarts (in-memory is sufficient for a single-instance demo).
