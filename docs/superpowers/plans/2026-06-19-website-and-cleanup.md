# Plan: AINSTEIN cleanup, hardening & website (2026-06-19) — DONE

Branch: `feature/website-and-cleanup`. Spec: `../specs/2026-06-19-website-and-cleanup-design.md`.

## Part A — fixes & hardening ✅
- [x] Remove stray `src/AINSTEIN/models/__init__.pyconfig/config.yaml`
- [x] `config_entity.py`: `temperature: int → float`; remove duplicate `dataclass` import; normalize indent
- [x] Dedupe imports in `configuration.py` and `common.py`; add path context to `read_yaml` error
- [x] `require_hf_token()` / `hf_token_available()` helper; wired into `main.py` + `evaluation_result.py`
- [x] `data_validation` missing-file guard; `stage_06` missing-status guard
- [x] Critique loop: best-effort last candidate + `critique_passed` flag
- [x] `requirements.txt` pinned; `requirements-dev.txt`; `.env.example`; repo `.gitignore`
- [x] README: fixed author-machine paths; added Environment + Web app sections; Dockerfile serves web

## Part B — tests ✅ (21 passing, offline)
- [x] Fixed pre-existing fixture drift (missing `*_md` fields) in eval/reporting tests
- [x] `test_guards.py`, `test_controller_loop.py`, `test_api.py` (FastAPI TestClient)

## Part D — website ✅
- [x] `web/services.py`, `web/runner.py`, `app.py` (FastAPI)
- [x] `templates/` base/index/paper/demo; `static/css/theme.css`; `static/js/app.js`
- [x] Bundled sample data via `scripts/make_sample_data.py` (opt-in `AINSTEIN_SAMPLE=1`)
- [x] Verified in browser: dashboard (light+dark), charts, table filter, paper drill-down,
      demo token-gated empty state; all endpoints 200; theme toggle fixed (double-bind bug)

## Remaining (not blocking)
- [ ] Run a real batch evaluation with a HF token to replace sample data
- [ ] Optional: post-merge code review pass; push branch / open PR
- [ ] Consider HEAD `/` support (preview health probe returns 405 — harmless)
