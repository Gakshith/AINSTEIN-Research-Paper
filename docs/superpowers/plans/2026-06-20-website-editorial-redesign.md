# AINSTEIN Editorial Science Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the AINSTEIN web UI as an Apple-grade "Editorial Science" experience (light-default, dark toggle retained) without changing any backend contract; keep tests green.

**Architecture:** Presentation-layer rewrite of `static/css/theme.css`, the four Jinja templates, and chart/motion cosmetics in `static/js/app.js`. All JS control flow and API contracts preserved. One trivial safe backend addition: a `HEAD /` route with a test.

**Tech Stack:** FastAPI + Jinja2 (existing), vanilla JS, Chart.js 4, CSS custom properties (no build step).

---

## File structure

- `static/css/theme.css` — single source of truth for design tokens (light+dark), type/spacing scales, and every component. Full rewrite.
- `templates/base.html` — document shell: sticky blur nav with SVG logo, footer, pre-paint theme restore (default light).
- `templates/index.html` — dashboard: hero, stat grid, chart sections, papers table.
- `templates/paper.html` — paper drill-down.
- `templates/demo.html` — live pipeline controls + results.
- `static/js/app.js` — same logic; upgraded chart options/palette, scroll-reveal, default-light.
- `app.py` — add `HEAD /` route.
- `tests/test_api.py` — add HEAD-route test.

Design tokens & language are fully specified in
`docs/superpowers/specs/2026-06-20-website-editorial-redesign-design.md` — implement to that.

---

## Task 1: Backend — HEAD `/` route (TDD)

**Files:**
- Modify: `app.py` (index route area)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test** in `tests/test_api.py`:

```python
def test_head_root_ok(client):
    resp = client.head("/")
    assert resp.status_code == 200
```

(Use the existing `client` fixture/pattern already in `tests/test_api.py`; if tests build the client inline, mirror that.)

- [ ] **Step 2: Run it, expect FAIL** (405 Method Not Allowed):
Run: `./venv/bin/python -m pytest tests/test_api.py::test_head_root_ok -v`
Expected: FAIL (405).

- [ ] **Step 3: Implement** — allow HEAD on the index route in `app.py`:

```python
@app.get("/", response_class=HTMLResponse)
@app.head("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"available": services.artifacts_available()}
    )
```

- [ ] **Step 4: Run it, expect PASS.**
Run: `./venv/bin/python -m pytest tests/test_api.py::test_head_root_ok -v`

- [ ] **Step 5: Commit.**

```bash
git add app.py tests/test_api.py
git commit -m "feat(web): support HEAD / for health checks"
```

## Task 2: CSS foundation — tokens, reset, type & layout primitives

**Files:** Modify (rewrite): `static/css/theme.css`

- [ ] Define `:root` light tokens and `[data-theme="dark"]` overrides per spec palette
  (canvas/surface/ink/dim/faint/border/accent/chart tints), radius scale, shadow scale,
  8px spacing scale, font stack (`-apple-system, "SF Pro Display", Inter, …`).
- [ ] Reset/base: box-sizing, body bg/ink, antialiasing, `.wrap` max-width container,
  type scale (`h1` display `clamp(34px,5vw,60px)` light weight, section `h2`, body 16–17px,
  `.lead`, `.eyebrow`, `.hint`), link/focus-visible styles, `prefers-reduced-motion` guard.
- [ ] Commit: `style(web): css design tokens + typographic foundation`.

## Task 3: CSS components — nav, hero, cards, table, chips, badges, buttons, inputs, metric bars, stages, panels, footer, motion

**Files:** Modify: `static/css/theme.css`

- [ ] Sticky `.nav` with `backdrop-filter: blur()`, hairline-on-scroll (`.nav.scrolled`),
  brand with SVG logo mark, active link state.
- [ ] `.hero` editorial spacing; `.grad` accent treatment (subtle, not neon).
- [ ] `.stats`/`.stat` grid; `.card`, `.grid-2`, `.chart-box`; `.section` rhythm.
- [ ] `.data` table: hairline rows, hover lift, `.title` cell, tabular-nums, clickable rows.
- [ ] `.chip-oral/.chip-spotlight/.chip-poster`, `.badge.ok/.no` with dot, `.btn.primary`,
  `.input`, `.select`, `.toolbar`.
- [ ] `.metric-row/.m-track/.m-fill` animated bars; `.stage-list/.stage` states + `.spinner`;
  `.panel/.sol-grid`; `.empty` state; `footer`.
- [ ] `.fade-in` + `.reveal` scroll-reveal classes; responsive breakpoints (<860px, <560px).
- [ ] Commit: `style(web): apple-grade components, table, charts shell, motion`.

## Task 4: Templates — base shell + three pages

**Files:** Modify: `templates/base.html`, `index.html`, `paper.html`, `demo.html`

- [ ] `base.html`: default `data-theme="light"`, pre-paint restore honoring saved choice
  then `prefers-color-scheme`; inline SVG logo mark in brand; refined footer; bump asset
  cache-bust (`theme.css?v=4`, `app.js?v=3`).
- [ ] `index.html`: editorial hero copy, stat grid container, chart sections, table toolbar
  — markup aligned to new classes; keep `{% if available %}` + empty-state branches.
- [ ] `paper.html`: restyled header/meta/badges, problem panel, solutions + judge verdicts,
  similarity metric bars — keep the `renderMetricBars` script call and field names.
- [ ] `demo.html`: restyled controls/stage list/result — keep ids `rowIndex/paperId/runBtn/
  runHint/progressSection/stageList/resultSection/resultBox` and the `initDemo()` call.
- [ ] Commit: `style(web): editorial templates for dashboard, paper, demo`.

## Task 5: JS — chart cosmetics, scroll-reveal, default light

**Files:** Modify: `static/js/app.js`

- [ ] Chart options: rounded bars (`borderRadius`), drop y gridlines or set to hairline,
  hide/lighten legend, muted token-driven palette, tooltip styling; keep all data wiring.
- [ ] Add IntersectionObserver that adds `.in` to `.reveal` elements on enter (guarded by
  `prefers-reduced-motion`); call from `initDashboard` and on other pages via `setupTheme`/
  DOMContentLoaded.
- [ ] Theme: keep toggle; first-load default light is handled in `base.html` (no regression).
- [ ] Keep `window.AINSTEIN` exports and all handlers identical.
- [ ] Commit: `style(web): refined charts + scroll reveal`.

## Task 6: Verify & deliver

- [ ] Run full suite: `./venv/bin/python -m pytest tests -q` → all green (22).
- [ ] Start server: `AINSTEIN_SAMPLE=1 ./venv/bin/uvicorn app:app --port 8011` (via preview).
- [ ] Load `/`, a `/paper/{id}`, `/demo`; check console clean; test search/filter + toggle.
- [ ] Screenshot light + dark dashboard and a paper page into `Artifacts/`.
- [ ] Hand user the preview link.

---

## Self-review

- Spec coverage: tokens/light+dark (T2), all components incl. nav blur/logo/charts/motion
  (T3/T5), all four templates (T4), HEAD route + test (T1), verification + screenshots (T6). ✓
- Placeholder scan: CSS/template content is specified by class inventory + the spec's design
  language (the spec is the detailed source); the one code-bearing backend change shows full
  code. No TBDs. ✓
- Consistency: element ids and JS export names preserved exactly to avoid breaking handlers. ✓
