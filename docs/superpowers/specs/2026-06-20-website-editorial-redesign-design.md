# AINSTEIN Website — Editorial Science Redesign (Design Spec)

Date: 2026-06-20
Branch: `feature/website-and-cleanup` (continues existing website work)
Status: Approved

## Goal

Transform the AINSTEIN web app from a competent-but-generic dark dashboard into an
Apple-grade **Editorial Science** experience that reads like a premium research
publication / Apple product page — deliberately *not* like a typical AI-generated
dashboard. The Python backend and all API/data contracts stay untouched; the existing
test suite stays green.

Non-goals: no pipeline changes, no real evaluation run (no HF token dependency), no
risky refactors, no new data sources. Presentation layer only, plus one trivial safe
backend addition (HEAD `/`).

## Design language

**Primary theme — Editorial Light (default & showpiece):**
- Canvas: `#FBFBFD` warm-white; surfaces `#FFFFFF` with hairline borders.
- Ink: `#1D1D1F` near-black; dim `#6E6E73`; faint `#86868B` (Apple-grade greys).
- Accent: electric indigo `#4F46E5` (single confident accent), with a restrained
  secondary tint for charts (e.g. teal `#0EA5A4`, amber `#D97706`) used sparingly.
- Whitespace-forward, hairline dividers, soft layered shadows, no neon, no gradient soup.

**Secondary theme — Dark (toggle retained):** a tasteful deep-charcoal variant
(`#0B0B0F` canvas, off-white ink, same indigo accent) so the existing toggle keeps
working. Light is the default on first load.

**Craft details that separate Apple-grade from generic:**
- A real **type scale**: large light-weight `clamp()` display headlines, tight readable
  body. SF Pro / -apple-system first, Inter fallback.
- An **8px spacing system** and consistent radius tokens.
- Sticky **backdrop-blur nav** that gains a hairline border on scroll.
- Custom inline-**SVG logo mark** replacing the plain "A" letter.
- **Scroll-reveal** on sections via IntersectionObserver; subtle hover lifts on cards/rows.
- Charts redrawn precise & muted: rounded bars, no chart-junk gridlines, restrained palette.
- Tabular numerals for metrics; accessible focus states; `prefers-reduced-motion` respected.

## Architecture (unchanged contracts)

The frontend consumes the existing JSON API exactly as today:
- `/api/summary`, `/api/tiers`, `/api/baselines`, `/api/papers`, `/api/papers/{id}`,
  `/api/report`, `/api/plots/{name}`, `/api/demo/run`, `/api/demo/{job_id}`,
  `/api/health`.
- Server-rendered pages: `/` (dashboard), `/paper/{id}`, `/demo`.

No endpoint shapes change. JS keeps the same control flow (theme, charts, table
search/filter, paper metric bars, live-demo polling). Only styling, markup structure,
and chart cosmetics change.

## Components & files

Presentation layer (rewrites):
- `static/css/theme.css` — full rewrite: design tokens (light + dark), type & spacing
  scales, nav, hero, stat grid, section rhythm, cards, data table, chips, badges,
  buttons, inputs, metric bars, stage list, panels, footer, motion, responsive.
- `templates/base.html` — refined sticky blur nav with SVG logo mark, refined footer,
  default `data-theme="light"` with pre-paint theme restore + `prefers-color-scheme`.
- `templates/index.html` — editorial hero, refined stat cards, sectioned charts, polished
  papers table + toolbar.
- `templates/paper.html` — restyled drill-down (header, problem, solutions, judge
  verdicts, similarity metric bars).
- `templates/demo.html` — restyled live-demo controls, stage list, result panels.
- `static/js/app.js` — **all logic preserved**; upgrade chart options/palette, add
  scroll-reveal IntersectionObserver, default theme light, keep toggle + all handlers.

Backend (minimal, safe):
- `app.py` — add `HEAD /` route returning 200 (roadmap nice-to-have; harmless).
- `tests/test_api.py` — add a test asserting `HEAD /` → 200.

## Data flow

Unchanged. Pages fetch JSON on load (dashboard/demo) or are server-rendered with the
paper dict (paper page). Sample data still surfaced via `AINSTEIN_SAMPLE=1` through the
existing `services.py` resolution — used for the preview/screenshots.

## Error handling

- Empty-state cards (no artifacts / no token) preserved and restyled, not removed.
- Demo 503/409 handling preserved in JS.
- `prefers-reduced-motion`: disable scroll-reveal/transitions.
- Theme restore guarded against missing `localStorage` (existing pattern kept).

## Testing & verification

- `pytest` → existing 21 tests + 1 new HEAD-route test all green (offline).
- Launch `uvicorn app:app` with `AINSTEIN_SAMPLE=1`; load `/`, `/paper/{id}`, `/demo`.
- Check browser console for errors; exercise table search/filter and the theme toggle.
- Capture screenshots (light + dark, dashboard + paper) as Artifacts and hand the user a
  live preview link.

## Rollout

Continue on `feature/website-and-cleanup`. Commit the redesign, keep tests green, deliver
preview link + screenshots. No deploy step.
