# AINSTEIN Website — Harvest "Golden-hour workbench" re-skin (Design Spec)

Date: 2026-06-20
Branch: `feature/website-and-cleanup`
Status: Implemented & verified
Supersedes (visual layer only): `2026-06-20-website-editorial-redesign-design.md`

## Goal

Re-skin the AINSTEIN web app to the user-supplied **Harvest** design system: a warm,
sunlit productivity-workspace language. The structure, templates, JS control flow, and all
backend/API contracts are unchanged — this is a design-token + treatment swap on top of the
existing presentation layer.

## Design language (from the supplied style reference)

- **Canvas:** warm cream `#fff8f1` (never pure white) — the signature brand surface.
- **Cards:** paper white `#ffffff` floating above the cream via warm-tinted shadows.
- **Accent:** one vivid orange flame `#fa5d00` — CTAs, links, active states, brand marks,
  chart focal series. No second accent color (system discipline).
- **Neutrals:** warm grays — ink `#1d1e1c`, warm stone `#615f5c`, driftwood `#8e8b87`,
  bone `#c0bbb6` borders.
- **Type:** Inter (MuotoWeb substitute) for all UI/body; **Fraunces** (Monarch substitute)
  reserved for the hero display headline only — the one serif "roar" against the sans body.
- **Shape:** 16px radius buttons/inputs, 20px cards, 999px pills; warm-tinted shadows
  (`rgba(250,166,0,0.18)` card glow, `rgba(0,0,0,0.2)` button lift) — never cold/blue.
- **Atmosphere:** a soft orange→marigold gradient wash behind the hero suggests warmth and
  movement without competing with content.

## Decisions / adaptations

- **Charts:** to keep multi-series legibility under the one-orange rule, series use orange,
  a lighter flame `#ffa05a`, and a warm-neutral gray `#8e8b87` — shades + neutrals, not a
  second hue.
- **Tier chips:** Oral carries the flame (standout); Spotlight/Poster stay warm-neutral so
  orange remains the single chromatic focal point.
- **Dark toggle:** the Harvest spec is light-only, but the user previously required a dark
  toggle. Kept as a **warm-dark** variant (roasted-brown canvas `#1a1512`, cream ink, same
  flame) rather than a cold dark — honors both the toggle requirement and the system warmth.
  Light is the brand default on first load; a saved toggle choice wins.

## Files changed

- `static/css/theme.css` — full token + component re-skin to the Harvest palette/typography/
  shapes/shadows (light + warm-dark).
- `templates/base.html` — load Inter + Fraunces, orange favicon, cache-bust bump.
- `static/js/app.js` — unchanged; its chart wiring reads `--brand/--brand-2/--accent/--warn`,
  so the token remap re-colors charts automatically.

## Verification

- `pytest tests -q` → **22 passed** (offline).
- Live (port 8011, `AINSTEIN_SAMPLE=1`), no console errors: dashboard hero + stats + tier/
  judge/baseline charts + per-paper table + tier chips; paper drill-down (orange metric bars,
  flame chips/badges); light + warm-dark toggle. Fonts (Fraunces hero, Inter body) load.
