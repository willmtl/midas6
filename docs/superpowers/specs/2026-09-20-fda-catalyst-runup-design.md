# FDA Catalyst Run-Up — Design Spec

**Date:** 2026-09-20
**Status:** approved (design), building Phase 1
**Author:** William + Claude

## Concept

A pharma/biotech strategy that captures the **pre-catalyst run-up**: buy ahead of an FDA
decision (drug approval action date) and **sell before the binary print** — harvest the
well-documented anticipation drift while avoiding the coin-flip outcome. This is a *separate
book* from the small-cap-value flagship, judged on its own.

## Core bet & philosophy fit

Event-driven demand (specialist/biotech funds accumulating into a known catalyst) tends to
push the stock **up into** a binary FDA event; the outcome itself is a coin-flip we do NOT
hold through. Capturing drift and refusing the binary matches the house philosophy
("capture drift, don't gamble binaries"). Absolute-return objective; long-only; no leverage.

## Data

- **Calendar (primary):** openFDA `drug/drugsfda` (api.fda.gov, free, egress-confirmed from
  backend, 25,561 AP records, full history). Pull **original NDA/BLA approvals + efficacy
  supplements (sNDA)**; drop generics (ANDA). Fields: sponsor_name, application_number,
  submission_type (ORIG/SUPPL), submission_status_date (= action date), brand_name.
  Store → `/app/.data/fda_approvals.json` (no DB model until Phase C).
- **Sponsor→ticker map** (main engineering lift): fuzzy-match `sponsor_name` to our universe's
  company names; persist `/app/.data/fda_ticker_map.json`; hand-verify biotech/pharma matches;
  log unmatched. Unmatched sponsors are simply out of the tradeable set (documented, not hidden).
- **PIT enrichment / cross-check:** `NewsItem` (2015–2026) "FDA accepts" / "PDUFA" headlines
  (~189/21) → acceptance dates for the PIT-robustness arm.
- Prices: existing `Candle` daily; benchmark `XBI` (already in DB).

## The crux — PIT honesty (lives or dies here)

openFDA gives the action date *ex-post*. The strategy assumes that date was **knowable in
advance** (PDUFA target set ~10 months earlier at NDA acceptance) — true for the majority.
Integrity guard:
1. **Assumed-date arm:** entry window `[action_date − T_entry, action_date − T_exit]` using the
   openFDA action date.
2. **Acceptance-triggered arm (PIT-pure):** enter on the dated news "FDA accepts" headline,
   hold to just before the decision.
If drift appears only in arm 1, it is a look-ahead artifact, not an edge.

## Build phases

**Phase 1 — data + event study (start here; cheap kill-switch)**
- `fetch_openfda_approvals.py` → `fda_approvals.json`.
- sponsor→ticker map.
- `fda_runup_study.py`: for each mapped event, run-up return over `[T_entry, T_exit]` before the
  action date. Sweep `T_entry ∈ {5,10,21,42}`, `T_exit ∈ {0,3,5}` trading days. Report
  mean/median, win rate, t-stat, **both halves**, all **vs XBI**. Segment by **cap bucket ×
  event-type (ORIG vs sNDA)** — never averaged (hard rule). Save `BacktestResult[fda_runup_study]`.

**Phase 2 — PIT robustness (arm B)**
- Acceptance-triggered version from news; confirm arm 1 isn't look-ahead.

**Phase 3 — portfolio book (only if Phase 1 survives the gauntlet)**
- Monthly-rebalanced EW portfolio of names currently inside a pre-decision window; each sold
  before its decision. Full gauntlet + costs vs XBI; then `FdaApproval` model + live tab.

## Success criteria (gauntlet — required before Phase 3)

Run-up drift **positive**, **both halves** same sign, **beats XBI** (not just cash — biotech
beta is large), **survives the acceptance-triggered PIT arm**, and holds in **≥1 cap bucket**.
Fails any → stop or pivot to event-reaction; do not build the book.

## Hard-rule compliance

Never fabricate (real openFDA + news, verified reachable); full history (2015+ / openFDA all);
save every run to BacktestResult(+JSON); segment by cap × event-type; standalone book (own
universe); snapshot before any change; absolute-return objective, no DD/leverage overlays.

## Out of scope (YAGNI)

Trial-readout catalysts (noisier, harder PIT), vendor PDUFA feeds (paid/forward-only), options
structures, binary-hold bets, non-US approvals.
