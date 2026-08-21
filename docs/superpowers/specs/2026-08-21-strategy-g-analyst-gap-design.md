# Strategy G — Analyst-Gap Diversified Value Book — Design

**Date:** 2026-08-21
**Status:** ⛔ REFUTED (2026-08-21) — implemented, then killed. The entire backtested edge was a
split-adjustment data bug: `upside = analyst_target / adjusted_close` divided UNADJUSTED analyst targets
by SPLIT-ADJUSTED prices. Once targets are split-adjusted (each divided by the product of split ratios
after its date, using the EODHD-refreshed `splits_cache`), G-core is **CAGR +7.8% / maxDD −57.4% /
Sharpe 0.40**, losing to SPY (15.1% / −24% / 1.01) in every walk-forward window (verdict: fragile). All
the headline numbers below (+1,648% / CAGR 31% / DD −24%) are BOGUS artifacts of the bug. Build left in
place but inert. Do not deploy. Lesson: any analyst-target signal MUST split-adjust targets first.
**Author:** William + Claude

## Problem

The flagship is a spiky, capacity-capped return engine (CAGR ~100%, ceiling ~$5–10M, 81%/mo turnover, DD ~−23%). It has no diversified, higher-capacity sibling. This session isolated a clean, orthogonal signal that can fill that role:

**Analyst-gap** = how far a stock trades **below** its analyst price target. Isolated cross-sectionally (`analyst_gap_study.py`, none of the flagship machinery), the top-minus-bottom quintile earns **+1.96%/mo, t = 6.69** (market-demeaned, honest monthly-panel N=128) — a real, standalone predictive signal. The question this spec answers: **can it be turned into a deployable, diversified, high-capacity strategy whose objective is lower drawdown and higher win rate** (a smooth complement to the flagship, not a second high-octane bet)?

**Answer: yes.** Validated in `analyst_gap_book.py`. Strategy G is defined below.

## Objective (differs from the flagship)

Unlike the flagship (maximize absolute return, DD tolerated), Strategy G's stated goal is **lower drawdown + higher win rate at high capacity**, while still beating SPY on absolute return. This is a deliberate, per-strategy objective set by the user for G specifically — it does **not** revise the flagship's return-first mandate.

## Definition — Strategy G-core (canonical)

Monthly, equal-weight, long-only:

1. **Universe:** every analyst-covered US-listed non-ETF name (median trailing-180d price target available), including delisted-with-candles names (survivorship-aware). ~4,937 covered; ~4,933 with price history.
2. **Liquidity/price floor:** 20-day average dollar-volume ≥ **$5M/day** and price **> $1** at rebalance. (This is the capacity/honesty gate — the edge must not be microcap noise. It survives.)
3. **Quality gate (the free lunch):** **TTM net income > 0** (PIT: rolling-4Q sum forward-filled by `avail_date`). This is the single most important lever for the objective — it took DD from −38% to −24% and lifted win rate, at *no* return cost. Low-debt / ROE-line / turn / vol gates were tested and **rejected** (over-select or raise DD).
4. **Selection:** rank the profit-passing pool by implied upside = `median(trailing-180d target) / close − 1`, descending (furthest **below** target first). Take the **top 5%** (~60 names).
5. **Weighting:** **equal weight.** (Inverse-vol weighting is a documented variant — see G-smooth — but equal-weight is canonical: it has the highest win rate and return.)
6. **Rebalance:** monthly (month-end), full reconstitution.

### Performance (2016–2026, 128 months, net of realistic costs)

| Cost (bps/side) | Total | CAGR | Sharpe | maxDD | monthly hit | % mo > SPY | per-pos win |
|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | +1,692% | 31.1% | 1.28 | −24.0% | 70.3% | 71.1% | 57.2% |
| 10 (liquid, realistic) | +1,602% | 30.4% | 1.26 | −24.3% | 69.5% | 71.1% | 57.2% |
| 25 (all-in small/mid) | +1,475% | 29.5% | 1.23 | −24.8% | 69.5% | 70.3% | 57.2% |
| 50 (pessimistic) | +1,285% | 27.9% | 1.18 | −25.6% | 69.5% | 66.4% | 57.2% |
| **SPY (buy & hold)** | +347% | 15.1% | 1.01 | −23.9% | 71.1% | — | — |

**Turnover is only 21% one-way/month** (the gap signal is slow-moving), so cost drag is small — the edge is robustly net-of-cost. G-core beats SPY on **all three** axes (2x CAGR, higher Sharpe, ~equal DD) and 70% of months beat SPY.

### G-smooth (documented variant, not canonical)

G-core **+ inverse-vol weighting**: CAGR 27.9%, DD −22.2%, Sharpe 1.34. Trades ~3pp CAGR for ~2pp less DD and the best Sharpe. Offered for a lower-DD mandate; wired as a config flag, off by default.

## Why these choices (what was tested and rejected)

Full lever sweep in `analyst_gap_book.py` (set `G_SWEEP=1`), against the lower-DD/higher-win-rate objective:

- **Profit guard (ni>0): KEEP.** The one lever that lowers DD *and* raises win rate *and* holds return. Concentration-dependent — helps at top-5%, hurts at decile/quintile (re-admits value traps).
- **Market-regime cash-gate: REJECT.** Whipsaw — return +1692→+395, hit rate crashes to 53%, DD no better.
- **Market-regime half-exposure: REJECT (as default).** Lowers DD to −21% but halves return (CAGR 23.8%), same win rate. Not worth it for G-core; a possible conservative overlay only.
- **Entry-turn gate (buy only names that stopped falling): REJECT.** Raises Sharpe but makes DD *worse* (−39%) — concentrates into correlated momentum names.
- **Wider diversification (decile/quintile): REJECT.** Counterintuitively raises DD (−34/−37%) and cuts return — the profit guard already removed the crash-prone names; widening re-admits them.
- **Inverse-vol weighting: variant (G-smooth).** Lowers DD, best Sharpe, small return cost.
- **Vol gate / trim extreme gaps: REJECT.** Trimming the most-extreme gaps is a *disaster* (+127%, DD −35%) — confirms *tail-not-average*: the return lives in the extreme-gap tail and cannot be smoothed away by avoiding it.

## Relationship to the flagship

Orthogonal and complementary, not a competitor:
- **Flagship:** ~1 name/month, drift-P/B + A/D-conviction + tl_rsi entry, sector-rotation filter, tiny illiquid names, CAGR ~100%, capacity ~$5–10M.
- **Strategy G:** ~60 liquid names/month, analyst-gap + profit guard, no rotation filter, CAGR ~30%, **high capacity** ($5M/day floor per name × 60 names → materially larger AUM).

Analyst overlays were separately proven to add **no** value *inside* the flagship (veto/tiebreak/driver all lose to drift-P/B — `ANALYST_OVERLAY_LAB`). Strategy G is the correct home for the analyst-gap edge: a separate vehicle, not a flagship tweak.

## Architecture

**Chosen: a standalone, self-contained engine `strategy_g.py` (root-mounted), independent of the flagship.** It owns its universe build, PIT analyst-upside panel, PIT profitability panel (reuses `seq_fundamental_study.load_financial_reports`), liquidity floor, selection, equal-weight monthly sim, and cost model. `analyst_gap_book.py` is the validated research prototype; `strategy_g.py` is its productionized form.

Rejected: (a) folding G into `survivorship_smallcap_study.py` — that file is the flagship and is already huge; G's objective and selection differ; keep them isolated. (b) A pure config-flag on the flagship — the analyst overlay was already proven inert/harmful there.

**Non-breaking:** purely additive. Reuses `load_financial_reports` and `.data/analyst_ratings.jsonl` read-only. New `BacktestResult` kind `strategy_g`. New optional URL/tab in a later phase.

## Components

### 1. `strategy_g.py` — engine (root-mounted, PIT)
- Universe: analyst-covered US non-ETF ∪ delisted-with-candles (survivorship-aware), from `.data/analyst_ratings.jsonl`.
- Panels (monthly, PIT): month-end close, 20d-avg dollar-volume, implied-upside (median trailing-180d target/close−1), TTM net income (>0 gate), trailing 60d daily-vol (for G-smooth only).
- `sim(top_frac=0.05, profit_gate=True, weight="equal"|"invvol", cost_bps=…)` → monthly returns, holdings, turnover, win rates. Charges `2 × one-way-turnover × cost_bps` per month.
- CLI: `--db` (persist `BacktestResult`), `--variant core|smooth`, `--cost-bps N`, `--sweep` (full lever table). Saves `strategy_g_picks.json` (latest month's holdings) for a live scanner.
- **Every run persists a `BacktestResult(+JSON)`** (project rule: no throwaway prints).

### 2. Persistence & doc (Phase 1 — minimum viable)
- `BacktestResult` kind `strategy_g` with the net-of-cost curve, holdings history, turnover, win-rate stats, and the lever-sweep table (for the record of what was rejected).
- Short tearsheet reusing the flagship doc pipeline pattern if cheap; otherwise a static summary.

### 3. Live scanner + API + frontend tab (Phase 2 — optional, gated on Phase 1 sign-off)
- Monthly-pick scanner writing current G-core holdings (mirrors the flagship live-pick plumbing).
- `/api/strategy-g` endpoint + a frontend tab. Deferred until the persisted study is reviewed.

## Data integrity & PIT correctness (must hold)
- **No look-ahead:** upside uses only targets dated ≤ month-end within a trailing 180d window; profitability uses TTM sums forward-filled by `avail_date` (filing availability), never `period_end`.
- **Survivorship:** universe includes delisted-with-candles names (the coverage fix this session: 70%→95% picks). Delisted names exit at last/deal price; −100% only on confirmed bankruptcy (per the verified delisted-survivorship handling).
- **Currency:** US-listed only (`"." not in ticker`), so target and price currencies match.
- **Sentinel/seam hygiene:** relies on the cleaned Candle table (garbage sentinel bars deleted, seams verified) from this session's data-integrity audit.

## Risks & open items
- **Analyst-data coverage skew:** coverage is denser post-2015; pre-2015 not backtested here (panel starts 2015-12). In-sample only 2016–2026 — no separate walk-forward OOS yet. **Recommend** a subperiod / walk-forward robustness pass in the implementation plan before Phase 2 wiring.
- **Capacity not formally sized:** $5M/day floor × ~60 names suggests high capacity but a real capacity curve (like `flagship-capacity`) is not yet run.
- **Costs:** modeled as flat bps on turnover; no borrow (long-only, N/A) and no market-impact curve. Validated robust to 100 bps/side, so low sensitivity.
- **Tax/holding:** monthly rebalance → short-term gains; noted, not optimized (return-first framing).

## Success criteria
1. `strategy_g.py` reproduces `analyst_gap_book.py`'s G-core net-of-cost numbers (CAGR ~29.5–30.4% at 10–25 bps; DD ~−25%; monthly hit ~70%).
2. Persisted `BacktestResult(strategy_g)` with full curve + lever-sweep record.
3. Walk-forward / subperiod pass confirms the edge is not a single-regime artifact (gate for Phase 2).
4. (Phase 2, optional) live monthly picks reconcile with the engine.
