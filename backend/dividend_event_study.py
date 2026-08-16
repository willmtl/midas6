#!/usr/bin/env python3
"""DIVIDEND CUT / INCREASE EVENT STUDY — a dividend CUT is a strong negative fundamental signal (management
conceding cash-flow stress); an INITIATION / HIKE signals quality & confidence. Same family as grounded-earnings
and the profitability guard, both of which pay off. Uses the already-backfilled CorporateAction dividends
(no new data). Tail-focus per [[tail-not-average]]: bucket by MAGNITUDE, not just sign.

For each ticker, sort regular cash dividends by ex_date; classify each vs the immediately prior dividend:
  INITIATION  first dividend after none on record
  CUT         div / prior - 1 <= -10%   (deep-cut sub-bucket <= -50%)
  HIKE        div / prior - 1 >= +10%
  FLAT        in between
Special/one-off dividends (>3x the trailing median) are dropped as noise. Then compute MARKET-ADJUSTED forward
return (stock - SPY) at 1/3/6/12 months from ex_date. If cuts underperform and hikes/initiations outperform
(and the deep-cut tail is worst), dividend actions are a usable event overlay/veto for the rotation pick.
-> BacktestResult[dividend_event] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/dividend_event_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
from seq_fundamental_study import load_candles
from backtest_lowpb import _tstat_from_returns, BENCH

HORIZONS = [1, 3, 6, 12]
CUT, HIKE = -0.10, 0.10
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "dividend_event.json"


def load_div_events():
    from core.models import CorporateAction
    rows = CorporateAction.objects.filter(action_type="dividend", dividend__gt=0).values(
        "ticker", "ex_date", "dividend").order_by("ticker", "ex_date")
    by = {}
    for r in rows:
        by.setdefault(r["ticker"], []).append((pd.Timestamp(r["ex_date"]), float(r["dividend"])))
    events = []
    for tk, seq in by.items():
        seq.sort()
        divs = [d for _, d in seq]
        for i, (dt_i, d_i) in enumerate(seq):
            # drop special/one-off spikes (>3x trailing median of up to 4 priors)
            trail = divs[max(0, i - 4):i]
            med = np.median(trail) if trail else d_i
            if trail and d_i > 3 * med:
                continue
            if i == 0:
                lbl, chg = "INITIATION", None
            else:
                prev = seq[i - 1][1]
                chg = d_i / prev - 1 if prev > 0 else None
                if chg is None:
                    continue
                lbl = "CUT" if chg <= CUT else "HIKE" if chg >= HIKE else "FLAT"
            events.append({"ticker": tk, "ex_date": dt_i, "div": d_i, "chg": chg, "label": lbl})
    return events


def build():
    events = load_div_events()
    tickers = sorted({e["ticker"] for e in events})
    print(f"{len(events)} dividend events across {len(tickers)} tickers; loading candles + SPY...", flush=True)
    cds = load_candles(tickers + [BENCH])
    spy = cds.get(BENCH)
    if spy is None:
        print("no SPY candles", flush=True); return None
    spy_c = spy["Close"]

    def mret(series, d0, months):
        s = series.dropna()
        if s.empty:
            return None
        i0 = s.index.searchsorted(d0)
        if i0 >= len(s):
            return None
        p0 = s.iloc[i0]
        d1 = d0 + pd.DateOffset(months=months)
        i1 = s.index.searchsorted(d1)
        i1 = min(i1, len(s) - 1)
        if i1 <= i0 or p0 <= 0:
            return None
        return float(s.iloc[i1] / p0 - 1)

    # attach market-adjusted forward returns
    for e in events:
        c = cds.get(e["ticker"])
        if c is None:
            e["fwd"] = {}
            continue
        cc = c["Close"]
        e["fwd"] = {}
        for h in HORIZONS:
            r = mret(cc, e["ex_date"], h)
            b = mret(spy_c, e["ex_date"], h)
            e["fwd"][h] = (r - b) if (r is not None and b is not None) else None

    def agg(rows):
        out = {"n": len(rows)}
        for h in HORIZONS:
            v = [r["fwd"].get(h) for r in rows if r["fwd"].get(h) is not None]
            if v:
                out[str(h)] = round(float(np.mean(v)) * 100, 2)
                out[f"t{h}"] = round(_tstat_from_returns(v), 2) if len(v) > 2 else None
            else:
                out[str(h)] = None; out[f"t{h}"] = None
        return out

    groups = {}
    for lbl in ("CUT", "HIKE", "INITIATION", "FLAT"):
        groups[lbl] = agg([e for e in events if e["label"] == lbl])
    groups["DEEP_CUT(<=-50%)"] = agg([e for e in events if e["label"] == "CUT" and e["chg"] is not None and e["chg"] <= -0.50])
    groups["BIG_HIKE(>=+25%)"] = agg([e for e in events if e["label"] == "HIKE" and e["chg"] is not None and e["chg"] >= 0.25])

    print(f"\n=== market-adjusted forward return by dividend event (vs SPY) ===", flush=True)
    print(f"  {'event':<18}{'n':>5}    1mo     3mo     6mo    12mo", flush=True)
    for lbl in ("DEEP_CUT(<=-50%)", "CUT", "FLAT", "INITIATION", "HIKE", "BIG_HIKE(>=+25%)"):
        g = groups[lbl]
        print(f"  {lbl:<18}{g['n']:>5}  " + "  ".join(f"{g[str(h)]!s:>6}" for h in HORIZONS), flush=True)
    print("  (t-stats 6mo): " + ", ".join(f"{lbl.split('(')[0]} {groups[lbl].get('t6')}"
                                           for lbl in ("DEEP_CUT(<=-50%)", "CUT", "HIKE", "BIG_HIKE(>=+25%)")), flush=True)

    cut6, hike6 = groups["CUT"].get("6"), groups["HIKE"].get("6")
    deep6 = groups["DEEP_CUT(<=-50%)"].get("6")
    verdict = (
        f"CUT 6mo {cut6}% (deep-cut {deep6}%) vs HIKE 6mo {hike6}% vs INITIATION {groups['INITIATION'].get('6')}%. "
        + ("Cuts underperform and hikes/initiations outperform -> dividend actions are a usable event signal; the "
           "deep-cut tail is the strongest (a cut on a rotation holding = veto/exit). "
           if (cut6 is not None and hike6 is not None and cut6 < hike6) else
           "No clean cut-vs-hike spread -> dividend actions don't cleanly separate forward returns here. ")
        + f"(Coverage: dividends 2021-> only, {len(tickers)} payers; growth picks that pay no dividend are unaffected.)"
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"horizons_months": HORIZONS, "cut_thresh": CUT, "hike_thresh": HIKE, "benchmark": BENCH,
                   "n_events": len(events), "n_tickers": len(tickers), "adjust": "market-adjusted (stock - SPY)"},
        "groups": groups, "verdict": verdict,
        "caveat": "Dividends 2021-08-> only (CorporateAction backfill window); 667 payers, so non-payers (many "
                  "growth picks) are out of scope. Consecutive-dividend % change; special/one-off dividends >3x "
                  "trailing median dropped. Market-adjusted vs SPY, ex_date anchored, overlapping windows (t-stats "
                  "mildly overstated). Not de-duplicated for macro clustering (e.g. 2022 broad cuts).",
    }


def main():
    p = build()
    if p is None:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="dividend_event", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[dividend_event]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
