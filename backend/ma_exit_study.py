# -*- coding: utf-8 -*-
"""M&A ANNOUNCEMENT-EXIT BACKTEST — the payoff step for the LLM-classified deal events.

Question: for a flagship hold that gets an M&A bid mid-month, does SELLING INTO THE ANNOUNCEMENT POP
(exit on the deal-announcement day) beat the current month-end / delisting exit — and does dodging the
announcement automatically dodge the later WITHDRAWALS?

Method (clean A/B, reuses the exact flagship picks):
  - baseline holds = flagship_history.json (every usca_small pick, entry `date` -> exit `ndate`, monthly).
  - deal events   = ma_events.jsonl (LLM target events: is_target, status, dt, offer_price, confidence).
  - For each hold [date, ndate] of ticker p, if p has a TARGET deal event with dt in (date, ndate]:
        sell into the pop  = first daily close on/after the announcement dt (then cash for the rest of month).
    Variants: announced/completed only  |  + rumored (sell the rumor pop, dodge the withdrawal crash).
  - All prices from the SAME daily Candle series so the only A/B difference is the EXIT DATE (pop vs month-end).
    Unaffected holds keep their trace return. Recompute the weighted monthly basket + total for each variant.

Run: docker exec rotation-backend-1 python -u /app/ma_exit_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from collections import Counter
from seq_fundamental_study import load_candles

TRACE = Path("/app/.data/studies/flagship_history.json")
EVENTS = Path("/app/.data/ma_events.jsonl")
OUT = Path("/app/.data/studies/ma_exit_study.json")
CONF_MIN = 0.6
ANN = {"announced", "completed"}
POP = {"announced", "completed", "rumored"}


def load_events():
    ev = {}
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if not e.get("is_target") or e.get("status") in (None, "none"):
            continue
        if (e.get("confidence") or 0) < CONF_MIN or not e.get("dt"):
            continue
        ev.setdefault(e["ticker"], []).append(
            {"dt": pd.Timestamp(e["dt"]).tz_localize(None), "status": e["status"],
             "acq": e.get("acquirer"), "offer": e.get("offer_price"), "title": e.get("title")})
    for t in ev:
        ev[t].sort(key=lambda x: x["dt"])
    return ev


def px_on(df, ts, forward=False):
    if df is None or "Close" not in df:
        return None, None
    idx = df.index
    if forward:
        sub = df[idx >= ts]
        if len(sub) == 0:
            return None, None
        return float(sub["Close"].iloc[0]), sub.index[0]
    sub = df[idx <= ts]
    if len(sub) == 0:
        return None, None
    return float(sub["Close"].iloc[-1]), sub.index[-1]


def perf(rets):
    r = np.asarray(rets, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), annual=round(ann, 1), sharpe=round(sh, 2), dd=round(dd, 1), months=n)


def main():
    trace = json.load(open(TRACE))
    months = trace["months"]
    events = load_events()
    print(f"target deal events (conf>={CONF_MIN}): {sum(len(v) for v in events.values())} across {len(events)} tickers",
          flush=True)

    # --- find affected holds (a pick whose ticker has a deal event inside its hold window) ---
    affected = []           # (mi, pick_index, event)
    for mi, m in enumerate(months):
        d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"])
        for pi, p in enumerate(m["picks"]):
            if p["ret"] is None:
                continue
            for e in events.get(p["ticker"], []):
                if d0 < e["dt"] <= d1:
                    affected.append((mi, pi, e))
                    break
    stat = Counter(e["status"] for _, _, e in affected)
    print(f"affected holds (deal announced DURING a flagship hold): {len(affected)}  status={dict(stat)}", flush=True)
    if not affected:
        print("No flagship hold overlaps a deal window — M&A exit cannot move this strategy. Done.", flush=True)
        return

    # --- daily prices for the affected tickers only ---
    tks = sorted({months[mi]["picks"][pi]["ticker"] for mi, pi, _ in affected})
    daily = load_candles(tks)
    print(f"loaded daily candles for {len(daily)}/{len(tks)} affected tickers", flush=True)

    # --- per affected hold: entry, announcement-pop, month-end prices (same series) ---
    detail = []
    repl = {}   # (mi,pi) -> {"ma": ma_ret, "base": base_ret, "status":...}
    for mi, pi, e in affected:
        m = months[mi]; p = m["picks"][pi]
        d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"])
        df = daily.get(p["ticker"])
        entry, _ = px_on(df, d0)
        annp, annd = px_on(df, e["dt"], forward=True)
        exitp, _ = px_on(df, d1)
        if not entry or entry <= 0 or not annp:
            continue
        base = (exitp / entry - 1) if (exitp and exitp > 0) else p["ret"]   # daily-basis month-end (fallback trace)
        # only sell into the pop if the announcement price is at/above entry-ish (a real pop, not a crashed
        # 'withdrawn' print) AND the announcement is before month-end anyway
        ma = annp / entry - 1
        repl[(mi, pi)] = {"ma": ma, "base": base, "status": e["status"]}
        detail.append({"date": m["date"], "ticker": p["ticker"], "company": p.get("company"),
                       "sector": p["sector"], "status": e["status"], "acquirer": e["acq"], "offer": e["offer"],
                       "ann_date": str(annd.date()) if annd is not None else None,
                       "trace_ret": round(p["ret"], 4), "base_ret": round(base, 4), "ma_ret": round(ma, 4),
                       "delta": round(ma - base, 4), "title": e["title"]})

    # --- recompute weighted monthly basket for each variant ---
    def totals(use_ma_for):
        """use_ma_for: set of statuses whose affected holds sell into the pop; else month-end (daily base)."""
        rets = []
        for mi, m in enumerate(months):
            wsum = rr = 0.0
            for pi, p in enumerate(m["picks"]):
                if p["ret"] is None:
                    continue
                w = p["weight"]
                r = p["ret"]
                if (mi, pi) in repl:
                    r = repl[(mi, pi)]["ma"] if repl[(mi, pi)]["status"] in use_ma_for else repl[(mi, pi)]["base"]
                wsum += w; rr += w * r
            if wsum > 0:
                rets.append(rr / wsum)
        return perf(rets)

    base_trace = perf([m["basket_ret"] for m in months])           # pure trace (authoritative 790.4)
    base_daily = totals(set())                                     # affected holds on daily month-end basis
    ann_exit = totals(ANN)                                         # sell into announced/completed pop
    pop_exit = totals(POP)                                         # + sell into rumored pop

    detail.sort(key=lambda x: -x["delta"])
    out = {"conf_min": CONF_MIN, "n_events": sum(len(v) for v in events.values()), "n_affected": len(affected),
           "affected_status": dict(stat),
           "results": {"baseline_trace": base_trace, "baseline_daily": base_daily,
                       "announced_exit": ann_exit, "announced_plus_rumored_exit": pop_exit},
           "detail": detail}
    OUT.write_text(json.dumps(out, indent=2, default=str))

    print("\n=== M&A ANNOUNCEMENT-EXIT A/B (same daily price basis; only the exit DATE differs) ===", flush=True)
    print(f"  {'variant':<28}{'total':>9}{'annual':>8}{'Sharpe':>8}{'DD':>8}", flush=True)
    for k, r in out["results"].items():
        print(f"  {k:<28}{r['total']:>8}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>7}%", flush=True)
    d_ann = ann_exit["total"] - base_daily["total"]; d_pop = pop_exit["total"] - base_daily["total"]
    print(f"\n  announced-exit vs month-end (same basis): {d_ann:+.1f}pp total, {ann_exit['sharpe']-base_daily['sharpe']:+.2f} Sharpe", flush=True)
    print(f"  +rumored-exit  vs month-end (same basis): {d_pop:+.1f}pp total", flush=True)
    print(f"\n  top pop-capture holds (ma vs month-end delta):", flush=True)
    for x in detail[:12]:
        print(f"    {x['date']} {x['ticker']:6} [{x['status']:9}] base {x['base_ret']*100:+6.1f}% -> pop {x['ma_ret']*100:+6.1f}%  "
              f"(delta {x['delta']*100:+5.1f}) {(x['acquirer'] or '')[:22]}", flush=True)
    print(f"\n  worst (pop-exit gave back vs month-end):", flush=True)
    for x in detail[-5:]:
        print(f"    {x['date']} {x['ticker']:6} [{x['status']:9}] base {x['base_ret']*100:+6.1f}% -> pop {x['ma_ret']*100:+6.1f}%  (delta {x['delta']*100:+5.1f})", flush=True)

    from core.models import BacktestResult
    from django.utils import timezone
    BacktestResult.objects.update_or_create(kind="ma_exit_study",
        defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
    print(f"\nSaved BacktestResult[ma_exit_study] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
