#!/usr/bin/env python3
"""REAL IMPLIED MOVE vs REALIZED for earnings, from Polygon option contract prices. For every option-liquid
earnings event: on t-1 (day before report) reconstruct the front-week ATM STRADDLE (nearest expiry on/after
report_date, strike nearest spot) = call_close + put_close from real daily bars -> implied_move% = straddle/spot.
Compare to REALIZED signed gap (close[t+1]/close[t-1]-1) and |gap|. This is the earnings vol-risk-premium:
VRP = |realized| - implied_move ( >0 => buyers win / straddle underpriced ; <0 => sellers win ).
Resumable: appends one JSON line per event to .data/studies/preearn_straddle.jsonl; rerun to continue; --summarize
to aggregate + persist BacktestResult[preearn_straddle]. Threaded Polygon fetch (backend has egress).
Run:  MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_straddle.py [--summarize]"""
import os, sys, json, datetime as dt
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR
from api.tasks import _polygon_get, _polygon_paginate

OUT = Path("/app/.data/studies/preearn_straddle.jsonl")
LIQ_MIN = 50e6                    # option-liquid floor ($ADV) — go a bit wider than 100M to size the small tier
EXP_MAXDAYS = 24                  # accept nearest expiry up to this many days after report (weeklies or 1 monthly)


def cap_bucket(mc):
    if mc is None or not np.isfinite(mc) or mc <= 0:
        return "cap?"
    b = mc / 1e9
    return "mega>200" if b >= 200 else "large10-200" if b >= 10 else "mid2-10" if b >= 2 else "small<2"


def liq_tier(dv):
    m = dv / 1e6
    return "ADV>500M" if m >= 500 else "ADV100-500M" if m >= 100 else "ADV50-100M"


def shares_ff(reps, tk, dates):
    r = reps.get(tk)
    if r is None or "shares_outstanding" not in r.columns:
        return None
    d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
    if d.empty:
        return None
    s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).values


def build_worklist():
    """DB-only: every option-liquid earnings event with spot(t-1), realized gap, cap+liq tier."""
    universe, _ = _universe()
    from core.models import EarningsEvent
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    work = []
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            sh = shares_ff(reps, tk, dates); di = dates.values
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t < 1 or t + 1 >= n:
                    continue
                spot = float(close[t - 1])
                if spot <= PRICE_FLOOR or not np.isfinite(spot):
                    continue
                dv = float(dvol20[t - 1]) if np.isfinite(dvol20[t - 1]) else 0.0
                if dv < LIQ_MIN:
                    continue
                realized = float(close[t + 1]) / spot - 1.0
                mc = (sh[t - 1] * spot) if (sh is not None and np.isfinite(sh[t - 1])) else None
                work.append(dict(tk=tk, report=str(pd.Timestamp(rd).date()), tm1=str(dates[t - 1].date()),
                                 spot=round(spot, 4), realized=round(realized, 5),
                                 cap=cap_bucket(mc), liq=liq_tier(dv)))
        done += len(ch)
        if done % 200 < 40:
            print(f"  worklist scan {done}/{len(names)}  events {len(work)}", flush=True)
    return work


def bar_close(opt_ticker, day):
    d = _polygon_get(f"/v2/aggs/ticker/{opt_ticker}/range/1/day/{day}/{day}", adjusted="true")
    if d and d.get("results"):
        return float(d["results"][0]["c"])
    return None


def fetch_straddle(w):
    """Return implied_move% (straddle/spot) for one event, or None. Adds exp/atm/dte to the record."""
    tk, tm1, report, spot = w["tk"], w["tm1"], w["report"], w["spot"]
    rd = dt.date.fromisoformat(report)
    lo, hi = round(spot * 0.90, 1), round(spot * 1.10, 1)
    contracts = _polygon_paginate(
        "/v3/reference/options/contracts", cap=2000, underlying_ticker=tk, as_of=tm1, limit=1000,
        **{"expiration_date.gte": report, "expiration_date.lte": str(rd + dt.timedelta(days=EXP_MAXDAYS)),
           "strike_price.gte": lo, "strike_price.lte": hi})
    if not contracts:
        return None
    exps = sorted(set(c["expiration_date"] for c in contracts))
    exp = exps[0]
    cands = [c for c in contracts if c["expiration_date"] == exp]
    strikes = sorted(set(c["strike_price"] for c in cands))
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - spot))
    call = next((c for c in cands if c["strike_price"] == atm and c["contract_type"] == "call"), None)
    put = next((c for c in cands if c["strike_price"] == atm and c["contract_type"] == "put"), None)
    if not call or not put:
        return None
    cc = bar_close(call["ticker"], tm1); pc = bar_close(put["ticker"], tm1)
    if cc is None or pc is None or cc <= 0 or pc <= 0:
        return None
    straddle = cc + pc
    dte = (dt.date.fromisoformat(exp) - dt.date.fromisoformat(tm1)).days
    return dict(implied=round(straddle / spot, 5), exp=exp, atm=atm, dte=dte,
                call_px=round(cc, 3), put_px=round(pc, 3))


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["tk"], r["report"]))
            except Exception:
                pass
    return done


def run_fetch(workers=10):
    work = build_worklist()
    print(f"\nTOTAL option-liquid earnings events: {len(work)}", flush=True)
    done = load_done()
    todo = [w for w in work if (w["tk"], w["report"]) not in done]
    print(f"already fetched: {len(done)} | to fetch: {len(todo)}", flush=True)
    fh = OUT.open("a"); got = miss = 0; t0 = pd.Timestamp.utcnow()

    def work_one(w):
        try:
            s = fetch_straddle(w)
        except Exception:
            s = None
        return w, s
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (w, s) in enumerate(ex.map(work_one, todo)):
            if s is None:
                miss += 1
                rec = {**w, "implied": None}
            else:
                got += 1
                rec = {**w, **s}
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 250 == 0:
                fh.flush()
                el = (pd.Timestamp.utcnow() - t0).total_seconds()
                print(f"  {i+1}/{len(todo)}  got {got} miss {miss}  {(i+1)/max(el,1):.1f}/s", flush=True)
    fh.close()
    print(f"DONE fetch: got {got}  miss {miss}", flush=True)


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["implied"].notna()].copy()
    df["implied_pct"] = df["implied"] * 100
    df["abs_real"] = df["realized"].abs() * 100
    df["vrp"] = df["abs_real"] - df["implied_pct"]          # |realized| - implied ; >0 buyers win
    df["year"] = df["report"].str[:4]
    print(f"\nEvents with real straddle: {len(df)}  ({df['report'].min()}..{df['report'].max()})", flush=True)

    def blk(sub, label):
        if len(sub) < 30:
            return None
        return dict(label=label, n=len(sub),
                    impl=round(sub["implied_pct"].mean(), 2), absreal=round(sub["abs_real"].mean(), 2),
                    vrp=round(sub["vrp"].mean(), 2), buyer_win=round((sub["vrp"] > 0).mean() * 100, 1),
                    med_impl=round(sub["implied_pct"].median(), 2), med_absreal=round(sub["abs_real"].median(), 2))

    out = {}
    print("\n=== IMPLIED (straddle) vs REALIZED |move| — the earnings vol-risk-premium ===", flush=True)
    print(f"  {'segment':16}{'n':>7}{'impl%':>8}{'|real|%':>9}{'VRP':>8}{'buyerWin%':>11}", flush=True)
    segs = [("ALL", df)]
    segs += [(f"liq:{t}", df[df["liq"] == t]) for t in ("ADV>500M", "ADV100-500M", "ADV50-100M")]
    segs += [(f"cap:{c}", df[df["cap"] == c]) for c in ("mega>200", "large10-200", "mid2-10", "small<2")]
    for label, sub in segs:
        b = blk(sub, label)
        if not b:
            continue
        out[label] = b
        print(f"  {label:16}{b['n']:>7}{b['impl']:>8.2f}{b['absreal']:>9.2f}{b['vrp']:>8.2f}{b['buyer_win']:>11.1f}", flush=True)

    print("\n=== BY YEAR (ALL liquid) ===", flush=True)
    yr = {}
    for y in sorted(df["year"].unique()):
        s = df[df["year"] == y]
        if len(s) < 30:
            continue
        yr[y] = dict(n=len(s), impl=round(s["implied_pct"].mean(), 2), absreal=round(s["abs_real"].mean(), 2),
                     vrp=round(s["vrp"].mean(), 2), buyer_win=round((s["vrp"] > 0).mean() * 100, 1))
        print(f"  {y}  n{len(s):>6}  impl {yr[y]['impl']:>5.2f}%  |real| {yr[y]['absreal']:>5.2f}%  "
              f"VRP {yr[y]['vrp']:>+5.2f}  buyerWin {yr[y]['buyer_win']:>4.1f}%", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), n_events=len(df),
                   window=[df["report"].min(), df["report"].max()], segments=out, by_year=yr,
                   caveat="Front-week ATM straddle (nearest expiry <= report+24d, strike nearest spot) priced from "
                   "real Polygon daily contract closes on t-1. implied_move = straddle/spot. VRP = |realized gap| - "
                   "implied. Buyer-win = P(|realized|>implied). Straddle ~= expected move (no 0.8 haircut applied).")
    Path("/app/.data/studies/preearn_straddle.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_straddle",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_straddle]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run_fetch(workers=int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 10)
        summarize()
