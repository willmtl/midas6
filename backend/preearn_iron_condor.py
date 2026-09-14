#!/usr/bin/env python3
"""EARNINGS IRON CONDOR backtest with REAL Polygon option prices — the DEFINED-RISK way to harvest the earnings
vol-risk-premium (implied>realized). On t-1 (day before print, peak IV) sell an OTM put spread + OTM call spread
on the nearest expiry on/after earnings; hold to expiry (intrinsic payoff from real spot). Short strikes placed at
spot +/- m * expected_move (EM = ATM straddle / spot); wings WING*EM further OTM. P&L = credit - spread losses at
expiry; return-on-risk = P&L / (max spread width - credit). Reports win%, mean/median ROR, expectancy, by-year, per
m in {1.0, 1.5}. Gross (close prices) AND net (per-leg spread haircut). Resumable JSONL; --summarize aggregates.
Coverage 2022-06+ (older option aggs 403). Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_iron_condor.py [--summarize]"""
import os, sys, json, datetime as dt
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from seq_fundamental_study import load_candles, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR
from api.tasks import _polygon_get, _polygon_paginate

OUT = Path("/app/.data/studies/preearn_iron_condor.jsonl")
LIQ_MIN = 100e6
EXP_MAXDAYS = 24
COV_START = "2022-06-01"
M_GRID = [1.0, 1.5]              # short strikes at spot +/- m*EM
WING = 0.5                       # wing WING*EM beyond the short strike
LEG_HAIRCUT = 0.05              # per-leg spread cost (fraction of that leg's premium); OTM => wider

_bar_cache = {}; _lock = Lock()


def liq_tier(dv):
    m = dv / 1e6
    return "ADV>500M" if m >= 500 else "ADV100-500M" if m >= 100 else "ADV<100M"


def build_worklist():
    universe, _ = _universe()
    from core.models import EarningsEvent
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.filter(report_date__gte=COV_START).values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    work = []
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values; di = dates.values
            # spot series for expiry lookup
            cser = pd.Series(close, index=dates)
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t < 1 or t + 1 >= n:
                    continue
                spot = float(close[t - 1])
                if spot <= PRICE_FLOOR or not np.isfinite(spot):
                    continue
                if not (np.isfinite(dvol20[t - 1]) and dvol20[t - 1] >= LIQ_MIN):
                    continue
                work.append(dict(tk=tk, report=str(pd.Timestamp(rd).date()), t1=str(dates[t - 1].date()),
                                 spot=round(spot, 4), liq=liq_tier(float(dvol20[t - 1]))))
    return work


def bar_close(opt_ticker, day):
    key = (opt_ticker, day)
    with _lock:
        if key in _bar_cache:
            return _bar_cache[key]
    d = _polygon_get(f"/v2/aggs/ticker/{opt_ticker}/range/1/day/{day}/{day}", adjusted="true")
    v = float(d["results"][0]["c"]) if d and d.get("results") else None
    with _lock:
        _bar_cache[key] = v
    return v


def spot_on(tk, day):
    """real spot at expiry from candles (as-of; nearest <= day)."""
    from core.models import Candle
    c = Candle.objects.filter(ticker=tk, interval="1d", date__lte=day).order_by("-date").values_list("close", flat=True).first()
    return float(c) if c is not None else None


def fetch_event(w):
    tk, report, t1, spot = w["tk"], w["report"], w["t1"], w["spot"]
    rd = dt.date.fromisoformat(report)
    cons = _polygon_paginate("/v3/reference/options/contracts", cap=3000, underlying_ticker=tk, as_of=t1,
        limit=1000, **{"expiration_date.gte": report, "expiration_date.lte": str(rd + dt.timedelta(days=EXP_MAXDAYS))})
    if not cons:
        return None
    exps = sorted(set(c["expiration_date"] for c in cons))
    exp = exps[0]
    cand = [c for c in cons if c["expiration_date"] == exp]
    calls = {c["strike_price"]: c["ticker"] for c in cand if c["contract_type"] == "call"}
    puts = {c["strike_price"]: c["ticker"] for c in cand if c["contract_type"] == "put"}
    if not calls or not puts:
        return None
    cstr = sorted(calls); pstr = sorted(puts)
    # EM from ATM straddle
    atm = min(set(cstr) & set(pstr), key=lambda k: abs(k - spot), default=None)
    if atm is None:
        return None
    ac, ap = bar_close(calls[atm], t1), bar_close(puts[atm], t1)
    if not ac or not ap or ac <= 0 or ap <= 0:
        return None
    em = (ac + ap) / spot                          # expected-move fraction
    exp_spot = spot_on(tk, exp)
    if exp_spot is None:
        return None
    out = {"exp": exp, "em": round(em, 5), "atm": atm, "exp_spot": round(exp_spot, 4), "legs": {}}

    def nearest(strikes, target):
        return min(strikes, key=lambda k: abs(k - target))
    for m in M_GRID:
        sp_k = nearest(pstr, spot * (1 - m * em))          # short put
        lp_k = nearest(pstr, spot * (1 - (m + WING) * em))  # long put (further OTM, lower)
        sc_k = nearest(cstr, spot * (1 + m * em))          # short call
        lc_k = nearest(cstr, spot * (1 + (m + WING) * em))  # long call (further OTM, higher)
        if lp_k >= sp_k or lc_k <= sc_k:                    # degenerate (strike spacing too coarse)
            continue
        sp_px = bar_close(puts[sp_k], t1); lp_px = bar_close(puts[lp_k], t1)
        sc_px = bar_close(calls[sc_k], t1); lc_px = bar_close(calls[lc_k], t1)
        if None in (sp_px, lp_px, sc_px, lc_px):
            continue
        out["legs"][str(m)] = dict(sp_k=sp_k, lp_k=lp_k, sc_k=sc_k, lc_k=lc_k,
                                   sp=sp_px, lp=lp_px, sc=sc_px, lc=lc_px)
    return out if out["legs"] else None


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                r = json.loads(line); done.add((r["tk"], r["report"]))
            except Exception:
                pass
    return done


def run_fetch(workers=12):
    work = build_worklist()
    print(f"option-liquid events (report>={COV_START}): {len(work)}", flush=True)
    done = load_done()
    todo = [w for w in work if (w["tk"], w["report"]) not in done]
    print(f"already {len(done)} | to fetch {len(todo)}", flush=True)
    fh = OUT.open("a"); ok = miss = 0; t0 = pd.Timestamp.utcnow()

    def one(w):
        try:
            r = fetch_event(w)
        except Exception:
            r = None
        return w, r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (w, r) in enumerate(ex.map(one, todo)):
            rec = {"tk": w["tk"], "report": w["report"], "spot": w["spot"], "liq": w["liq"], "r": r}
            fh.write(json.dumps(rec) + "\n"); ok += 1 if r else 0; miss += 0 if r else 1
            if (i + 1) % 200 == 0:
                fh.flush(); el = (pd.Timestamp.utcnow() - t0).total_seconds()
                print(f"  {i+1}/{len(todo)}  ok {ok} miss {miss}  {(i+1)/max(el,1):.1f}/s  cache {len(_bar_cache)}", flush=True)
    fh.close()
    print(f"DONE ok {ok} miss {miss}", flush=True)


def condor_pnl(spot, exp_spot, lg, haircut):
    """return-on-risk of the iron condor held to expiry, per share. haircut reduces received credit."""
    credit = (lg["sp"] + lg["sc"]) - (lg["lp"] + lg["lc"])
    # net of per-leg spread: you sell shorts cheaper, buy longs dearer
    credit_net = (lg["sp"] * (1 - haircut) + lg["sc"] * (1 - haircut)) - (lg["lp"] * (1 + haircut) + lg["lc"] * (1 + haircut))
    c = credit if haircut == 0 else credit_net
    if c <= 0:
        return None
    put_loss = max(0.0, lg["sp_k"] - exp_spot) - max(0.0, lg["lp_k"] - exp_spot)
    call_loss = max(0.0, exp_spot - lg["sc_k"]) - max(0.0, exp_spot - lg["lc_k"])
    pnl = c - put_loss - call_loss
    put_w = lg["sp_k"] - lg["lp_k"]; call_w = lg["lc_k"] - lg["sc_k"]
    max_risk = max(put_w, call_w) - c
    if max_risk <= 0:
        return None
    return pnl / max_risk, c / max(put_w, call_w)     # ROR, credit-as-frac-of-width


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    recs = []
    for row in rows:
        r = row.get("r")
        if not r:
            continue
        for m, lg in r.get("legs", {}).items():
            for hc, tag in ((0.0, "gross"), (LEG_HAIRCUT, "net")):
                res = condor_pnl(row["spot"], r["exp_spot"], lg, hc)
                if res is None:
                    continue
                ror, credfrac = res
                recs.append(dict(tk=row["tk"], report=row["report"], liq=row["liq"], m=float(m),
                                 cost=tag, ror=ror, credfrac=credfrac, year=row["report"][:4]))
    df = pd.DataFrame(recs)
    if df.empty:
        print("no data yet", flush=True); return
    print(f"\nIron-condor trades priced: {df[df.cost=='net'].report.nunique()} reports "
          f"({df.report.min()}..{df.report.max()})", flush=True)

    def blk(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        if len(a) < 20:
            return None
        return dict(n=len(a), win=round((a > 0).mean() * 100, 1), mean=round(a.mean() * 100, 1),
                    med=round(float(np.median(a)) * 100, 1), p10=round(float(np.percentile(a, 10)) * 100, 1))
    out = {}
    print("\n=== IRON CONDOR held-to-expiry — return-on-risk %/trade ===", flush=True)
    print(f"  {'strikes':10}{'cost':6}{'n':>6}{'win%':>7}{'meanROR':>9}{'medROR':>8}{'p10':>8}{'credit%w':>9}", flush=True)
    for m in M_GRID:
        for tag in ("gross", "net"):
            sub = df[(df.m == m) & (df.cost == tag)]
            b = blk(sub["ror"])
            if not b:
                continue
            cf = round(sub["credfrac"].mean() * 100, 1)
            out[f"m{m}|{tag}"] = {**b, "credit_frac_width": cf}
            print(f"  ±{m:.1f}EM   {tag:6}{b['n']:>6}{b['win']:>7.1f}{b['mean']:>9.1f}{b['med']:>8.1f}{b['p10']:>8.1f}{cf:>9.1f}", flush=True)

    print("\n=== BY YEAR — ±1.0EM NET ===", flush=True)
    yr = {}
    s = df[(df.m == 1.0) & (df.cost == "net")]
    for y in sorted(s.year.unique()):
        a = s[s.year == y]["ror"].values
        if len(a) < 15:
            continue
        yr[y] = dict(n=len(a), win=round((a > 0).mean() * 100, 1), mean=round(float(np.mean(a)) * 100, 1))
        print(f"  {y}  n{len(a):>5}  win {yr[y]['win']:>5.1f}%  meanROR {yr[y]['mean']:>+6.1f}%", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), m_grid=M_GRID, wing=WING,
                   leg_haircut=LEG_HAIRCUT, cov_start=COV_START, cells=out, by_year_m1_net=yr,
                   caveat="Iron condor sold t-1 (peak IV), held to expiry; short strikes at spot+/-m*EM (EM=ATM "
                   "straddle/spot), wings WING*EM further OTM; strikes snapped to listed. ROR=P&L/(width-credit). "
                   "Real Polygon closes; net = per-leg 5% spread haircut. Coverage 2022-06+.")
    Path("/app/.data/studies/preearn_iron_condor.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_iron_condor",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_iron_condor]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run_fetch()
        summarize()
