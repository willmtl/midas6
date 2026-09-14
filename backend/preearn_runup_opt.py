#!/usr/bin/env python3
"""RUN-UP-VIA-OPTIONS with REAL Polygon prices. The trade the user means: BUY the ATM option 1-2 weeks BEFORE
earnings (entry IV still low), then let IV RAMP into the print. Two exits:
  PRE  = sell at t-1 (day before print): harvest run-up drift + IV ramp, NO event risk, NO VRP, NO IV crush.
  POST = sell at t+1 (day after print): adds the realized event move MINUS the IV crush.
For each option-liquid earnings event (report>=2022-06, Polygon options-aggs coverage): pick expiry nearest
on/after report (<=report+24d); at entry t-K choose ATM strike nearest entry spot; price CALL and STRADDLE from
real daily contract closes at t-K, t-1, t+1. K in {5 (~1wk), 10 (~2wk)}. Net of per-side spread haircut.
Resumable JSONL. --summarize aggregates + BacktestResult[preearn_runup_opt].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_runup_opt.py [--summarize]"""
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

OUT = Path("/app/.data/studies/preearn_runup_opt.jsonl")
LIQ_MIN = 100e6
KS = [5, 10]                     # trading days before the print (≈1wk, ≈2wk)
EXP_MAXDAYS = 24
HAIRCUT = 0.03                   # per-side half-spread (fraction of premium)
COV_START = "2022-06-01"         # Polygon options-aggs coverage floor on our plan

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
    work = []; done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values; di = dates.values
            for rd in edates.get(tk, []):
                t = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if t - max(KS) < 0 or t + 1 >= n:
                    continue
                if not (np.isfinite(dvol20[t - 1]) and dvol20[t - 1] >= LIQ_MIN):
                    continue
                sp = {str(K): (round(float(close[t - K]), 4), str(dates[t - K].date())) for K in KS}
                if any(v[0] <= PRICE_FLOOR for v in sp.values()):
                    continue
                work.append(dict(tk=tk, report=str(pd.Timestamp(rd).date()), t1=str(dates[t - 1].date()),
                                 tp1=str(dates[t + 1].date()), entry=sp, liq=liq_tier(float(dvol20[t - 1]))))
        done += len(ch)
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


def contracts_asof(tk, asof, report):
    rd = dt.date.fromisoformat(report)
    return _polygon_paginate("/v3/reference/options/contracts", cap=2000, underlying_ticker=tk, as_of=asof,
        limit=1000, **{"expiration_date.gte": report,
                       "expiration_date.lte": str(rd + dt.timedelta(days=EXP_MAXDAYS))})


def net_ret(entry_px, exit_px):
    ecost = entry_px * (1 + HAIRCUT); eval_ = exit_px * (1 - HAIRCUT)
    return None if ecost <= 0 else round((eval_ - ecost) / ecost, 5)


def fetch_event(w):
    """For each K, price CALL and STRADDLE at entry(t-K), pre(t-1), post(t+1). Returns dict K->metrics."""
    tk, report, t1, tp1 = w["tk"], w["report"], w["t1"], w["tp1"]
    res = {}
    for K in KS:
        spot, ed = w["entry"][str(K)]
        try:
            cons = contracts_asof(tk, ed, report)
        except Exception:
            cons = None
        if not cons:
            continue
        exps = sorted(set(c["expiration_date"] for c in cons))
        exp = exps[0]
        cand = [c for c in cons if c["expiration_date"] == exp]
        strikes = sorted(set(c["strike_price"] for c in cand))
        if not strikes:
            continue
        atm = min(strikes, key=lambda k: abs(k - spot))
        call = next((c for c in cand if c["strike_price"] == atm and c["contract_type"] == "call"), None)
        put = next((c for c in cand if c["strike_price"] == atm and c["contract_type"] == "put"), None)
        if not call or not put:
            continue
        ct, pt = call["ticker"], put["ticker"]
        c_e, p_e = bar_close(ct, ed), bar_close(pt, ed)
        c_1, p_1 = bar_close(ct, t1), bar_close(pt, t1)
        c_2, p_2 = bar_close(ct, tp1), bar_close(pt, tp1)
        if not all(x and x > 0 for x in (c_e, p_e, c_1, p_1, c_2, p_2)):
            continue
        res[str(K)] = dict(
            atm=atm, exp=exp, dte_entry=(dt.date.fromisoformat(exp) - dt.date.fromisoformat(ed)).days,
            call_pre=net_ret(c_e, c_1), call_post=net_ret(c_e, c_2),
            strad_pre=net_ret(c_e + p_e, c_1 + p_1), strad_post=net_ret(c_e + p_e, c_2 + p_2),
            iv_ramp_proxy=round((c_1 + p_1) / (c_e + p_e) - 1, 4))   # straddle value change pre-event (drift+ramp-theta)
    return res


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
    print(f"already done {len(done)} | to fetch {len(todo)}", flush=True)
    fh = OUT.open("a"); got = miss = 0; t0 = pd.Timestamp.utcnow()

    def one(w):
        try:
            r = fetch_event(w)
        except Exception:
            r = {}
        return w, r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (w, r) in enumerate(ex.map(one, todo)):
            rec = {"tk": w["tk"], "report": w["report"], "liq": w["liq"], "K": r}
            fh.write(json.dumps(rec) + "\n")
            got += 1 if r else 0; miss += 0 if r else 1
            if (i + 1) % 200 == 0:
                fh.flush(); el = (pd.Timestamp.utcnow() - t0).total_seconds()
                print(f"  {i+1}/{len(todo)}  ok {got} miss {miss}  {(i+1)/max(el,1):.1f}/s  cache {len(_bar_cache)}", flush=True)
    fh.close()
    print(f"DONE ok {got} miss {miss}", flush=True)


def summarize():
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    recs = []
    for r in rows:
        for K, m in (r.get("K") or {}).items():
            recs.append(dict(tk=r["tk"], report=r["report"], liq=r["liq"], K=int(K), **m))
    df = pd.DataFrame(recs)
    if df.empty:
        print("no data yet", flush=True); return
    df["year"] = df["report"].str[:4]
    print(f"\nEvents priced: {df['report'].nunique()} reports, {len(df)} (event×K) rows "
          f"({df['report'].min()}..{df['report'].max()})", flush=True)

    def desc(a):
        a = np.asarray(a, float) * 100; a = a[np.isfinite(a)]
        if len(a) < 20:
            return None
        return dict(n=len(a), mean=round(a.mean(), 2), med=round(float(np.median(a)), 2),
                    win=round((a > 0).mean() * 100, 1), p10=round(float(np.percentile(a, 10)), 1),
                    p90=round(float(np.percentile(a, 90)), 1))
    out = {}
    for K in KS:
        sub = df[df["K"] == K]
        print(f"\n=== BUY {K} trading days before (net {int(HAIRCUT*100)}%/side), %/trade ===", flush=True)
        print(f"  {'leg/exit':16}{'n':>6}{'mean':>8}{'med':>8}{'win%':>7}{'p10':>7}{'p90':>7}", flush=True)
        for col in ("call_pre", "call_post", "strad_pre", "strad_post", "iv_ramp_proxy"):
            d = desc(sub[col])
            if not d:
                continue
            out[f"K{K}|{col}"] = d
            print(f"  {col:16}{d['n']:>6}{d['mean']:>8.2f}{d['med']:>8.2f}{d['win']:>7.1f}{d['p10']:>7.1f}{d['p90']:>7.1f}", flush=True)

    # by-year for the key play: K=10 straddle sold pre-print (pure ramp harvest)
    print(f"\n=== BY YEAR — K=10 STRADDLE sold PRE-print (ramp harvest) ===", flush=True)
    yr = {}
    s10 = df[df["K"] == 10]
    for y in sorted(s10["year"].unique()):
        a = s10[s10["year"] == y]["strad_pre"].dropna().values * 100
        if len(a) < 15:
            continue
        yr[y] = dict(n=len(a), mean=round(a.mean(), 2), win=round((a > 0).mean() * 100, 1))
        print(f"  {y}  n{len(a):>5}  mean {a.mean():>+6.2f}%  win {(a>0).mean()*100:>4.1f}%", flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), ks=KS, haircut=HAIRCUT, cov_start=COV_START,
                   n_rows=len(df), cells=out, by_year_k10_strad_pre=yr,
                   caveat="Real Polygon daily contract closes. Buy ATM (nearest expiry <=report+24d) at t-K; exits "
                   "PRE=t-1, POST=t+1. Net per-side haircut. PRE avoids event+IV-crush (harvests drift+IV ramp). "
                   "Coverage 2022-06+ (older aggs 403 on plan).")
    Path("/app/.data/studies/preearn_runup_opt.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_runup_opt",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_runup_opt]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    if "--summarize" in sys.argv:
        summarize()
    else:
        run_fetch()
        summarize()
