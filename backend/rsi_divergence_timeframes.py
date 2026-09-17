#!/usr/bin/env python3
"""RSI(14) bullish-DIVERGENCE bounce book run on the DAILY, WEEKLY, and MONTHLY candle (user: "test that on
daily/weekly/monthly"). Same divergence rule at each timeframe — price makes a 20-bar lower-low while RSI(14)
makes a higher low (rsi > rsi.shift(10)) — computed on THAT timeframe's candle. EW long book, hold 1 & 2 bars,
survivorship-aware (covered u delisted, through delisting), split-adjusted daily close, $5M/day + >$5 liquidity.

Honors the hard rules: STANDALONE selector with its own $5M floor (this book), reported GROSS + NET (10/25/50
bps/side) since the edge is short-lived so costs/turnover are make-or-break, by regime (all/bear/bull) vs SPY,
and SEGMENTED by cap bucket (micro/small/large) at each timeframe + a cap x sector grid for the best timeframe.
Saves BacktestResult[rsi_divergence_timeframes] + JSON.

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rsi_divergence_timeframes.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from core.models import Candle, Sector, DelistedCompany, Fundamental

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0
LOW_WIN = 20        # price N-bar lower-low window (bars, per timeframe)
RSI_LAG = 10        # RSI compared to its value LAG bars ago (higher low)
START = "2014-06-01"
OUT = "/app/.data/studies/rsi_divergence_timeframes.json"
TF = {"daily": ("D", 252), "weekly": ("W-FRI", 52), "monthly": ("ME", 12)}

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        try:
            tk = json.loads(line).get("ticker")
            if tk and "." not in tk and tk not in etfs:
                covered.add(tk)
        except Exception:
            pass
delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
universe = sorted((covered | delisted) - etfs)
print(f"universe {len(universe)} ({len(delisted)} delisted)", flush=True)

# static cap + sector snapshot for segmentation (caveat: not PIT — acceptable for a bounce signal, cap growth
# is not the edge here; flagged in the payload)
mcap = dict(Fundamental.objects.filter(ticker__in=universe).values_list("ticker", "market_cap"))
fsec = dict(Fundamental.objects.filter(ticker__in=universe).values_list("ticker", "sector"))


def cap_bucket(tk):
    mc = mcap.get(tk)
    if not mc or mc <= 0:
        return "unknown"
    return "micro(<0.5B)" if mc < 5e8 else "small(<2B)" if mc < 2e9 else "large(>=2B)"


spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte=START).values_list("date", "close")),
                   columns=["date", "close"]); spy["date"] = pd.to_datetime(spy["date"])
spy_c = spy.set_index("date")["close"].astype(float).sort_index()
spy_bull_d = spy_c > spy_c.rolling(200).mean()


def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---- load daily candles ONCE, derive per-timeframe fired-signal + close panels ----
# per timeframe: FIRED[tf] (bool [idx x tk]), CL[tf] (through-delisting close), fwd1/fwd2
daily_close, daily_dvol = {}, {}
for i in range(0, len(universe), 300):
    rows = Candle.objects.filter(ticker__in=universe[i:i + 300], date__gte=START).values_list("ticker", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float); df["volume"] = df["volume"].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        c = s["close"]
        if c.notna().sum() < 300:
            continue
        daily_close[tk] = c.ffill(limit=252)                       # through-delisting daily close
        daily_dvol[tk] = (c * s["volume"]).rolling(20, min_periods=10).mean()
    print(f"  loaded {min(i+300,len(universe))}/{len(universe)}", flush=True)

names = sorted(daily_close)
print(f"names with history {len(names)}", flush=True)

panels = {}     # tf -> dict(FIRED, CL, fwd1, fwd2, idx, spy_bull, spy_fwd1)
for tfname, (rule, _ppy) in TF.items():
    daily_tf = (tfname == "daily")
    if daily_tf:
        idx = spy_c.index                              # NATIVE trading days (calendar "D" resample injects
        def agg(s):                                    # weekend NaNs that break RSI -> 0 fires artifact)
            return s.reindex(idx)
    else:
        idx = spy_c.resample(rule).last().index
        def agg(s, _r=rule):
            return s.resample(_r).last().reindex(idx)
    spy_tf = agg(spy_c)
    spy_bull = agg(spy_bull_d).fillna(False)
    spy_fwd1 = spy_tf.shift(-1) / spy_tf - 1.0
    FIRED, CL = {}, {}
    for tk in names:
        c_tf = agg(daily_close[tk])
        if c_tf.notna().sum() < max(30, LOW_WIN + RSI_LAG + 5):
            continue
        r = rsi(c_tf, 14)
        bulldiv = (c_tf <= c_tf.rolling(LOW_WIN, min_periods=max(10, LOW_WIN - 5)).min()) & (r > r.shift(RSI_LAG))
        dvol_tf = agg(daily_dvol[tk])
        liq = (dvol_tf >= DVOL_FLOOR) & (c_tf > PRICE_FLOOR)
        FIRED[tk] = (bulldiv & liq).fillna(False)
        CL[tk] = c_tf
    FIRED = pd.DataFrame(FIRED).fillna(False); CL = pd.DataFrame(CL).reindex(columns=FIRED.columns)
    fwd1 = CL.shift(-1) / CL - 1.0; fwd2 = CL.shift(-2) / CL - 1.0
    panels[tfname] = dict(FIRED=FIRED, CL=CL, fwd1=fwd1, fwd2=fwd2, idx=idx, spy_bull=spy_bull, spy_fwd1=spy_fwd1)
    print(f"[{tfname}] bars {idx[0].date()}..{idx[-1].date()} names {FIRED.shape[1]} total fires {int(FIRED.values.sum())}", flush=True)


def run_book(P, regime, hold, cost_bps, cols=None):
    FIRED, spy_bull = P["FIRED"], P["spy_bull"]; idx = P["idx"]
    fwd = P["fwd1"] if hold == 1 else P["fwd2"]
    if cols is not None:
        keep = [c for c in cols if c in FIRED.columns]
        FIRED = FIRED[keep]; fwd = fwd[keep]
    pr, turns, held, prev = [], [], [], set()
    for d in idx[:-hold]:
        if regime == "bear" and bool(spy_bull.get(d, False)):
            pr.append(0.0); held.append(0); continue
        if regime == "bull" and not bool(spy_bull.get(d, False)):
            pr.append(0.0); held.append(0); continue
        row = FIRED.loc[d].astype(bool); frow = fwd.loc[d]
        nm = [t for t in row.index if bool(row[t]) and pd.notna(frow[t])]
        if len(nm) < 5:
            pr.append(0.0); held.append(0); prev = set(); continue
        r = fwd.loc[d, nm].clip(-1.0, 1.5).mean()
        cur = set(nm); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
        pr.append(float(r) - turn * 2 * cost_bps / 1e4 / hold); turns.append(turn); prev = cur
        held.append(len(nm))
    return pd.Series(pr, index=idx[:-hold]), (np.mean(turns) if turns else 0), (np.mean([h for h in held if h]) if any(held) else 0)


def stats(pr, ppy, bench=None):
    pr = pr.dropna()
    if len(pr) < max(12, ppy // 4):
        return {"n": int(len(pr))}
    eq = np.cumprod(1 + pr.values); yrs = len(pr) / ppy
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = pr.mean() / pr.std() * math.sqrt(ppy) if pr.std() > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100); hit = float((pr.values > 0).mean() * 100)
    o = {"n": int(len(pr)), "total_pct": float((eq[-1] - 1) * 100), "cagr_pct": float(cagr),
         "sharpe": float(sh), "maxdd_pct": dd, "hit_pct": hit}
    if bench is not None:
        b = bench.reindex(pr.index).fillna(0.0)
        o["vs_spy_pp"] = float(((eq[-1] - 1) - (np.cumprod(1 + b.values)[-1] - 1)) * 100)
    return o


res = {"params": {"low_win": LOW_WIN, "rsi_lag": RSI_LAG, "dvol_floor": DVOL_FLOOR, "price_floor": PRICE_FLOOR,
                  "start": START, "note": "same bullish-divergence rule per timeframe candle; static cap/sector snapshot (caveat)"},
       "by_timeframe": {}, "by_cap": {}, "cap_x_sector": {}}

print("\n=== RSI(14) bullish-divergence book by TIMEFRAME (GROSS then NET of costs) ===", flush=True)
for tfname, (rule, ppy) in TF.items():
    P = panels[tfname]
    res["by_timeframe"][tfname] = {"total_fires": int(P["FIRED"].values.sum()), "holds": {}}
    print(f"\n-- {tfname.upper()} candle (bars/yr {ppy}) --", flush=True)
    for hold in (1, 2):
        res["by_timeframe"][tfname]["holds"][hold] = {}
        for regime in ("all", "bear", "bull"):
            g, tn, nm = run_book(P, regime, hold, 0)
            sg = stats(g, ppy, bench=P["spy_fwd1"] if regime == "all" else None); sg["turn_pct"] = float(tn * 100); sg["avg_names"] = float(nm)
            cells = {"gross": sg}
            for cb in (10, 25, 50):
                sc, _, _ = run_book(P, regime, hold, cb)
                cells[f"net_{cb}"] = stats(sc, ppy, bench=P["spy_fwd1"] if regime == "all" else None)
            res["by_timeframe"][tfname]["holds"][hold][regime] = cells
            n25 = cells["net_25"]
            print(f"  hold{hold} {regime:4} GROSS CAGR {sg.get('cagr_pct',0):+6.1f}% Sh {sg.get('sharpe',0):4.2f} hit {sg.get('hit_pct',0):4.1f}% turn {sg['turn_pct']:3.0f}% n~{nm:.0f}"
                  f"  | NET25 CAGR {n25.get('cagr_pct',0):+6.1f}% Sh {n25.get('sharpe',0):4.2f}", flush=True)

# SPY reference per timeframe
for tfname, (rule, ppy) in TF.items():
    sref = stats(panels[tfname]["spy_fwd1"], ppy)
    res.setdefault("spy_ref", {})[tfname] = sref
    print(f"  SPY buy&hold [{tfname}] CAGR {sref.get('cagr_pct',0):+.1f}% Sh {sref.get('sharpe',0):.2f}", flush=True)

# ---- cap-bucket segmentation (all-regime, hold1, GROSS + NET25) per timeframe ----
print("\n=== by CAP bucket (all regime, hold1) ===", flush=True)
buckets = ["micro(<0.5B)", "small(<2B)", "large(>=2B)"]
by_cap_cols = {b: [t for t in names if cap_bucket(t) == b] for b in buckets}
for tfname, (rule, ppy) in TF.items():
    P = panels[tfname]; res["by_cap"][tfname] = {}
    for b in buckets:
        cols = by_cap_cols[b]
        if len(cols) < 20:
            continue
        g, tn, nm = run_book(P, "all", 1, 0, cols=cols); n25c, _, _ = run_book(P, "all", 1, 25, cols=cols)
        sg = stats(g, ppy, bench=P["spy_fwd1"]); s25 = stats(n25c, ppy, bench=P["spy_fwd1"])
        res["by_cap"][tfname][b] = {"n_names": len(cols), "gross": sg, "net_25": s25, "avg_fires": float(nm)}
        print(f"  [{tfname:7}] {b:12} n={len(cols):4} GROSS CAGR {sg.get('cagr_pct',0):+6.1f}% Sh {sg.get('sharpe',0):4.2f}"
              f"  NET25 CAGR {s25.get('cagr_pct',0):+6.1f}% Sh {s25.get('sharpe',0):4.2f}  vsSPY {s25.get('vs_spy_pp',0):+.0f}pp", flush=True)

# ---- cap x sector grid for the BEST timeframe (by all/hold1/net25 Sharpe) ----
best_tf = max(TF, key=lambda t: (res["by_timeframe"][t]["holds"][1]["all"]["net_25"].get("sharpe", -9)))
print(f"\n=== cap x SECTOR grid for BEST timeframe = {best_tf} (all regime, hold1, NET25) ===", flush=True)
P = panels[best_tf]; ppy = TF[best_tf][1]
secs = sorted({fsec.get(t) for t in names if fsec.get(t)})
for b in buckets:
    res["cap_x_sector"].setdefault(b, {})
    for sec in secs:
        cols = [t for t in names if cap_bucket(t) == b and fsec.get(t) == sec]
        if len(cols) < 15:
            continue
        s25, _, nm = run_book(P, "all", 1, 25, cols=cols)
        st = stats(s25, ppy, bench=P["spy_fwd1"])
        if st.get("n", 0) < ppy // 2 or nm < 2:
            continue
        res["cap_x_sector"][b][sec] = {"n_names": len(cols), "net_25": st, "avg_fires": float(nm)}
        print(f"  {b:12} {str(sec)[:22]:22} n={len(cols):4} NET25 CAGR {st.get('cagr_pct',0):+6.1f}% Sh {st.get('sharpe',0):4.2f} vsSPY {st.get('vs_spy_pp',0):+.0f}pp", flush=True)

Path(OUT).write_text(json.dumps(res, indent=2, default=float))
print(f"\nwrote {OUT}", flush=True)
try:
    from core.models import BacktestResult
    from django.utils import timezone
    BacktestResult.objects.update_or_create(kind="rsi_divergence_timeframes",
        defaults={"payload": res, "computed_at": timezone.now()})
    print("Saved BacktestResult[rsi_divergence_timeframes]", flush=True)
except Exception as e:
    print("save skipped:", e, flush=True)
