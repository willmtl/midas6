#!/usr/bin/env python3
"""CRISIS STRESS-TEST of the acceleration signal (the one regime we could never test). Our stock+fundamental
history is only ~5y (2021-26, one mild bear). The 9 original SPDR sector ETFs go back to Dec 1998, so we CAN
test the CORE sector-rotation SIGNAL through the dot-com crash (2000-02), the GFC (2007-09), and COVID (2020)
even though the stock-pick layer can't reach back. Question is NOT 'does ETF-rotation beat SPY' (we know it
doesn't, [[rotation-alpha]]) — it's: in a real deleveraging, does ranking sectors by ACCELERATION rotate you
DEFENSIVELY (cushioning the drawdown) or does it chase and get whipsawed (amplifying it)?
Pull long history via yfinance in-memory; rank 9 sectors by accel; hold top-K; compare RETURN and especially
DRAWDOWN vs SPY, full-period and inside each crisis window. -> BacktestResult[crisis_stress].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/crisis_backtest.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY"]   # 9 original SPDRs (Dec 1998)
BENCH = "SPY"
START = "1998-12-22"
CRISES = {
    "dotcom_2000_02": ("2000-03-01", "2002-10-31"),
    "gfc_2007_09":    ("2007-10-01", "2009-03-31"),
    "covid_2020":     ("2020-02-01", "2020-04-30"),
    "bear_2022":      ("2022-01-01", "2022-10-31"),
}


def _dd(returns):
    if not len(returns):
        return 0.0
    eq = np.cumprod(1 + np.asarray(returns, float))
    return float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0, n=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                dd=round(_dd(r), 1), win=round((r > 0).mean() * 100, 1), n=n)


def _load():
    import yfinance as yf
    cache = "/app/.data/crisis_prices.parquet"
    if os.path.exists(cache):
        px = pd.read_parquet(cache)
        print(f"loaded cached prices {px.index.min().date()}..{px.index.max().date()}", flush=True)
        return px
    print("downloading long history via yfinance...", flush=True)
    raw = yf.download(SECTORS + [BENCH], start=START, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(how="all")
    try:
        os.makedirs("/app/.data", exist_ok=True); close.to_parquet(cache)
    except Exception as e:
        print("cache write skipped:", e, flush=True)
    print(f"downloaded {close.shape[0]} rows {close.index.min().date()}..{close.index.max().date()}", flush=True)
    return close


def build():
    close = _load()
    monthly = close.resample("ME").last()
    sec = [s for s in SECTORS if s in monthly.columns]
    accel = monthly[sec].pct_change(3) - monthly[sec].pct_change(3).shift(3)
    fwd = monthly.pct_change().shift(-1)              # next-month return aligned at month t
    idx = monthly.index

    variants = {}
    series = {}
    for K in (2, 3, 4):
        rets, sprets, dts = [], [], []
        crisis_pick_log = {}
        for i in range(9, len(idx) - 1):
            date = idx[i]
            a = accel.loc[date].dropna()
            if len(a) < K or pd.isna(fwd.loc[date, BENCH]):
                continue
            top = a.sort_values(ascending=False).head(K).index
            r = fwd.loc[date, top].dropna()
            if not len(r):
                continue
            rets.append(float(r.mean())); sprets.append(float(fwd.loc[date, BENCH])); dts.append(idx[i + 1])
        s = pd.Series(rets, index=pd.DatetimeIndex(dts)); sp = pd.Series(sprets, index=pd.DatetimeIndex(dts))
        variants[f"accel_top{K}"] = _stats(rets, sprets)
        series[f"accel_top{K}"] = (s, sp)

    # equal-weight all 9 sectors + SPY buy-hold reference (aligned to accel_top3 dates)
    base_s, base_sp = series["accel_top3"]
    ew_rets = []
    for d in base_s.index:
        # d is next-month; map back to selection month
        pass
    # simpler: recompute EW over same loop window
    ew, ewsp, ewdt = [], [], []
    for i in range(9, len(idx) - 1):
        date = idx[i]
        if pd.isna(fwd.loc[date, BENCH]):
            continue
        r = fwd.loc[date, sec].dropna()
        if not len(r):
            continue
        ew.append(float(r.mean())); ewsp.append(float(fwd.loc[date, BENCH])); ewdt.append(idx[i + 1])
    variants["equal_weight_9"] = _stats(ew, ewsp)
    series["equal_weight_9"] = (pd.Series(ew, index=pd.DatetimeIndex(ewdt)), pd.Series(ewsp, index=pd.DatetimeIndex(ewdt)))

    print(f"\n=== FULL PERIOD ({idx.min().date()} .. {idx.max().date()}, {len(base_s)} months) ===", flush=True)
    for k in ("accel_top2", "accel_top3", "accel_top4", "equal_weight_9"):
        v = variants[k]
        print(f"  {k:15} total {v['total']:>8}%  vsSPY {v['vs_spy']:>8}  Sh {v['sharpe']:>5}  maxDD {v['dd']:>7}%  win {v['win']}%", flush=True)
    spy_full = _stats(base_sp.values, base_sp.values)
    print(f"  {'SPY (buy-hold)':15} total {spy_full['total']:>8}%  {'':>14} maxDD {spy_full['dd']:>7}%", flush=True)

    # ---- crisis windows: return + drawdown of accel_top3 vs SPY ----
    print(f"\n=== CRISIS WINDOWS — does acceleration cushion or amplify the crash? (accel_top3) ===", flush=True)
    s3, sp3 = series["accel_top3"]
    crisis_out = {}
    for name, (lo, hi) in CRISES.items():
        mask = (s3.index >= lo) & (s3.index <= hi)
        cs, csp = s3[mask].values, sp3[mask].values
        if not len(cs):
            print(f"  {name:16} (no data)", flush=True); continue
        strat_ret = float(np.prod(1 + cs) - 1) * 100; spy_ret = float(np.prod(1 + csp) - 1) * 100
        strat_dd, spy_dd = _dd(cs), _dd(csp)
        cushion = strat_dd - spy_dd     # positive = shallower (better) drawdown than SPY
        crisis_out[name] = dict(months=len(cs), strat_ret=round(strat_ret, 1), spy_ret=round(spy_ret, 1),
                                strat_dd=round(strat_dd, 1), spy_dd=round(spy_dd, 1), cushion_pp=round(cushion, 1))
        flag = "CUSHIONS" if cushion > 2 else ("AMPLIFIES" if cushion < -2 else "~matches")
        print(f"  {name:16} strat {strat_ret:>7.1f}% (DD {strat_dd:>6.1f}%)  vs SPY {spy_ret:>7.1f}% (DD {spy_dd:>6.1f}%)  "
              f"-> DD {flag} ({cushion:+.1f}pp)", flush=True)

    cushions = [c["cushion_pp"] for c in crisis_out.values()]
    avg_cushion = float(np.mean(cushions)) if cushions else 0.0
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"sectors": sec, "benchmark": BENCH, "start": str(idx.min().date()), "end": str(idx.max().date()),
                   "months": len(base_s)},
        "full_period": variants, "spy_full": spy_full, "crises": crisis_out, "avg_crisis_cushion_pp": round(avg_cushion, 1),
        "verdict": (f"Acceleration-rotation avg crisis DRAWDOWN cushion {avg_cushion:+.1f}pp vs SPY across "
                    f"{len(crisis_out)} crises. " + (
                    "It rotates DEFENSIVELY in crashes — shallower drawdowns than SPY. The core signal survives a "
                    "real deleveraging (the biggest untested risk); this materially raises confidence in the engine."
                    if avg_cushion > 2 else
                    "It does NOT reliably cushion crashes (drawdowns ~ or worse than SPY) — acceleration is a return "
                    "signal, not crisis protection; the strategy still needs an explicit bear/regime gate for a real "
                    "deleveraging." if avg_cushion < -2 else
                    "It roughly MATCHES SPY drawdowns in crises — neither a hedge nor a catastrophe; a pure sector "
                    "signal can't escape a market-wide crash, so crisis defense must come from an explicit gate.")),
        "caveat": "ETF-LEVEL signal only (the stock-pick alpha layer can't reach pre-2021, so absolute returns are NOT "
                  "the strategy's — only the SIGNAL's crisis behavior is being tested). 9 classic sectors (no XLRE/XLC/"
                  "thematic). yfinance adjusted closes, monthly, no fees.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path("/app/.data/studies/crisis_stress.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="crisis_stress", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[crisis_stress]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
