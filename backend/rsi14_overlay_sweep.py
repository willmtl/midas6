#!/usr/bin/env python3
"""Layer overlays on RSI(14) suppressed-then-cross (large-cap) and see which turn the
tail-driven bear-only edge into a RELIABLE per-trade beat of SPY (esp. in BULL regime).

Base = RSI(14) held <50 & <its SMA for >=10 consec days, then cross up; top-500 mktcap.
Metric per overlay = excess over SPY (stock fwd 6m - SPY fwd 6m, matched windows), split
all/bull/bear, plus beat-SPY%. Candles come from the /tmp/beat_candles.pkl cache (instant,
avoids the DB-fragile hypertable load). Fundamentals are POINT-IN-TIME (latest report with
avail_date <= entry). -> BacktestResult[rsi14_overlay_sweep] + JSON.

Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/rsi14_overlay_sweep.py
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi14_overlay_sweep.json"
CACHE = Path("/tmp/beat_candles.pkl")
MIN_STREAK, LEVEL, HOLD, TOP_N = 10, 50, 126, 500
PRICE_FLOOR, DVOL_FLOOR = 5.0, 5_000_000.0


def suppressed_then_cross(close, win):
    rsi = ta.momentum.rsi(close, window=win)
    sma = rsi.rolling(win).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    suppressed = (rsi < LEVEL) & (rsi < sma)
    grp = (~suppressed).cumsum()
    streak = suppressed.groupby(grp).cumsum()
    return (cross & (streak.shift(1) >= MIN_STREAK)).fillna(False).values


def episode_dedup(idxs, gap=HOLD):
    out, last = [], -10**9
    for i in sorted(idxs):
        if i - last >= gap:
            out.append(i); last = i
    return out


def robust(fn, tries=6):
    from django.db import connection
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            connection.close(); time.sleep(4 * (a + 1))
    return None


# ---- overlays: each maps a per-entry context dict -> bool (True = keep the trade) ----
def _has(c, *keys):
    return all(c.get(k) is not None and np.isfinite(c[k]) for k in keys)

OVERLAYS = {
    "base":            lambda c: True,
    "roe_pos":         lambda c: _has(c, "roe") and c["roe"] > 0,
    "ni_pos":          lambda c: _has(c, "ni_ttm") and c["ni_ttm"] > 0,
    "low_debt":        lambda c: _has(c, "de") and c["de"] < 1.0,
    "quality":         lambda c: _has(c, "roe", "de") and c["roe"] > 0 and c["de"] < 1.0,
    "pb_lt3":          lambda c: _has(c, "pb") and 0 < c["pb"] < 3,
    "pb_lt2":          lambda c: _has(c, "pb") and 0 < c["pb"] < 2,
    "value_trap_filt": lambda c: _has(c, "roe", "pb", "eq_growth") and c["roe"] > 0 and 0 < c["pb"] < 8 and c["eq_growth"],
    "rev_growth":      lambda c: _has(c, "rev_yoy") and c["rev_yoy"] > 0,
    "fcf_pos":         lambda c: _has(c, "fcf") and c["fcf"] > 0,
    "gross_margin_hi": lambda c: _has(c, "gm") and c["gm"] > 0.4,
    "curr_ratio_hi":   lambda c: _has(c, "cr") and c["cr"] > 1.5,
    "ad_up":           lambda c: c.get("ad_up") is True,
    "vol_confirm":     lambda c: c.get("vol_ok") is True,
    "higher_low":      lambda c: c.get("hl") is True,
    "dd40":            lambda c: _has(c, "dd") and c["dd"] <= -0.40,
    "sector_mom":      lambda c: c.get("sector_mom") is True,
    "quality_value":   lambda c: _has(c, "roe", "de", "pb") and c["roe"] > 0 and c["de"] < 1.0 and 0 < c["pb"] < 3,
    "quality_sector":  lambda c: _has(c, "roe", "de") and c["roe"] > 0 and c["de"] < 1.0 and c.get("sector_mom") is True,
    "quality_ad":      lambda c: _has(c, "roe", "de") and c["roe"] > 0 and c["de"] < 1.0 and c.get("ad_up") is True,
    "kitchen_sink":    lambda c: (_has(c, "roe", "de", "pb") and c["roe"] > 0 and c["de"] < 1.0
                                  and 0 < c["pb"] < 5 and c.get("sector_mom") is True and c.get("ad_up") is True),
    # --- follow-up: drawdown gradient (does deeper = monotonically better? where's the sweet spot?) ---
    "dd20":            lambda c: _has(c, "dd") and c["dd"] <= -0.20,
    "dd30":            lambda c: _has(c, "dd") and c["dd"] <= -0.30,
    "dd50":            lambda c: _has(c, "dd") and c["dd"] <= -0.50,
    "dd60":            lambda c: _has(c, "dd") and c["dd"] <= -0.60,
    # --- follow-up: orthogonal combos of the 3 winners (do they stack or over-filter?) ---
    "dd40_sector":     lambda c: _has(c, "dd") and c["dd"] <= -0.40 and c.get("sector_mom") is True,
    "dd40_cr":         lambda c: _has(c, "dd", "cr") and c["dd"] <= -0.40 and c["cr"] > 1.5,
    "sector_cr":       lambda c: c.get("sector_mom") is True and _has(c, "cr") and c["cr"] > 1.5,
    "dd30_sector":     lambda c: _has(c, "dd") and c["dd"] <= -0.30 and c.get("sector_mom") is True,
    "dd30_cr":         lambda c: _has(c, "dd", "cr") and c["dd"] <= -0.30 and c["cr"] > 1.5,
    "dd40_sector_cr":  lambda c: (_has(c, "dd", "cr") and c["dd"] <= -0.40 and c["cr"] > 1.5
                                  and c.get("sector_mom") is True),
    "dd30_sector_cr":  lambda c: (_has(c, "dd", "cr") and c["dd"] <= -0.30 and c["cr"] > 1.5
                                  and c.get("sector_mom") is True),
}


def main():
    from api.tasks import GICS2ETF
    from core.models import Fundamental

    print("loading candles from cache...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    tks = sorted(candles)
    print(f"  {len(candles)} tickers", flush=True)

    spy_close = robust(lambda: load_candles(["SPY"]))["SPY"]["Close"]
    spy_bull = (spy_close > spy_close.rolling(200).mean())
    spy_mom = spy_close.pct_change(HOLD)      # SPY trailing 6m return, for sector-momentum compare

    sec = {r["ticker"]: r["sector"] for r in
           robust(lambda: list(Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")))}
    t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
    etf_tks = sorted(set(t2etf.values()))
    etf_c = robust(lambda: load_candles(etf_tks)) or {}
    etf_mom = {e: etf_c[e]["Close"].pct_change(HOLD) for e in etf_c}
    print(f"  {len(t2etf)} tickers mapped to {len(etf_c)} sector ETFs", flush=True)

    reps = robust(lambda: load_financial_reports(tks)) or {}

    # mktcap -> large-cap top-N
    mcap = {}
    for t in tks:
        df = candles.get(t); rep = reps.get(t)
        if df is None or len(df) == 0 or rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["shares_outstanding"]).sort_values("avail_date")
        if len(r2):
            sh, px = float(r2["shares_outstanding"].iloc[-1]), float(df["Close"].iloc[-1])
            if sh > 0 and px > 0:
                mcap[t] = sh * px
    largecap = set(sorted(mcap, key=lambda t: mcap[t], reverse=True)[:TOP_N])
    print(f"  large-cap universe {len(largecap)}\n", flush=True)

    # PIT report arrays per ticker
    def prep_reports(rep):
        r = rep.dropna(subset=["avail_date"]).sort_values("avail_date").reset_index(drop=True)
        if not len(r):
            return None
        ad = pd.DatetimeIndex(pd.to_datetime(r["avail_date"]))
        col = lambda k: pd.to_numeric(r[k], errors="coerce").to_numpy(dtype=float) if k in r else np.full(len(r), np.nan)
        ni, rev, eq = col("net_income"), col("revenue"), col("total_equity")
        gp, ca, cl = col("gross_profit"), col("current_assets"), col("current_liabilities")
        debt, fcf, sh = col("total_debt"), col("free_cash_flow"), col("shares_outstanding")
        ni_ttm = pd.Series(ni).rolling(4).sum().to_numpy()
        rev_ttm = pd.Series(rev).rolling(4).sum().to_numpy()
        gp_ttm = pd.Series(gp).rolling(4).sum().to_numpy()
        return dict(ad=ad, ni_ttm=ni_ttm, rev_ttm=rev_ttm, gp_ttm=gp_ttm, eq=eq,
                    ca=ca, cl=cl, debt=debt, fcf=fcf, sh=sh)

    prepped = {t: prep_reports(reps[t]) for t in largecap if reps.get(t) is not None}

    # accumulate excess per overlay x regime
    acc = {k: {b: {"exc": [], "beat": []} for b in ("all", "bull", "bear")} for k in OVERLAYS}

    def push(key, regime, exc, beat):
        for b in ("all", regime):
            acc[key][b]["exc"].append(exc); acc[key][b]["beat"].append(beat)

    n_entries = 0
    for t in largecap:
        df = candles.get(t)
        if df is None or len(df) < 60 + 14:
            continue
        close = df["Close"].values; low = df["Low"].values; idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        n = len(close)
        # technicals
        ad = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
        ad_up = (ad > ad.rolling(20).mean()).values
        vol_ok = (df["Volume"] > df["Volume"].rolling(20).mean()).values
        roll_low = df["Low"].rolling(5).min()
        hl = (roll_low > roll_low.shift(5)).values
        ath = df["Close"].cummax().values
        dd = close / np.where(ath == 0, np.nan, ath) - 1.0
        pr = prepped.get(t)
        etf = t2etf.get(t)
        e_mom = etf_mom.get(etf)

        fires = suppressed_then_cross(df["Close"], 14)
        for i in episode_dedup([k for k in range(n) if fires[k]]):
            j = i + HOLD
            if j >= n:
                continue
            ep = close[i]
            if ep < PRICE_FLOOR or not np.isfinite(dvol[i]) or dvol[i] < DVOL_FLOOR:
                continue
            d0, d1 = idx[i], idx[j]
            s0 = spy_close.asof(d0); s1 = spy_close.asof(d1)
            if not (np.isfinite(s0) and np.isfinite(s1) and s0 > 0):
                continue
            r = (close[j] - ep) / ep * 100
            sr = (s1 / s0 - 1) * 100
            if not (np.isfinite(r) and np.isfinite(sr)):
                continue
            exc = r - sr
            beat = 1.0 if r > sr else 0.0
            regime = "bull" if bool(spy_bull.asof(d0)) else "bear"

            # build PIT context
            c = {"ad_up": bool(ad_up[i]), "vol_ok": bool(vol_ok[i]), "hl": bool(hl[i]), "dd": float(dd[i])}
            if e_mom is not None:
                sm = e_mom.asof(d0); spm = spy_mom.asof(d0)
                c["sector_mom"] = bool(np.isfinite(sm) and np.isfinite(spm) and sm > spm)
            if pr is not None:
                q = int(pr["ad"].searchsorted(d0, "right")) - 1
                if q >= 0:
                    eq = pr["eq"][q]; sh = pr["sh"][q]
                    c["ni_ttm"] = pr["ni_ttm"][q]
                    c["roe"] = pr["ni_ttm"][q] / eq if np.isfinite(eq) and eq != 0 else np.nan
                    c["de"] = pr["debt"][q] / eq if np.isfinite(eq) and eq != 0 else np.nan
                    c["pb"] = (ep * sh) / eq if np.isfinite(eq) and eq > 0 and np.isfinite(sh) else np.nan
                    c["fcf"] = pr["fcf"][q]
                    c["cr"] = pr["ca"][q] / pr["cl"][q] if np.isfinite(pr["cl"][q]) and pr["cl"][q] != 0 else np.nan
                    c["gm"] = pr["gp_ttm"][q] / pr["rev_ttm"][q] if np.isfinite(pr["rev_ttm"][q]) and pr["rev_ttm"][q] != 0 else np.nan
                    c["rev_yoy"] = (pr["rev_ttm"][q] - pr["rev_ttm"][q - 4]) if q >= 4 else np.nan
                    c["eq_growth"] = bool(q >= 4 and np.isfinite(eq) and np.isfinite(pr["eq"][q - 4]) and eq >= pr["eq"][q - 4])
            n_entries += 1
            for key, fn in OVERLAYS.items():
                try:
                    if fn(c):
                        push(key, regime, exc, beat)
                except Exception:
                    pass

    def summ(d):
        e = np.array(d["exc"]); b = np.array(d["beat"])
        if len(e) == 0:
            return {"n": 0}
        t_stat = float(e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))) if len(e) > 1 and e.std() > 0 else None
        return {"n": int(len(e)), "excess_pp": round(float(e.mean()), 2),
                "beat_pct": round(float(b.mean() * 100), 1),
                "excess_t": round(t_stat, 2) if t_stat is not None else None}

    results = {k: {b: summ(acc[k][b]) for b in ("all", "bull", "bear")} for k in OVERLAYS}

    base = results["base"]["all"]
    print(f"total base entries: {n_entries}\n", flush=True)
    print(f"{'overlay':16} | {'ALL n':>6} {'exc':>6} {'beat%':>6} {'t':>5} | "
          f"{'BULL n':>6} {'exc':>6} {'beat%':>6} {'t':>5} | {'BEAR n':>6} {'exc':>6} {'beat%':>6}", flush=True)
    print("-" * 108, flush=True)
    # sort by BULL excess desc (bull is the weak spot we want to fix), base first
    order = ["base"] + sorted([k for k in OVERLAYS if k != "base"],
                              key=lambda k: results[k]["bull"].get("excess_pp", -99) or -99, reverse=True)
    for k in order:
        a, bu, be = results[k]["all"], results[k]["bull"], results[k]["bear"]
        if a.get("n", 0) == 0:
            continue
        f = lambda s, key, w=6, p="+.2f": (format(s[key], p) if s.get(key) is not None else "-").rjust(w)
        print(f"{k:16} | {a['n']:>6} {f(a,'excess_pp')} {a['beat_pct']:>6.1f} {str(a['excess_t']):>5} | "
              f"{bu['n']:>6} {f(bu,'excess_pp')} {bu['beat_pct']:>6.1f} {str(bu['excess_t']):>5} | "
              f"{be['n']:>6} {f(be,'excess_pp')} {be['beat_pct']:>6.1f}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "params": {"win": 14, "hold_days": HOLD, "top_n": TOP_N, "base_entries": n_entries},
               "results": results,
               "caveat": "PIT overlays on rsi14 suppressed-cross, large-cap, 6m fwd, excess over SPY, "
                         "in-sample, no fees. Overlaps cluster in selloffs (bear t overstated). Per-trade "
                         "excess, NOT a portfolio book. Overlays with small n (<80) are unreliable."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rsi14_overlay_sweep",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rsi14_overlay_sweep]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
