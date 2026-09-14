#!/usr/bin/env python3
"""Re-validate the drawdown story with POINT-IN-TIME large-cap membership (user directive:
"use past market cap not current"). Fixes the recovered-survivor bias: a name counts as
large-cap only if it was top-500 by mktcap ON THE ENTRY DATE (PIT shares x close), not today.

mktcap_t(d) = shares_outstanding (latest report avail_date<=d) * close[d]; ranked cross-
sectionally each day; membership = daily rank <= TOP_N. Re-runs the drawdown gradient +
winner combos + the credit-stress lever under this PIT gate, vs SPY, split bull/bear.

NOTE: still can't cure DELISTED survivorship (cache universe = currently-listed tickers only);
this fixes the MEMBERSHIP look-ahead, which is what drove the deep-drawdown buckets.
-> BacktestResult[rsi14_pit_membership] + JSON. Candles from /tmp/beat_candles.pkl.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/rsi14_pit_membership.py
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi14_pit_membership.json"
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


def _f(c, k):
    return c.get(k) is not None and (not isinstance(c[k], float) or np.isfinite(c[k]))

OVERLAYS = {
    "base":            lambda c: True,
    "dd20":            lambda c: _f(c, "dd") and c["dd"] <= -0.20,
    "dd30":            lambda c: _f(c, "dd") and c["dd"] <= -0.30,
    "dd40":            lambda c: _f(c, "dd") and c["dd"] <= -0.40,
    "dd50":            lambda c: _f(c, "dd") and c["dd"] <= -0.50,
    "dd60":            lambda c: _f(c, "dd") and c["dd"] <= -0.60,
    "sector_mom":      lambda c: c.get("sector_mom") is True,
    "curr_ratio_hi":   lambda c: _f(c, "cr") and c["cr"] > 1.5,
    "hy_elevated":     lambda c: c.get("hy_elevated") is True,
    "dd30_sector":     lambda c: _f(c, "dd") and c["dd"] <= -0.30 and c.get("sector_mom") is True,
    "dd40_sector":     lambda c: _f(c, "dd") and c["dd"] <= -0.40 and c.get("sector_mom") is True,
    "sector_cr":       lambda c: c.get("sector_mom") is True and _f(c, "cr") and c["cr"] > 1.5,
    "dd30_cr":         lambda c: _f(c, "dd", ) and c["dd"] <= -0.30 and _f(c, "cr") and c["cr"] > 1.5,
    "dd30_sector_cr":  lambda c: (_f(c, "dd") and c["dd"] <= -0.30 and c.get("sector_mom") is True
                                  and _f(c, "cr") and c["cr"] > 1.5),
    "dd30_sector_hy":  lambda c: (_f(c, "dd") and c["dd"] <= -0.30 and c.get("sector_mom") is True
                                  and c.get("hy_elevated") is True),
}


def main():
    from api.tasks import GICS2ETF
    from core.models import Fundamental, MacroSeries

    print("loading candles from cache...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    tks = sorted(candles)
    print(f"  {len(candles)} tickers", flush=True)

    spy_close = robust(lambda: load_candles(["SPY"]))["SPY"]["Close"]
    master = spy_close.index
    spy_bull = (spy_close > spy_close.rolling(200).mean())
    spy_mom = spy_close.pct_change(HOLD)

    sec = {r["ticker"]: r["sector"] for r in
           robust(lambda: list(Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")))}
    t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
    etf_c = robust(lambda: load_candles(sorted(set(t2etf.values())))) or {}
    etf_mom = {e: etf_c[e]["Close"].pct_change(HOLD) for e in etf_c}

    # credit-stress regime
    mrows = robust(lambda: list(MacroSeries.objects.filter(series="BAMLH0A0HYM2").values("date", "value"))) or []
    hy = pd.DataFrame(mrows)
    hy["date"] = pd.to_datetime(hy["date"])
    hy = hy.set_index("date")["value"].sort_index().reindex(master, method="ffill").astype(float)
    reg_hy_elevated = (hy > hy.rolling(252, min_periods=60).median())

    reps = robust(lambda: load_financial_reports(tks)) or {}

    # ---- POINT-IN-TIME market-cap panel: shares(avail_date)<=d ffilled * close[d] ----
    print("building PIT mktcap panel...", flush=True)
    mcap_cols = {}
    shares_daily = {}
    for t in tks:
        df = candles.get(t); rep = reps.get(t)
        if df is None or len(df) == 0 or rep is None or not len(rep):
            continue
        r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
        if not len(r2):
            continue
        sh = pd.Series(pd.to_numeric(r2["shares_outstanding"], errors="coerce").to_numpy(),
                       index=pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]))).sort_index()
        sh = sh[~sh.index.duplicated(keep="last")].reindex(master, method="ffill")
        px = df["Close"].reindex(master, method="ffill")
        mcap_cols[t] = sh * px
        shares_daily[t] = sh
    mcap_panel = pd.DataFrame(mcap_cols)
    rank = mcap_panel.rank(axis=1, ascending=False)
    member = (rank <= TOP_N)                      # master x ticker booleans
    avg_members = float(member.sum(axis=1).replace(0, np.nan).dropna().mean())
    print(f"  panel {mcap_panel.shape}; avg PIT members/day ~{avg_members:.0f}\n", flush=True)

    acc = {k: {b: {"exc": [], "beat": []} for b in ("all", "bull", "bear")} for k in OVERLAYS}
    def push(key, regime, exc, beat):
        for b in ("all", regime):
            acc[key][b]["exc"].append(exc); acc[key][b]["beat"].append(beat)

    n_entries = 0
    for t in tks:
        df = candles.get(t)
        if df is None or len(df) < 74 or t not in mcap_cols:
            continue
        close = df["Close"].values; idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        n = len(close)
        ath = df["Close"].cummax().values
        dd = close / np.where(ath == 0, np.nan, ath) - 1.0
        e_mom = etf_mom.get(t2etf.get(t))
        mem_t = member[t] if t in member.columns else None
        # PIT current ratio from reports
        r2 = reps[t].dropna(subset=["avail_date"]).sort_values("avail_date").reset_index(drop=True)
        rad = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]))
        ca = pd.to_numeric(r2.get("current_assets"), errors="coerce").to_numpy(dtype=float) if "current_assets" in r2 else None
        cl = pd.to_numeric(r2.get("current_liabilities"), errors="coerce").to_numpy(dtype=float) if "current_liabilities" in r2 else None
        fires = suppressed_then_cross(df["Close"], 14)

        for i in episode_dedup([k for k in range(n) if fires[k]]):
            j = i + HOLD
            if j >= n:
                continue
            ep = close[i]
            if ep < PRICE_FLOOR or not np.isfinite(dvol[i]) or dvol[i] < DVOL_FLOOR:
                continue
            d0, d1 = idx[i], idx[j]
            # PIT membership gate
            if mem_t is None or not bool(mem_t.asof(d0)):
                continue
            s0 = spy_close.asof(d0); s1 = spy_close.asof(d1)
            if not (np.isfinite(s0) and np.isfinite(s1) and s0 > 0):
                continue
            r = (close[j] - ep) / ep * 100; sr = (s1 / s0 - 1) * 100
            if not (np.isfinite(r) and np.isfinite(sr)):
                continue
            exc = r - sr; beat = 1.0 if r > sr else 0.0
            regime = "bull" if bool(spy_bull.asof(d0)) else "bear"

            c = {"dd": float(dd[i])}
            if e_mom is not None:
                sm = e_mom.asof(d0); spm = spy_mom.asof(d0)
                c["sector_mom"] = bool(np.isfinite(sm) and np.isfinite(spm) and sm > spm)
            c["hy_elevated"] = bool(reg_hy_elevated.asof(d0)) if pd.notna(reg_hy_elevated.asof(d0)) else None
            if ca is not None and cl is not None and len(rad):
                q = int(rad.searchsorted(d0, "right")) - 1
                if q >= 0 and np.isfinite(cl[q]) and cl[q] != 0:
                    c["cr"] = ca[q] / cl[q]
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

    print(f"total PIT-member base entries: {n_entries}\n", flush=True)
    print(f"{'overlay':18} | {'ALL n':>6} {'exc':>7} {'beat%':>6} {'t':>5} | "
          f"{'BULL n':>6} {'exc':>7} {'beat%':>6} {'t':>5} | {'BEAR n':>6} {'exc':>7} {'beat%':>6}", flush=True)
    print("-" * 112, flush=True)
    def cell(s, wide=True):
        if s.get("n", 0) == 0:
            return f"{0:>6} {'-':>7} {'-':>6}" + (f" {'-':>5}" if wide else "")
        exc = format(s["excess_pp"], "+.2f")
        return f"{s['n']:>6} {exc:>7} {s['beat_pct']:>6.1f}" + (f" {str(s.get('excess_t')):>5}" if wide else "")
    for k in OVERLAYS:
        a, bu, be = results[k]["all"], results[k]["bull"], results[k]["bear"]
        print(f"{k:18} | {cell(a)} | {cell(bu)} | {cell(be, wide=False)}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "params": {"win": 14, "hold_days": HOLD, "top_n": TOP_N, "base_entries": n_entries,
                          "membership": "point_in_time_daily_mktcap_rank"},
               "results": results,
               "caveat": "PIT large-cap membership (top-500 by mktcap ON entry date) fixes the recovered-"
                         "survivor MEMBERSHIP bias. Does NOT fix delisted survivorship (cache=currently-listed "
                         "tickers only). In-sample, no fees, per-trade excess vs SPY, not a book."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rsi14_pit_membership",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rsi14_pit_membership]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
