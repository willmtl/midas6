#!/usr/bin/env python3
"""Round 2 of overlays on RSI(14) suppressed-cross (large-cap): ALT-DATA + MACRO regime gates
we hadn't pulled yet. Each tested standalone AND stacked on the current best (dd30+sector), to
see if it adds beyond the known winners (drawdown / sector-momentum / liquidity).

New criteria:
  MACRO (FRED MacroSeries, market-wide, PIT via date asof):
    net_liq_rising  net liquidity WALCL-1000*(RRPONTSYD+WTREGEN) higher than ~63 trading days ago
    hy_falling      HY OAS spread (BAMLH0A0HYM2) falling over ~21d (credit improving = risk-on)
    hy_elevated     HY spread above its trailing 1y median (stress regime)
    m2_rising       M2SL higher than ~126 trading days ago
  ALT-DATA (per-ticker, PIT):
    insider_buy     net insider buying (buy_value>sell_value) in the 90d BEFORE entry (InsiderBuy, filed<=entry)
    iv_skew_hi/lo   OptionSnapshot iv_skew above/below its trailing median at entry (COVERAGE 2022-09+ only)
    low_si/high_si  Fundamental.short_pct_float </> threshold  (⚠ CURRENT snapshot = LOOK-AHEAD, flagged)

-> BacktestResult[rsi14_overlay_altdata] + JSON. Candles from /tmp/beat_candles.pkl cache.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/rsi14_overlay_altdata.py
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "rsi14_overlay_altdata.json"
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

# base best-so-far to stack on
_BEST = lambda c: (_f(c, "dd") and c["dd"] <= -0.30 and c.get("sector_mom") is True)

OVERLAYS = {
    "base":                 lambda c: True,
    "dd30_sector(best)":    _BEST,
    # --- macro regime, standalone ---
    "net_liq_rising":       lambda c: c.get("net_liq_rising") is True,
    "hy_falling":           lambda c: c.get("hy_falling") is True,
    "hy_elevated":          lambda c: c.get("hy_elevated") is True,
    "m2_rising":            lambda c: c.get("m2_rising") is True,
    # --- alt-data, standalone ---
    "insider_buy":          lambda c: c.get("insider_buy") is True,
    "iv_skew_hi":           lambda c: c.get("iv_skew_hi") is True,
    "iv_skew_lo":           lambda c: c.get("iv_skew_lo") is True,
    "low_si":               lambda c: _f(c, "si_pct") and c["si_pct"] < 5.0,
    "high_si":              lambda c: _f(c, "si_pct") and c["si_pct"] > 10.0,
    # --- stacked on the best combo: does the new criterion ADD? ---
    "best+net_liq_rising":  lambda c: _BEST(c) and c.get("net_liq_rising") is True,
    "best+hy_falling":      lambda c: _BEST(c) and c.get("hy_falling") is True,
    "best+hy_elevated":     lambda c: _BEST(c) and c.get("hy_elevated") is True,
    "best+insider_buy":     lambda c: _BEST(c) and c.get("insider_buy") is True,
    "best+low_si":          lambda c: _BEST(c) and _f(c, "si_pct") and c["si_pct"] < 5.0,
}


def main():
    from api.tasks import GICS2ETF
    from core.models import Fundamental, MacroSeries, InsiderBuy, OptionSnapshot

    print("loading candles from cache...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    tks = sorted(candles)
    print(f"  {len(candles)} tickers", flush=True)

    spy_close = robust(lambda: load_candles(["SPY"]))["SPY"]["Close"]
    tidx = spy_close.index                       # trading-day calendar
    spy_bull = (spy_close > spy_close.rolling(200).mean())
    spy_mom = spy_close.pct_change(HOLD)

    # sector momentum
    sec = {r["ticker"]: r["sector"] for r in
           robust(lambda: list(Fundamental.objects.filter(ticker__in=tks).values("ticker", "sector")))}
    t2etf = {t: GICS2ETF[s] for t, s in sec.items() if s in GICS2ETF}
    etf_c = robust(lambda: load_candles(sorted(set(t2etf.values())))) or {}
    etf_mom = {e: etf_c[e]["Close"].pct_change(HOLD) for e in etf_c}

    # current short interest (LOOK-AHEAD snapshot, flagged)
    si_pct = {r["ticker"]: r["short_pct_float"] for r in
              robust(lambda: list(Fundamental.objects.filter(ticker__in=tks)
                                  .values("ticker", "short_pct_float"))) if r["short_pct_float"] is not None}

    # macro series -> daily-aligned regime booleans on the trading calendar
    mrows = robust(lambda: list(MacroSeries.objects.values("series", "date", "value"))) or []
    mdf = pd.DataFrame(mrows)
    def series(name):
        s = mdf[mdf["series"] == name].copy()
        s["date"] = pd.to_datetime(s["date"])
        return (s.set_index("date")["value"].sort_index()
                .reindex(tidx, method="ffill").astype(float))
    walcl, rrp, tga = series("WALCL"), series("RRPONTSYD"), series("WTREGEN")
    net_liq = walcl - 1000.0 * (rrp + tga)        # WALCL millions; RRP/TGA billions -> millions
    hy = series("BAMLH0A0HYM2"); m2 = series("M2SL")
    reg_net_liq_rising = (net_liq > net_liq.shift(63))
    reg_hy_falling = (hy < hy.shift(21))
    reg_hy_elevated = (hy > hy.rolling(252, min_periods=60).median())
    reg_m2_rising = (m2 > m2.shift(126))
    print(f"  macro aligned; net_liq valid {int(net_liq.notna().sum())}/{len(tidx)}", flush=True)

    # insider net-buy panel: per ticker sorted filed_date + cumulative net
    ins = {}
    for r in (robust(lambda: list(InsiderBuy.objects.values("ticker", "filed_date", "buy_value", "sell_value"))) or []):
        ins.setdefault(r["ticker"], []).append((pd.Timestamp(r["filed_date"]),
                                                 float(r["buy_value"] or 0), float(r["sell_value"] or 0)))
    for t in ins:
        ins[t].sort()

    # iv_skew per-ticker date->value
    skew = {}
    for r in (robust(lambda: list(OptionSnapshot.objects.values("ticker", "date", "iv_skew"))) or []):
        if r["iv_skew"] is not None:
            skew.setdefault(r["ticker"], {})[pd.Timestamp(r["date"])] = float(r["iv_skew"])
    skew_s = {t: pd.Series(d).sort_index() for t, d in skew.items()}
    print(f"  insider tickers {len(ins)}; iv_skew tickers {len(skew_s)}", flush=True)

    reps = robust(lambda: load_financial_reports(tks)) or {}
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
    print(f"  large-cap {len(largecap)}\n", flush=True)

    acc = {k: {b: {"exc": [], "beat": []} for b in ("all", "bull", "bear")} for k in OVERLAYS}
    def push(key, regime, exc, beat):
        for b in ("all", regime):
            acc[key][b]["exc"].append(exc); acc[key][b]["beat"].append(beat)

    def insider_net_buy(t, d):
        rows = ins.get(t)
        if not rows:
            return False
        lo = d - pd.Timedelta(days=90)
        b = s = 0.0
        for fd, bv, sv in rows:
            if fd > d:
                break
            if fd >= lo:
                b += bv; s += sv
        return b > s and b > 0

    n_entries = 0
    for t in largecap:
        df = candles.get(t)
        if df is None or len(df) < 74:
            continue
        close = df["Close"].values; idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        n = len(close)
        ath = df["Close"].cummax().values
        dd = close / np.where(ath == 0, np.nan, ath) - 1.0
        e_mom = etf_mom.get(t2etf.get(t))
        sk = skew_s.get(t)
        sk_med = sk.expanding(min_periods=20).median() if sk is not None else None
        fires = suppressed_then_cross(df["Close"], 14)     # compute ONCE per ticker (not per bar!)

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
            r = (close[j] - ep) / ep * 100; sr = (s1 / s0 - 1) * 100
            if not (np.isfinite(r) and np.isfinite(sr)):
                continue
            exc = r - sr; beat = 1.0 if r > sr else 0.0
            regime = "bull" if bool(spy_bull.asof(d0)) else "bear"

            c = {"dd": float(dd[i]), "si_pct": si_pct.get(t)}
            if e_mom is not None:
                sm = e_mom.asof(d0); spm = spy_mom.asof(d0)
                c["sector_mom"] = bool(np.isfinite(sm) and np.isfinite(spm) and sm > spm)
            c["net_liq_rising"] = bool(reg_net_liq_rising.asof(d0)) if pd.notna(reg_net_liq_rising.asof(d0)) else None
            c["hy_falling"] = bool(reg_hy_falling.asof(d0)) if pd.notna(reg_hy_falling.asof(d0)) else None
            c["hy_elevated"] = bool(reg_hy_elevated.asof(d0)) if pd.notna(reg_hy_elevated.asof(d0)) else None
            c["m2_rising"] = bool(reg_m2_rising.asof(d0)) if pd.notna(reg_m2_rising.asof(d0)) else None
            c["insider_buy"] = insider_net_buy(t, d0)
            if sk is not None and sk_med is not None:
                v = sk.asof(d0); md = sk_med.asof(d0)
                if pd.notna(v) and pd.notna(md):
                    c["iv_skew_hi"] = bool(v > md); c["iv_skew_lo"] = bool(v < md)
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

    print(f"total base entries: {n_entries}\n", flush=True)
    print(f"{'overlay':20} | {'ALL n':>6} {'exc':>6} {'beat%':>6} {'t':>5} | "
          f"{'BULL n':>6} {'exc':>6} {'beat%':>6} {'t':>5} | {'BEAR n':>6} {'exc':>6} {'beat%':>6}", flush=True)
    print("-" * 112, flush=True)
    def cell(s, wide=True):
        if s.get("n", 0) == 0:
            return f"{0:>6} {'-':>6} {'-':>6}" + (f" {'-':>5}" if wide else "")
        exc = format(s["excess_pp"], "+.2f") if s.get("excess_pp") is not None else "-"
        base = f"{s['n']:>6} {exc:>6} {s['beat_pct']:>6.1f}"
        return base + (f" {str(s.get('excess_t')):>5}" if wide else "")
    for k in OVERLAYS:
        a, bu, be = results[k]["all"], results[k]["bull"], results[k]["bear"]
        print(f"{k:20} | {cell(a)} | {cell(bu)} | {cell(be, wide=False)}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "params": {"win": 14, "hold_days": HOLD, "top_n": TOP_N, "base_entries": n_entries},
               "results": results,
               "caveat": "Alt-data/macro overlays on rsi14 suppressed-cross, large-cap, 6m, excess over SPY, "
                         "in-sample, no fees. iv_skew coverage 2022-09+ only (thin). low_si/high_si use a "
                         "CURRENT short_pct_float snapshot = LOOK-AHEAD (directional only). Still per-trade "
                         "excess on a survivorship-biased (current-mktcap) universe, NOT a portfolio book."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rsi14_overlay_altdata",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[rsi14_overlay_altdata]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
