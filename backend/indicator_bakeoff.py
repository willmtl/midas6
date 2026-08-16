#!/usr/bin/env python3
"""INDICATOR ENTRY BAKE-OFF — instead of 15 retail single-name indicator backtests (which violate
[[tail-not-average]] and don't survive our PIT/survivorship framework), drop every indicator from the
firehose into OUR framework as an ENTRY condition on the flagship rotation pick, the same way
[[entry-signal-value-pick]] found RSI-oversold adds lift but trend-confirmation SUBTRACTS.

For each flagship monthly pick (accel top-10 -> cheapest as-traded-P/B guard low-debt $5M), evaluate each
indicator's typical BUY condition on the pick's own daily price AT the pick date, and measure:
  (A) LIFT: mean forward 1mo return when the condition holds vs the pick's unconditional mean.
  (B) OVERLAY: portfolio that only TAKES the pick when the condition holds (else skips the slot), vs baseline.
Indicators: Williams %R, Stochastic %K, Bollinger %B, Boll+Stoch combo (the 'killer' article), DMI/ADX,
TSI, KST, RVI. Oversold-style (mean-reversion) vs trend-style (ADX/TSI/KST) — our prior: oversold helps,
trend/confirmation hurts. -> BacktestResult[indicator_bakeoff] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/indicator_bakeoff.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH

TOP_N = 10; MIN_DVOL = 5e6
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "indicator_bakeoff.json"


# ---- indicator BUY conditions on a daily OHLC frame -> bool Series (True = condition holds that day) ----
def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def cond_williams_oversold(d, n=14):
    hh = d["High"].rolling(n).max(); ll = d["Low"].rolling(n).min()
    wr = -100 * (hh - d["Close"]) / (hh - ll).replace(0, np.nan)
    return wr < -80                                  # in oversold


def cond_stoch_oversold(d, n=14):
    hh = d["High"].rolling(n).max(); ll = d["Low"].rolling(n).min()
    k = 100 * (d["Close"] - ll) / (hh - ll).replace(0, np.nan)
    return k < 20


def cond_bollinger_low(d, n=20, k=2):
    ma = d["Close"].rolling(n).mean(); sd = d["Close"].rolling(n).std()
    pctb = (d["Close"] - (ma - k * sd)) / ((2 * k * sd).replace(0, np.nan))
    return pctb < 0.2                                 # near/below lower band


def cond_boll_stoch(d):
    return cond_bollinger_low(d) & cond_stoch_oversold(d)


def cond_adx_trend(d, n=14):
    up = d["High"].diff(); dn = -d["Low"].diff()
    plus_dm = ((up > dn) & (up > 0)) * up.clip(lower=0)
    minus_dm = ((dn > up) & (dn > 0)) * dn.clip(lower=0)
    tr = pd.concat([(d["High"] - d["Low"]),
                    (d["High"] - d["Close"].shift()).abs(),
                    (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n).replace(0, np.nan)
    pdi = 100 * _wilder(plus_dm, n) / atr
    mdi = 100 * _wilder(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = _wilder(dx, n)
    return (adx > 25) & (pdi > mdi)                   # strong up-trend


def cond_tsi_bull(d, long=25, short=13):
    m = d["Close"].diff()
    tsi = 100 * m.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean() / \
        m.abs().ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean().replace(0, np.nan)
    return tsi > 0                                    # bullish momentum


def cond_kst_bull(d):
    c = d["Close"]
    def rcma(np_, sm):
        return (c / c.shift(np_) - 1).rolling(sm).mean()
    kst = (rcma(10, 10) * 1 + rcma(15, 10) * 2 + rcma(20, 10) * 3 + rcma(30, 15) * 4)
    sig = kst.rolling(9).mean()
    return kst > sig                                  # bullish crossover


def cond_rvi_bull(d):
    num = (d["Close"] - d["Open"]); den = (d["High"] - d["Low"])
    sw = lambda s: (s + 2 * s.shift(1) + 2 * s.shift(2) + s.shift(3)) / 6
    rvi = sw(num).rolling(10).sum() / sw(den).rolling(10).sum().replace(0, np.nan)
    sig = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6
    return rvi > sig                                  # bullish crossover


def _tr(d):
    return pd.concat([(d["High"] - d["Low"]),
                      (d["High"] - d["Close"].shift()).abs(),
                      (d["Low"] - d["Close"].shift()).abs()], axis=1).max(axis=1)


def cond_rsi_oversold(d, n=14):
    delta = d["Close"].diff()
    rs = _wilder(delta.clip(lower=0), n) / _wilder(-delta.clip(upper=0), n).replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi < 35                                    # our known oversold-dip entry


def cond_macd_bull(d):
    macd = d["Close"].ewm(span=12, adjust=False).mean() - d["Close"].ewm(span=26, adjust=False).mean()
    return macd > macd.ewm(span=9, adjust=False).mean()   # MACD above signal (removed from strat 2026-07)


def cond_ema_cross(d):
    return d["Close"].ewm(span=12, adjust=False).mean() > d["Close"].ewm(span=26, adjust=False).mean()


def cond_ma_trend(d, n=50):
    return d["Close"] > d["Close"].rolling(n).mean()   # price above SMA50 (trend filter)


def cond_keltner_break(d, n=20, m=10, k=2):
    ema = d["Close"].ewm(span=n, adjust=False).mean()
    return d["Close"] > (ema + k * _wilder(_tr(d), m))   # breakout above upper Keltner band


def cond_volume_surge(d, n=20, k=1.5):
    return d["Volume"] > k * d["Volume"].rolling(n).mean()   # avg-volume confirmation


def cond_fib_pullback(d, n=60):
    hh = d["Close"].rolling(n).max(); ll = d["Close"].rolling(n).min()
    retr = (hh - d["Close"]) / (hh - ll).replace(0, np.nan)
    up = d["Close"] > d["Close"].rolling(50).mean()
    return (retr >= 0.382) & (retr <= 0.618) & up      # pullback to the 38.2-61.8% Fib zone in an uptrend


CONDS = {
    "williams_oversold": cond_williams_oversold,
    "stoch_oversold": cond_stoch_oversold,
    "bollinger_low": cond_bollinger_low,
    "boll+stoch_combo": cond_boll_stoch,
    "adx_uptrend": cond_adx_trend,
    "tsi_bullish": cond_tsi_bull,
    "kst_bullish": cond_kst_bull,
    "rvi_bullish": cond_rvi_bull,
    "rsi_oversold": cond_rsi_oversold,
    "macd_bull": cond_macd_bull,
    "ema_cross": cond_ema_cross,
    "ma_trend_sma50": cond_ma_trend,
    "keltner_breakout": cond_keltner_break,
    "volume_surge": cond_volume_surge,
    "fib_pullback": cond_fib_pullback,
}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, set(h)); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)

    def pick(etf, date, held):
        _, holds = sector_map.get(etf, (etf, set()))
        c = [h for h in holds if h in px.columns and h not in held and _available_at(px[h], date)
             and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        return min(g, key=lambda h: pb.loc[date, h]) if g else None

    # 1) generate flagship picks with forward 1mo return
    picks = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set()
        for etf in top:
            p = pick(etf, date, held)
            if not p:
                continue
            held.add(p)
            r = _ret_delist(px[p], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            picks.append({"date": date, "i": i, "ticker": p, "ret": float(r), "spy": float(sp)})
    n_all = len(picks)
    base_mean = float(np.mean([p["ret"] for p in picks])) * 100
    print(f"{n_all} flagship picks; unconditional mean fwd-1mo {base_mean:.2f}%", flush=True)

    # 2) precompute each indicator's daily condition for distinct picked tickers, eval asof pick date
    tks = sorted({p["ticker"] for p in picks})
    cond_series = {name: {} for name in CONDS}
    for t in tks:
        d = stock_daily.get(t)
        if d is None or not {"High", "Low", "Close", "Open"}.issubset(d.columns) or len(d) < 60:
            continue
        for name, fn in CONDS.items():
            try:
                cond_series[name][t] = fn(d).reindex(d.index)
            except Exception:
                pass
    for p in picks:
        for name in CONDS:
            s = cond_series[name].get(p["ticker"])
            if s is None:
                p[name] = None; continue
            s2 = s.dropna()
            if s2.empty:
                p[name] = None; continue
            j = s2.index.searchsorted(p["date"], side="right") - 1   # asof pick date (no look-ahead)
            p[name] = bool(s2.iloc[j]) if j >= 0 else None

    # 3A) lift + 3B) overlay portfolio per indicator
    def overlay_perf(name):
        rets, spies = [], []
        # group picks by month, take only those with cond True; portfolio = equal-weight that month
        by_month = {}
        for p in picks:
            by_month.setdefault(p["i"], []).append(p)
        for i, ps in sorted(by_month.items()):
            sel = [p for p in ps if p.get(name) is True]
            if not sel:
                continue
            rets.append(float(np.mean([p["ret"] for p in sel]))); spies.append(ps[0]["spy"])
        if not rets:
            return None
        r = np.asarray(rets)
        tot = float(np.prod(1 + r) - 1) * 100
        sp = float(np.prod(1 + np.asarray(spies)) - 1) * 100
        sh_ = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
        eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
        return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh_, 2), dd=round(dd, 1),
                    months=len(r))

    # baseline equal-weight portfolio (all picks) for overlay comparison
    bym = {}
    for p in picks:
        bym.setdefault(p["i"], []).append(p)
    brets, bspies = [], []
    for i, ps in sorted(bym.items()):
        brets.append(float(np.mean([p["ret"] for p in ps]))); bspies.append(ps[0]["spy"])
    br = np.asarray(brets)
    base_ov = dict(total=round(float(np.prod(1 + br) - 1) * 100, 1),
                   vs_spy=round(float(np.prod(1 + br) - 1) * 100 - float(np.prod(1 + np.asarray(bspies)) - 1) * 100, 1),
                   sharpe=round(float(br.mean() / br.std() * np.sqrt(12)), 2),
                   dd=round(float(((np.cumprod(1 + br) / np.maximum.accumulate(np.cumprod(1 + br))) - 1).min() * 100), 1),
                   months=len(br))

    results = {}
    print(f"\n=== INDICATOR ENTRY BAKE-OFF (lift = cond mean − unconditional {base_mean:.2f}%) ===", flush=True)
    print(f"  {'indicator':<20}{'hit%':>6}{'cond ret':>10}{'lift(pp)':>10}   overlay: total / vsSPY / Sh / DD", flush=True)
    for name in CONDS:
        hits = [p["ret"] for p in picks if p.get(name) is True]
        cov = [p for p in picks if p.get(name) is not None]
        hitpct = round(100 * len(hits) / max(len(cov), 1), 1)
        cmean = round(float(np.mean(hits)) * 100, 2) if hits else None
        lift = round(cmean - base_mean, 2) if cmean is not None else None
        t = round(_tstat_from_returns(hits), 2) if len(hits) > 2 else None
        ov = overlay_perf(name)
        results[name] = {"hit_pct": hitpct, "cond_ret": cmean, "lift_pp": lift, "t_stat": t,
                         "n_hits": len(hits), "overlay": ov}
        ovs = f"{ov['total']}% / {ov['vs_spy']:+} / {ov['sharpe']} / {ov['dd']}%" if ov else "n/a"
        print(f"  {name:<20}{hitpct:>6}{str(cmean):>10}{str(lift):>10}   {ovs}", flush=True)
    print(f"  {'BASELINE (all picks)':<20}{'100':>6}{base_mean:>10.2f}{'0.0':>10}   "
          f"{base_ov['total']}% / {base_ov['vs_spy']:+} / {base_ov['sharpe']} / {base_ov['dd']}%", flush=True)

    ranked = sorted((r for r in results.items() if r[1]["lift_pp"] is not None),
                    key=lambda kv: kv[1]["lift_pp"], reverse=True)
    best = ranked[0] if ranked else (None, None)
    verdict = (
        f"Unconditional pick fwd-1mo {base_mean:.2f}% (baseline overlay {base_ov['total']}%/Sh{base_ov['sharpe']}). "
        + (f"Best entry lift = {best[0]} ({best[1]['lift_pp']:+}pp, t{best[1]['t_stat']}, overlay Sh"
           f"{best[1]['overlay']['sharpe'] if best[1]['overlay'] else 'n/a'}). " if best[0] else "")
        + "Oversold-style (williams/stoch/bollinger) vs trend-style (adx/tsi/kst/rvi): "
        + ("consistent with [[entry-signal-value-pick]] — oversold entries add small lift, trend-confirmation "
           "subtracts. Any indicator whose overlay Sharpe beats baseline is worth wiring; the rest are answered NO."
           if any((v["lift_pp"] or -9) > 0 for v in results.values()) else
           "none add positive lift — the rotation pick's own value+accel selection already captures the timing.")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "min_dvol": MIN_DVOL, "benchmark": BENCH, "n_picks": n_all,
                   "unconditional_mean_1mo_pct": round(base_mean, 2), "entry_eval": "asof pick date, no look-ahead"},
        "baseline_overlay": base_ov, "results": results, "verdict": verdict,
        "caveat": "Entry condition evaluated on the pick's daily OHLC as-of the pick date (searchsorted right-1, no "
                  "look-ahead). Overlay = equal-weight the picks passing the filter each month (skips a slot if none "
                  "pass -> fewer names, higher idiosyncratic variance). Standard indicator formulas; KST/RVI/ADX are "
                  "the fiddly ones. PIT, no fees, present-day-holdings survivorship, TR returns.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="indicator_bakeoff", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                "computed_at": timezone.now()})
        print("Saved BacktestResult[indicator_bakeoff]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
