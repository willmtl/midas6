#!/usr/bin/env python3
"""A vs B vs VALUE, as the PER-SECTOR pick inside the acceleration sectors. Rotation (accel) is fixed;
only the stock-selection rule changes. Is there a case to sometimes pick a capitulation (A) or dip-in-
uptrend (B) stock instead of always the cheapest-P/B value stock (C)?

  C_value      cheapest positive-P/B, guard + low-debt  (the validated pick — baseline)
  A_capitul    most OVERSOLD guarded stock (min RSI10) — capitulation, ignore value
  A_value      cheapest-P/B among DEEP-oversold (RSI10<30) guarded — value INTERSECT capitulation
  B_dip        cheapest-P/B among ABOVE-200d-MA + RSI10<45 guarded — value INTERSECT dip-in-uptrend
  best_regime  risk-off (SPY<200dMA) -> A_value ; risk-on -> B_dip ; (adaptive by market regime)
All fall back to C_value when their filter is empty (never lose a sector). vs SPY / Sharpe / DD / win.
-> BacktestResult[ab_vs_value] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/ab_vs_value_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "ab_vs_value.json"
LOOKBACK, TOP_N = 6, 10


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, win_pct=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_close = etf_daily[BENCH]["Close"]
    spy_m = spy_close.resample("ME").last().reindex(midx)
    etf_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_riskoff = (spy_close < spy_close.rolling(200).mean()).resample("ME").last().reindex(midx).fillna(False)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    rsi = pd.DataFrame({t: _rsi(stock_daily[t]["Close"]).resample("ME").last().reindex(midx)
                        for t in common if t in stock_daily and len(stock_daily[t]) > 20}).reindex(index=midx, columns=common)
    above200 = pd.DataFrame({t: (stock_daily[t]["Close"] > stock_daily[t]["Close"].rolling(200).mean())
                             .resample("ME").last().reindex(midx) for t in common
                             if t in stock_daily and len(stock_daily[t]) >= 200}).reindex(index=midx, columns=common).fillna(False)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 12

    def pick(mode, date, guarded):
        """guarded = list of (ticker) passing posP/B + guard + low_debt. Return chosen ticker or None."""
        if not guarded:
            return None
        cheapest = lambda ts: pb.loc[date, ts].idxmin() if ts else None
        if mode == "C_value":
            return cheapest(guarded)
        if mode == "A_capitul":
            rs = rsi.loc[date, [g for g in guarded if pd.notna(rsi.loc[date, g])]]
            return rs.idxmin() if len(rs) else cheapest(guarded)
        if mode == "A_value":
            deep = [g for g in guarded if pd.notna(rsi.loc[date, g]) and rsi.loc[date, g] < 30]
            return cheapest(deep) if deep else cheapest(guarded)
        if mode == "B_dip":
            dip = [g for g in guarded if bool(above200.loc[date, g]) and pd.notna(rsi.loc[date, g]) and rsi.loc[date, g] < 45]
            return cheapest(dip) if dip else cheapest(guarded)
        if mode == "best_regime":
            if bool(spy_riskoff.loc[date]):
                deep = [g for g in guarded if pd.notna(rsi.loc[date, g]) and rsi.loc[date, g] < 30]
                return cheapest(deep) if deep else cheapest(guarded)
            dip = [g for g in guarded if bool(above200.loc[date, g]) and pd.notna(rsi.loc[date, g]) and rsi.loc[date, g] < 45]
            return cheapest(dip) if dip else cheapest(guarded)
        return cheapest(guarded)

    MODES = ["C_value", "A_capitul", "A_value", "B_dip", "best_regime"]

    def run(mode):
        rets, spies, pk = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                guarded = [c for c in cands if bool(low_debt.loc[date, c])] or cands
                p = pick(mode, date, guarded)
                if p is None:
                    continue
                r = _ret_delist(px[p], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r)); pk.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp))
        s = _stats(rets, spies)
        s["pick_win_pct"] = round(float((np.array(pk) > 0).mean() * 100), 1) if pk else 0
        return s

    results = {m: run(m) for m in MODES}
    base = results["C_value"]
    print("\n=== A vs B vs VALUE inside the acceleration sectors ===", flush=True)
    for m in MODES:
        s = results[m]
        d = "" if m == "C_value" else f"  ({'+' if s['vs_spy']-base['vs_spy']>=0 else ''}{round(s['vs_spy']-base['vs_spy'],1)}pp)"
        print(f"  {m:12} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  win {s['pick_win_pct']}%{d}", flush=True)

    best = max(MODES, key=lambda m: results[m]["vs_spy"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "results": results, "best": best,
        "verdict": (f"Best per-sector rule = {best} ({results[best]['vs_spy']}% vs C_value {base['vs_spy']}%). "
                    + ("An A/B overlay beats straight value inside the accel sectors." if results[best]["vs_spy"] > base["vs_spy"] + 20
                       else "Straight VALUE (cheapest-P/B) is hard to beat inside the accel sectors; A/B pickers "
                       "don't add — the value pick already captures the deep-weakness edge A chases, and B's uptrend "
                       "filter is momentum-confirmation which subtracts.")),
        "caveat": "In-sample, no fees, ~5y, 12mo warmup. A/B fall back to C when their filter is empty.",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="ab_vs_value", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                          "computed_at": timezone.now()})
        print("Saved BacktestResult[ab_vs_value]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
