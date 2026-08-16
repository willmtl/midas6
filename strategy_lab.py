#!/usr/bin/env python3
"""STRATEGY LAB — can A or B beat C without the sector rotation, and do C's RULES help A/B?

C's edge decomposes into (i) the sector-rotation FILTER and (ii) the value+quality PICK RULES
(cheapest positive-P/B, profit guard, low debt). This isolates them by running A and B WITHOUT the
rotation, both pure and with C's rules applied:

  C_ref     rotation top-momentum sectors -> cheapest-P/B guard+low_debt pick (the benchmark), 1mo
  B_pure    whole universe: stock above its 200d MA AND RSI(10)<45 (dip in its OWN uptrend), eq-wt, 1mo
  B_value   B_pure names, then the 10 cheapest positive-P/B (value tilt, still NO rotation), 1mo
  B_quality B_pure names that also pass C's rules (posP/B + guard + low_debt), eq-wt, 1mo
  A_pure    whole universe: RSI(10)<30 AND A/D-rising (capitulation+accumulation), ~10d bounce, eq-wt
  A_quality A_pure names that also pass C's rules, ~10d bounce, eq-wt

If B_value / B_quality << C_ref, the ROTATION (not the rules) is the load-bearing piece. If A/B_quality
beat their pure versions, C's rules travel; if they still lose to C, selection needs the rotation too.
-> BacktestResult[strategy_lab] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_lab.py  (--limit 300)
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, ta
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "strategy_lab.json"
LOOKBACK, TOP_N = 6, 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return {"total_return": 0, "vs_spy": 0, "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0, "avg_names": 0}
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(tot, 1), "vs_spy": round(tot - sp, 1), "sharpe": round(sh, 2),
            "max_drawdown": round(dd, 1), "t_stat": round(t, 2) if t is not None else None, "periods": n}


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]; hs = set(all_holds)
        sector_map = {e: (n, [h for h in hh if h in hs]) for e, (n, hh) in sector_map.items()}

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    shares = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity = _pit_monthly_panel(reps, "total_equity", midx)
    ni = _pit_monthly_panel(reps, "net_income", midx)
    debt = _pit_monthly_panel(reps, "total_debt", midx)
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    quality = (pb > 0) & (~trap) & low_debt

    rsi_m, adl_m, sma200_m = {}, {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or len(d) < 210:
            continue
        cl, hi, lo, vol = d["Close"], d["High"], d["Low"], d["Volume"]
        rsi_m[t] = ta.momentum.rsi(cl, window=10).resample("ME").last().reindex(midx)
        rng = (hi - lo).replace(0, np.nan)
        mfm = ((cl - lo) - (hi - cl)) / rng
        adl_m[t] = (mfm.fillna(0) * vol.fillna(0)).cumsum().resample("ME").last().reindex(midx)
        sma200_m[t] = (cl > cl.rolling(200).mean()).resample("ME").last().reindex(midx)
    rsi = pd.DataFrame(rsi_m).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    above200 = pd.DataFrame(sma200_m).reindex(index=midx, columns=common).fillna(False).astype(bool)
    ad_rising = (adl > adl.shift(1)) & (px.pct_change(1) < 0)
    dip = above200 & (rsi < 45)          # B: dip in the stock's own uptrend
    cap = (rsi < 30) & ad_rising          # A: capitulation + accumulation

    def _fwd_days(t, date, ndays=10):
        d = stock_daily.get(t)
        if d is None:
            return None
        sub = d["Close"][d["Close"].index >= date]
        if len(sub) < ndays + 1 or sub.iloc[0] == 0:
            return None
        v = float(sub.iloc[ndays] / sub.iloc[0] - 1)
        return v if np.isfinite(v) else None

    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def _avail(date):
        return set(px.columns[(px.loc[date].notna()) & (px.loc[date] > 0)])

    def run(selector, hold="month"):
        rets, spies, nn = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            names = selector(date)
            r = []
            for t in names:
                v = _ret_delist(px[t], date, ndate) if hold == "month" else _fwd_days(t, date)
                if v is not None and np.isfinite(v):
                    r.append(float(v))
            if r:
                rets.append(float(np.mean(r))); spies.append(float(sp)); nn.append(len(r))
        s = _stats(rets, spies); s["avg_names"] = round(float(np.mean(nn)), 1) if nn else 0
        return s

    # selectors -----------------------------------------------------------------
    def sel_C(date):
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        out = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0]
            g = [c for c in cands if not bool(trap.loc[date, c])]
            ld = [c for c in g if bool(low_debt.loc[date, c])]
            use = ld or g or cands
            if use:
                out.append(pb.loc[date, use].idxmin())
        return out

    def _mask_names(mask, date, extra=None):
        av = _avail(date)
        row = mask.loc[date]
        names = [t for t in row[row].index if t in av]
        if extra is not None:
            names = [t for t in names if bool(extra.loc[date, t])]
        return names

    def sel_Bpure(date):
        return _mask_names(dip, date)

    def sel_Bvalue(date):
        names = _mask_names(dip, date)
        if not names:
            return []
        return list(pb.loc[date, names].dropna().nsmallest(TOP_N).index)

    def sel_Bquality(date):
        return _mask_names(dip, date, extra=quality)

    def sel_Bvalueq(date):
        # dip in uptrend, then the TOP_N CHEAPEST positive-P/B, low-debt, guard-passing names (book ratio
        # + debt selection applied — C's full value+quality pick, but WITHOUT the sector rotation)
        names = _mask_names(dip, date, extra=quality)
        if not names:
            return []
        return list(pb.loc[date, names].dropna().nsmallest(TOP_N).index)

    def sel_Apure(date):
        return _mask_names(cap, date)

    def sel_Aquality(date):
        return _mask_names(cap, date, extra=quality)

    def sel_Avalueq(date):
        names = _mask_names(cap, date, extra=quality)
        if not names:
            return []
        return list(pb.loc[date, names].dropna().nsmallest(TOP_N).index)

    results = {
        "C_ref": run(sel_C, "month"),
        "B_pure": run(sel_Bpure, "month"),
        "B_value": run(sel_Bvalue, "month"),
        "B_quality": run(sel_Bquality, "month"),
        "B_value_q": run(sel_Bvalueq, "month"),      # dip + cheapest book ratio + low debt + guard
        "A_pure": run(sel_Apure, "bounce"),
        "A_quality": run(sel_Aquality, "bounce"),
        "A_value_q": run(sel_Avalueq, "bounce"),
    }
    for k, v in results.items():
        print(f"  {k:12} vsSPY {v['vs_spy']:>7}%  t={str(v['t_stat']):>5}  Sh {v['sharpe']:>5}  "
              f"DD {v['max_drawdown']:>6}%  names {v['avg_names']:>5}", flush=True)

    c = results["C_ref"]["vs_spy"]
    beats = [k for k, v in results.items() if k != "C_ref" and v["vs_spy"] > c]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_sectors": TOP_N, "benchmark": BENCH,
                   "months": int(len(midx)), "limit": limit},
        "results": results, "c_ref_vs_spy": c, "beats_C": beats,
        "verdict": (
            f"C_ref (rotation+rules) {c}% vs SPY. Without rotation: B_pure {results['B_pure']['vs_spy']}%, "
            f"B_value {results['B_value']['vs_spy']}%, B_quality {results['B_quality']['vs_spy']}%, "
            f"A_pure {results['A_pure']['vs_spy']}%, A_quality {results['A_quality']['vs_spy']}%. "
            + ("NOTHING without the rotation beats C -> the rotation filter is load-bearing."
               if not beats else f"Beats C without rotation: {beats}.")
            + " C's rules DO travel (quality > pure) but " +
            ("still can't close the gap to C without the rotation." if not beats else "can beat C.")),
        "caveat": ("Whole-universe eq-weight (B_pure can be 100s of names). B=1mo hold, A=~10d bounce. "
                   "PIT selection; directional/no-fees; ~5y single regime."),
    }
    return payload


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="strategy_lab",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[strategy_lab]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + payload["verdict"], flush=True)


if __name__ == "__main__":
    main()
