#!/usr/bin/env python3
"""DECISIVE GATE for the iv_skew signal: a STANDALONE time-stepped LONG book of HIGH-iv_skew names (fear-priced /
put-demand), net of cost, vs SPY, WITH regime attribution — the exact test that exposed the insider signal as
pro-cyclical beta. iv_skew was scouted-positive (high skew -> fwd outperformance, monthly + daily) but never
taken to a standalone book; as a flagship OVERLAY it was redundant. This resolves it.

Signal: a name is 'on' if its most-recent OptionSnapshot.iv_skew (as-of the bar, <=5 trading days stale) is in
the TOP QUINTILE of the pooled skew distribution (fear-priced). Enter next bar (PIT), hold N days, equal-weight
across concurrent positions, gross capped 1x (cash when none), net 10bps turnover. REGIME: split daily P&L by
SPY>200dMA — is the drift real alpha or bull-only beta? Variants: all-cap / mega+large only; hold 10/20.
Benchmark SPY same window. Persists BacktestResult[uoa_skew_book]+JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/uoa_skew_book.py"""
import os, sys, json, bisect, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from django.db import connection
with connection.cursor() as cur:
    cur.execute("SET max_parallel_workers_per_gather = 0")
from seq_fundamental_study import load_candles
from signal_discovery import _universe

FEE, DVOL_FLOOR = 1e-3, 5e6
HOLDS = [10, 20]
STALE = 5           # a snapshot is usable for up to 5 trading days
MEGA_LARGE = 200e6  # cap-restricted variant floor


def main():
    from core.models import OptionSnapshot, Candle
    universe, _ = _universe()
    snap = pd.DataFrame(list(OptionSnapshot.objects.filter(iv_skew__isnull=False)
                             .values_list("ticker", "date", "iv_skew")), columns=["ticker", "date", "iv_skew"])
    snap = snap[snap["ticker"].isin(set(universe))]
    snap["date"] = pd.to_datetime(snap["date"])
    thr = float(np.nanpercentile(snap["iv_skew"].values, 80))     # top-quintile skew (fear-priced)
    print(f"iv_skew snapshots {len(snap)}; top-quintile threshold (q80) = {thr:.2f}", flush=True)
    hi = snap[snap["iv_skew"] >= thr]
    tickers = sorted(set(hi["ticker"]))
    print(f"high-skew name-days {len(hi)} over {len(tickers)} names", flush=True)

    daily = load_candles(tickers + ["SPY"])
    close = pd.DataFrame({t: daily[t]["Close"] for t in daily if t != "SPY"}).sort_index()
    volp = pd.DataFrame({t: daily[t]["Volume"] for t in daily if t != "SPY"}).reindex_like(close)
    R = close.pct_change().fillna(0.0)
    dvol = (close * volp).rolling(20).mean()
    idx = close.index
    dates = [d.date() for d in idx]
    col = {t: i for i, t in enumerate(close.columns)}

    spy = daily["SPY"]["Close"].reindex(idx, method="ffill")
    regime_up = (spy > spy.rolling(200).mean()).fillna(False).values
    spy_ret = spy.pct_change().fillna(0.0)

    # high-skew events -> entry bar (next trading day on/after snapshot date)
    hi_by_tk = {tk: sorted(g["date"].dt.date.tolist()) for tk, g in hi.groupby("ticker")}

    def entry_matrix(cap_floor):
        E = np.zeros(close.shape); ntr = 0
        for tk, ds in hi_by_tk.items():
            if tk not in col:
                continue
            c = col[tk]
            for fd in ds:
                j = bisect.bisect_left(dates, fd)
                if j >= len(dates):
                    continue
                dv = dvol.iat[j, c]
                if not np.isfinite(dv) or dv < cap_floor:
                    continue
                if E[j, c] == 0.0:
                    ntr += 1
                E[j, c] = 1.0
        return E, ntr

    # options-era mask (2022-09-01+) for the apples-to-apples comparison
    era = pd.Series(idx >= pd.Timestamp("2022-09-01"), index=idx).values

    def _stats(net_series, mask=None):
        net = net_series[mask] if mask is not None else net_series
        eq = (1 + net).cumprod(); yrs = len(net) / 252.0
        return dict(total=round((eq.iloc[-1] - 1) * 100, 1),
                    cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1) if eq.iloc[-1] > 0 else -100.0,
                    dd=round(((eq / eq.cummax()) - 1).min() * 100, 1),
                    sharpe=round(net.mean() / net.std() * np.sqrt(252), 2) if net.std() > 0 else 0.0)

    def run(cap_floor, hold):
        E_arr, ntr = entry_matrix(cap_floor)
        E = pd.DataFrame(E_arr, index=idx, columns=close.columns)
        held = (E.shift(1).rolling(hold, min_periods=1).max().fillna(0) > 0)
        hs = held.sum(axis=1)
        W = held.div(hs.where(hs > 0, np.nan), axis=0).fillna(0.0)
        port = (W.shift(1) * R).sum(axis=1)
        turn = (W - W.shift(1)).abs().sum(axis=1)
        net = port - turn * FEE
        up = net[regime_up]; dn = net[~pd.Series(regime_up, index=idx).values]
        full = _stats(net); eraS = _stats(net, era)
        eraSPY = _stats(spy_ret, era)
        return dict(ntrades=ntr, total=full["total"], cagr=full["cagr"], dd=full["dd"], sharpe=full["sharpe"],
                    inv=round((hs > 0).mean() * 100, 0), avgn=round(float(hs[hs > 0].mean()), 1),
                    up_bpday=round(up.mean() * 1e4, 1), dn_bpday=round(dn.mean() * 1e4, 1),
                    up_days=int(regime_up.sum()), dn_days=int((~regime_up).sum()),
                    era_total=eraS["total"], era_cagr=eraS["cagr"], era_dd=eraS["dd"], era_sharpe=eraS["sharpe"],
                    era_spy_total=eraSPY["total"], era_spy_cagr=eraSPY["cagr"], era_spy_dd=eraSPY["dd"],
                    era_spy_sharpe=eraSPY["sharpe"], era_vs_spy=round(eraS["cagr"] - eraSPY["cagr"], 1))

    spy_eq = (1 + spy_ret).cumprod()
    spy_tot = round((spy_eq.iloc[-1] - 1) * 100, 1)
    spy_dd = round(((spy_eq / spy_eq.cummax()) - 1).min() * 100, 1)
    print(f"\nwindow {idx[0].date()}..{idx[-1].date()}  SPY {spy_tot:+.0f}% DD {spy_dd:.1f}%\n", flush=True)
    print("APPLES-TO-APPLES on the LIVE options window (2022-09+): book vs SPY, both fully live.\n", flush=True)
    print(f"  {'variant':16}{'hold':>5}{'trades':>7} | {'ERA book%':>10}{'CAGR':>6}{'DD':>6}{'Sh':>5}"
          f" | {'ERA SPY%':>9}{'CAGR':>6}{'DD':>6}{'Sh':>5} | {'vsSPY':>6}  regime bp/day(bull|bear)", flush=True)
    out = {"spy_fullwin": dict(total=spy_tot, dd=spy_dd), "window": f"{idx[0].date()}..{idx[-1].date()}",
           "skew_threshold_q80": round(thr, 2), "variants": {}}
    for label, floor in (("all-cap$5M", DVOL_FLOOR), ("mega+large$200M", MEGA_LARGE)):
        for hold in HOLDS:
            r = run(floor, hold)
            out["variants"][f"{label}|h{hold}"] = r
            print(f"  {label:16}{hold:>5}{r['ntrades']:>7} | {r['era_total']:>10.0f}{r['era_cagr']:>6.1f}"
                  f"{r['era_dd']:>6.1f}{r['era_sharpe']:>5.2f} | {r['era_spy_total']:>9.0f}{r['era_spy_cagr']:>6.1f}"
                  f"{r['era_spy_dd']:>6.1f}{r['era_spy_sharpe']:>5.2f} | {r['era_vs_spy']:>+6.1f}"
                  f"  {r['up_bpday']:>5.1f}|{r['dn_bpday']:>5.1f}", flush=True)

    out["caveat"] = ("Standalone LONG book of top-quintile iv_skew (fear-priced) names, net 10bps, EW gross 1x, "
                     "hold 10/20d, entry next bar after snapshot. Regime split SPY>200dMA. Options era 2022-09+ "
                     "(single provider, ~4yr one+ regime). Mirrors f_insider_book for comparability.")
    from pathlib import Path
    Path("/app/.data/studies/uoa_skew_book.json").write_text(json.dumps(out, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="uoa_skew_book",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[uoa_skew_book]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
