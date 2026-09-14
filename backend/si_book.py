#!/usr/bin/env python3
"""DECISIVE GATE for the short-interest anomaly: a monthly cross-sectional LONG book of LOW si_pct_float
(lightly-shorted / uncrowded) names, net of cost, vs SPY, WITH regime attribution (SPY>200dMA). The screen
(si_screen) showed low-SI outperforms high-SI monotonically across caps, multi-regime 2017-2026, and low-SI
MEGA-caps beat SPY (+1.25%/63d t4.1). Unlike insider/skew (capitulation = pro-cyclical), low-SI = quality/
uncrowded, so it MIGHT hold in bear regimes. This resolves it.

Each month-end m: for every tradable name (dvol>=$5M, price>floor) with a DISSEMINATED si_pct_float as-of m
(settlement + 10 trading-day lag), rank cross-sectionally; go long the BOTTOM QUINTILE (lowest SI%). Equal-weight,
hold 1 month, net 10bps turnover-ish (charge round-trip on the rebalanced fraction). Compare to SPY same months;
split monthly P&L by SPY>200dMA. Variants: all-cap / mega+large only; also report the HIGH-SI (top-quintile)
book to confirm the short leg underperforms. si_pct_float = 100*short_interest/shares_outstanding (PIT shares).
Persists BacktestResult[si_book]+JSON. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/si_book.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

SI_FILE = Path("/app/.data/short_interest.jsonl")
DVOL_FLOOR = 5e6
PUB_LAG = 10
FEE = 10 / 1e4          # round-trip on turnover
MEGA_LARGE = 200e6


def perf(mret, ann=12):
    mret = mret.dropna()
    if len(mret) < 12:
        return None
    eq = (1 + mret.values).cumprod(); yrs = len(mret) / ann
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = mret.mean() / mret.std() * math.sqrt(ann) if mret.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return dict(total=round((eq[-1] - 1) * 100, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), maxdd=round(dd, 1))


def main():
    universe, _ = _universe()
    from core.models import Candle
    have_candles = set(Candle.objects.values_list("ticker", flat=True).distinct())
    # SI panel
    si = defaultdict(list)
    with SI_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            tk = r["ticker"]
            if tk in have_candles and tk in set(universe) and r.get("settlement_date") and r.get("short_interest"):
                si[tk].append((pd.Timestamp(r["settlement_date"]), float(r["short_interest"])))
    for tk in si:
        si[tk].sort()
    names = sorted(si)
    print(f"tradable SI names: {len(names)}", flush=True)

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_m = spy_c.resample("ME").last()
    spy_mret = spy_m.pct_change()
    spy_ma200 = spy_c.rolling(200).mean()

    # monthly rebalance dates from SPY
    months = spy_m.index

    def shares_ff(reps, tk, dates):
        r = reps.get(tk)
        if r is None or "shares_outstanding" not in r.columns:
            return None
        d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
        if d.empty:
            return None
        s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(s.index.union(pd.DatetimeIndex(dates))).ffill().reindex(pd.DatetimeIndex(dates)).values

    # per (month) -> list of (ticker, si_pct_float, cap_dvol, fwd_1m_ret)
    panel = defaultdict(list)
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS or tk not in si:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            di = dates.values
            cl_m = pd.Series(close, index=dates).resample("ME").last()
            dv_m = pd.Series(dvol20, index=dates).resample("ME").last()
            sh = shares_ff(reps, tk, cl_m.index)
            obs = si[tk]
            obs_dates = [o[0] for o in obs]
            for mi, m in enumerate(cl_m.index):
                if mi + 1 >= len(cl_m):
                    continue
                px = cl_m.iloc[mi]; pxn = cl_m.iloc[mi + 1]
                if not (np.isfinite(px) and np.isfinite(pxn) and px > PRICE_FLOOR):
                    continue
                dv = dv_m.iloc[mi]
                if not (np.isfinite(dv) and dv >= DVOL_FLOOR):
                    continue
                # latest disseminated SI as-of m: settlement + PUB_LAG trading days <= m
                # find most recent obs whose settlement is at least PUB_LAG trading days before m
                cutoff_idx = int(np.searchsorted(di, np.datetime64(m), side="right")) - 1 - PUB_LAG
                if cutoff_idx < 0:
                    continue
                cutoff_date = pd.Timestamp(di[cutoff_idx])
                k = np.searchsorted(obs_dates, cutoff_date, side="right") - 1
                if k < 0:
                    continue
                sint = obs[k][1]
                if sh is None or not np.isfinite(sh[mi]) or sh[mi] <= 0:
                    continue
                sipf = 100.0 * sint / sh[mi]
                fwd = pxn / px - 1.0
                panel[m].append((tk, sipf, float(dv), fwd))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    def build(cap_floor, leg="low", qn=5):
        """leg='low' -> long bottom-quintile si_pct_float; 'high' -> top-quintile. Returns monthly return series."""
        rets = {}; navg = []
        for m in sorted(panel):
            rows = [(tk, s, dv, fwd) for (tk, s, dv, fwd) in panel[m] if dv >= cap_floor]
            if len(rows) < 25:
                continue
            svals = np.array([r[1] for r in rows])
            qe = np.quantile(svals, [0.2, 0.8])
            if leg == "low":
                sel = [r for r in rows if r[1] <= qe[0]]
            else:
                sel = [r for r in rows if r[1] >= qe[1]]
            if len(sel) < 5:
                continue
            rets[m] = float(np.mean([r[3] for r in sel])) - FEE      # EW, charge ~1 side/mo turnover proxy
            navg.append(len(sel))
        s = pd.Series(rets).sort_index()
        return s, (float(np.mean(navg)) if navg else 0)

    def regime_split(mret):
        up = []; dn = []
        for m, r in mret.items():
            ma = spy_ma200.reindex([m], method="ffill").iloc[0]
            sp = spy_c.reindex([m], method="ffill").iloc[0]
            (up if (np.isfinite(ma) and sp > ma) else dn).append(r)
        return (round(np.mean(up) * 100, 2) if up else None, len(up),
                round(np.mean(dn) * 100, 2) if dn else None, len(dn))

    out = {"variants": {}}
    print(f"\n=== SHORT-INTEREST LONG BOOK — monthly cross-sectional, net {int(FEE*1e4)}bps, vs SPY + regime ===", flush=True)
    print(f"  {'variant':22}{'nmo':>5}{'avgN':>6}{'total%':>9}{'CAGR%':>7}{'Sh':>6}{'maxDD%':>8}"
          f"{'vsSPY':>7}   bull%|bear% (monthly)", flush=True)
    for label, floor, leg in (("low-SI all-cap", DVOL_FLOOR, "low"), ("low-SI mega+large", MEGA_LARGE, "low"),
                              ("HIGH-SI all-cap", DVOL_FLOOR, "high")):
        mret, avgn = build(floor, leg)
        bp = perf(mret)
        if not bp:
            continue
        sp = perf(spy_mret.reindex(mret.index))
        bull, nu, bear, nd = regime_split(mret)
        out["variants"][label] = dict(nmo=len(mret), avgN=round(avgn, 1), book=bp, spy=sp,
                                      vs_spy=round(bp["cagr"] - sp["cagr"], 1), bull_mo=bull, bull_n=nu,
                                      bear_mo=bear, bear_n=nd)
        print(f"  {label:22}{len(mret):>5}{avgn:>6.0f}{bp['total']:>9.0f}{bp['cagr']:>7.1f}{bp['sharpe']:>6.2f}"
              f"{bp['maxdd']:>8.1f}{bp['cagr']-sp['cagr']:>+7.1f}   {bull}|{bear}  ({nu}u/{nd}d)", flush=True)
    # SPY reference over the panel window
    pm = sorted(panel)
    spref = perf(spy_mret.reindex(pd.DatetimeIndex(pm)))
    out["spy_ref"] = spref
    print(f"  {'SPY (same months)':22}{len(pm):>5}{'':>6}{spref['total']:>9.0f}{spref['cagr']:>7.1f}{spref['sharpe']:>6.2f}{spref['maxdd']:>8.1f}", flush=True)

    out["caveat"] = ("Short-interest LONG book, monthly cross-sectional, long bottom-quintile si_pct_float "
                     "(=100*SI/shares, PIT). Entry lagged 10td after settlement (dissemination). EW, net 10bps, "
                     "hold 1mo. Regime split SPY>200dMA. 2018-2026 (SI 2017-12+). HIGH-SI leg shown to confirm "
                     "short-side underperformance. vs SPY same months.")
    Path("/app/.data/studies/si_book.json").write_text(json.dumps(out, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="si_book",
            defaults={"payload": json.loads(json.dumps(out, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[si_book]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
