#!/usr/bin/env python3
"""TRUE (daily mark-to-market) drawdown of the flagship (user, 2026-08-19). The engine only produces MONTH-END
returns, so its reported DD (-23.4%) is monthly-marked and CANNOT see intra-month troughs — it misses the
Feb/Mar-2020 crash almost entirely (monthly close ~-12% vs intra-month ~-34% on SPY, worse on small-caps).
This reconstructs the flagship's DAILY equity curve: hold each month's weighted picks and mark them to market
every trading day from entry, chaining month-to-month, then compute the real peak-to-trough.

Reads .data/studies/flagship_history.json (the wired tl_rsi flagship, 104,939%).
-> prints daily vs monthly DD + worst episodes; BacktestResult[flagship_daily_dd].
Run: docker exec -w /app rotation-backend-1 python -u flagship_daily_dd.py

Caveat: uses each name's (split/div-adjusted) daily CLOSE in its LOCAL currency — a single name's daily return
is ~FX-invariant intra-month, so this is a faithful DD for the US/CA-heavy book (FX drift over a month is
immaterial to trough depth). Delisted-mid-hold names hold their last traded price (matches _ret_delist)."""
import os, sys, json, warnings
sys.path.insert(0, "/app")
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import pandas as pd
from pathlib import Path
from seq_fundamental_study import load_candles

TRACE = Path("/app/.data/studies/flagship_history.json")


def _maxdd(eq):
    eq = np.asarray(eq, float)
    return float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    D = json.load(open(TRACE))
    months = [m for m in D["months"] if m.get("ndate") and m.get("picks")]
    tickers = sorted({p["ticker"] for m in months for p in m["picks"] if p.get("ticker")})
    cand = load_candles(tickers + ["SPY"])
    spy = cand["SPY"]["Close"]
    print(f"trace {len(months)} months, {len(tickers)} names, monthly total {D['perf']['total']}%", flush=True)

    val = 1.0
    mbooks = []            # per-month daily cumulative book-return series (since entry) — for the stop-loss sim
    dcurve = []            # (Timestamp, portfolio value)  — daily
    mcurve = []            # (Timestamp, value) — month-end only (to reproduce the -23.4% monthly DD)
    spy_curve = []         # daily SPY, same calendar, for reference
    val0_spy = None
    for m in months:
        d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"])
        picks = [(p["ticker"], float(p["weight"])) for p in m["picks"] if p.get("ticker") and p.get("weight")]
        if not picks:
            continue
        days = spy.loc[d0:d1].index                    # trading-day calendar for the hold window (incl d0)
        if len(days) < 2:
            continue
        w = pd.Series(dict(picks), dtype=float)
        # daily close for each held name over the window, ffilled (delisted -> last price holds), entry = close at d0
        px = pd.DataFrame({t: cand[t]["Close"].reindex(days, method="ffill") for t, _ in picks if t in cand})
        w = w.reindex(px.columns).dropna(); px = px[w.index]
        ent = px.iloc[0]                               # entry price at d0 (= prior month-end close)
        ok = ent[ent > 0].index; px = px[ok]; w = w[ok]; ent = ent[ok]
        if not len(w):
            continue
        rel = px.div(ent, axis=1) - 1.0                # each name's return since entry, per day
        book = (rel.mul(w, axis=1)).sum(axis=1) / w.sum()   # weighted book return since d0, per day
        mbooks.append(book)
        vals = val * (1.0 + book)
        # append daily marks for days AFTER d0 (d0 is the prior month-end, already in the curve)
        for ts, v in vals.iloc[1:].items():
            dcurve.append((ts, float(v)))
        val = float(vals.iloc[-1])
        mcurve.append((d1, val))
        # SPY reference on the same daily calendar
        s = spy.reindex(days, method="ffill"); s0 = s.iloc[0]
        if val0_spy is None:
            val0_spy = 1.0; _sv = 1.0
        else:
            _sv = spy_curve[-1][1] if spy_curve else 1.0
        for ts, sp in (s / s0 * _sv).iloc[1:].items():
            spy_curve.append((ts, float(sp)))

    dts = [t for t, _ in dcurve]; deq = [v for _, v in dcurve]
    # WORST SINGLE-DAY book return (user) — overall + the 6 worst days (COVID should dominate)
    _dr = np.diff(np.asarray(deq)) / np.asarray(deq)[:-1]
    _order = np.argsort(_dr)[:6]
    print("\n  WORST SINGLE-DAY book returns (daily mark-to-market):", flush=True)
    for _i in sorted(_order, key=lambda k: _dr[k]):
        print(f"    {dts[_i + 1].date()}  {_dr[_i] * 100:+6.1f}%", flush=True)
    daily_dd = _maxdd(deq)
    monthly_dd = _maxdd([v for _, v in mcurve])
    # worst peak-to-trough episode on daily marks
    eq = np.asarray(deq); peak = np.maximum.accumulate(eq); ddser = eq / peak - 1
    ti = int(ddser.argmin()); trough_dt = dts[ti]
    pk_i = int(np.argmax(eq[:ti + 1])); peak_dt = dts[pk_i]
    # per-calendar-year worst daily DD
    print(f"\n=== FLAGSHIP TRUE DRAWDOWN (daily mark-to-market) ===", flush=True)
    print(f"  monthly-marked DD (what the engine reports): {monthly_dd:6.1f}%", flush=True)
    print(f"  DAILY-marked DD (the real number):           {daily_dd:6.1f}%", flush=True)
    print(f"  worst episode: peak {peak_dt.date()} -> trough {trough_dt.date()}  ({daily_dd:.1f}%)", flush=True)
    df = pd.DataFrame({"v": eq}, index=pd.to_datetime(dts))
    # EX-COVID drawdown (user: "that's covid, doesn't count"): worst peak-to-trough EXCLUDING the Jan-Jun 2020
    # crash+recovery. Measured independently on the pre-COVID segment and the post-recovery segment (peak resets
    # after COVID since the book made new all-time highs), so no stitch artifact.
    _cov0, _cov1 = pd.Timestamp("2020-01-16"), pd.Timestamp("2020-06-30")
    pre = df.loc[:_cov0, "v"].values
    post = df.loc[_cov1:, "v"].values
    dd_pre = _maxdd(pre) if len(pre) > 1 else 0.0
    dd_post = _maxdd(post) if len(post) > 1 else 0.0
    dd_excov = min(dd_pre, dd_post)
    # worst ex-covid episode (whichever segment) + its dates
    seg = df.loc[_cov1:] if dd_post <= dd_pre else df.loc[:_cov0]
    _e = seg["v"].values; _pk = np.maximum.accumulate(_e); _dd = _e / _pk - 1
    _ti = int(_dd.argmin()); _pi = int(np.argmax(_e[:_ti + 1]))
    excov_peak, excov_trough = seg.index[_pi], seg.index[_ti]
    # post-2021 (fully modern regime, no COVID at all)
    dd_2021 = _maxdd(df.loc["2021-01-01":, "v"].values)
    print(f"\n  EX-COVID DD (exclude Jan-Jun 2020):          {dd_excov:6.1f}%   "
          f"({excov_peak.date()} -> {excov_trough.date()})", flush=True)
    print(f"  2021+ only (fully modern regime):            {dd_2021:6.1f}%", flush=True)
    print("\n  worst DAILY drawdown by calendar year:", flush=True)
    for y, g in df.groupby(df.index.year):
        e = g["v"].values
        print(f"    {y}: {_maxdd(e):6.1f}%", flush=True)

    # ── STOP-LOSS SIM (user): liquidate to CASH for the rest of the month when the book (a) drops >=X% in a SINGLE
    # DAY, or (b) is down >=X% from month-entry (intra-month drawdown), then re-enter at next month-end. ──
    def _sim(trig):
        v = 1.0; eq = []
        for bk in mbooks:
            v *= (1.0 + trig(bk)); eq.append(v)
        r = np.array([eq[0] - 1] + [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))])
        sh = r.mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 1e-9 else 0.0
        return (v - 1) * 100, sh, _maxdd(eq)
    def _none(bk): return float(bk.iloc[-1])
    def _cum(thr):                                   # exit first day book (since entry) <= thr; realize that level
        def f(bk):
            hit = bk[bk <= thr]
            return float(hit.iloc[0]) if len(hit) else float(bk.iloc[-1])
        return f
    def _day(thr):                                   # exit first day a SINGLE-DAY return <= thr; realize cum thru it
        def f(bk):
            dr = (1.0 + bk).pct_change()
            idx = dr[dr <= thr].index
            return float(bk.loc[idx[0]]) if len(idx) else float(bk.iloc[-1])
        return f
    bt, bsh, bdd = _sim(_none)
    print("\n=== STOP-LOSS: liquidate to cash on trigger, re-enter next month-end ===", flush=True)
    print(f"  {'policy':28}{'total':>12}{'Sharpe':>8}{'dailyDD':>9}", flush=True)
    print(f"  {'NO STOP (flagship)':28}{bt:>11.0f}%{bsh:>8.2f}{bdd:>8.1f}%", flush=True)
    for thr in (-0.10, -0.15, -0.20):
        t, s, dd = _sim(_day(thr))
        print(f"  {('single-day <= '+str(int(thr*100))+'%'):28}{t:>11.0f}%{s:>8.2f}{dd:>8.1f}%", flush=True)
    for thr in (-0.10, -0.15, -0.20):
        t, s, dd = _sim(_cum(thr))
        print(f"  {('down '+str(int(thr*100))+'% from entry'):28}{t:>11.0f}%{s:>8.2f}{dd:>8.1f}%", flush=True)

    try:
        from core.models import BacktestResult
        from django.utils import timezone
        payload = {"daily_dd_pct": daily_dd, "monthly_dd_pct": monthly_dd,
                   "worst_peak": str(peak_dt.date()), "worst_trough": str(trough_dt.date()),
                   "by_year": {int(y): _maxdd(g["v"].values) for y, g in df.groupby(df.index.year)},
                   "monthly_total_pct": D["perf"]["total"], "n_months": len(months)}
        BacktestResult.objects.update_or_create(
            kind="flagship_daily_dd", defaults={"payload": json.loads(json.dumps(payload, default=str)),
                                                "computed_at": timezone.now()})
        print("\nSaved BacktestResult[flagship_daily_dd]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
