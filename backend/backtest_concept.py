#!/usr/bin/env python3
"""'Is our concept actually working' backtest -> .data/studies/backtest_concept.json.

Phase 1 — TWO sector-rotation equity curves vs SPY, side by side, both holding ETFs (apples-to-apples),
  monthly rebalance, equal weight:
    (a) RSI rotation  = the LIVE dashboard rule: hold sectors that are BULLISH
        (RSI(10) > SMA(10) AND Omega(10) > 1) at month-end.
    (b) Momentum rotation = rank sectors by trailing 9-month return, hold the top 20 ETFs.
Phase 2 — OOS split: top signals' per-trade edge on IN-SAMPLE (older 70%) vs OUT-OF-SAMPLE (recent 30%).
Phase 3 — Top-signals portfolio: equal-weight, monthly-compounded equity curve of trading the top
  signals across the sector universe, vs SPY.

Run AFTER any heavy recompute finishes (avoids OOM competing for memory):
  docker exec -i rotation-backend-1 python -u backtest_concept.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import ta
from pathlib import Path
import config
from core.models import Sector, StockStudy
from studies import SIGNALS, EXITS, _rolling_omega, _tstat_from_returns

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "backtest_concept.json"
BENCH = config.BENCHMARK  # "SPY"
MOM_LOOKBACK = 9
MOM_TOPN = 20


def _load(dfs_tickers):
    from seq_fundamental_study import load_candles
    return load_candles(dfs_tickers)


def _bullish_monthend(df):
    """Month-end BULLISH state per the dashboard rule: RSI(10)>SMA(10) AND Omega(10)>1."""
    close = df["Close"]
    rsi = ta.momentum.rsi(close, window=10)
    sma = rsi.rolling(10).mean()
    omega = _rolling_omega(df)
    bull = ((rsi > sma) & (omega > 1)).fillna(False)
    return bull.resample("ME").last()


def _stats(rets, spy_rets):
    r = np.array(rets, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_return": 0, "spy_total": 0, "annual_return": 0, "sharpe": 0,
                "max_drawdown": 0, "t_stat": None, "periods": 0}
    total = float(np.prod(1 + r) - 1) * 100
    spy_total = float(np.prod(1 + np.array(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r)
    dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(total, 1), "spy_total": round(spy_total, 1),
            "annual_return": round(ann, 1), "sharpe": round(sharpe, 2),
            "max_drawdown": round(dd, 1), "t_stat": round(t, 2) if t is not None else None,
            "periods": n}


def _curve(rets, spy_rets, index):
    eq = np.cumprod(1 + np.array(rets)) if rets else []
    seq = np.cumprod(1 + np.array(spy_rets)) if spy_rets else []
    return [{"date": str(pd.Timestamp(d).date()), "strat": round(float(s), 4), "spy": round(float(sp), 4)}
            for d, s, sp in zip(index, eq, seq)]


def _monthly_frame(dfs, spy_df):
    m = {e: df["Close"].resample("ME").last() for e, df in dfs.items() if len(df) >= 60}
    mclose = pd.DataFrame(m).sort_index()
    spy_m = spy_df["Close"].resample("ME").last().reindex(mclose.index)
    return mclose, spy_m


def rotation_lab(dfs, spy_df):
    """Backtest a MENU of monthly, equal-weight, ETF-holding sector-rotation rules on one calendar,
    each vs SPY, ranked by total return. All start after a 12-month warmup so they're comparable."""
    mclose, spy_m = _monthly_frame(dfs, spy_df)
    fwd = mclose.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)
    trail = {k: mclose.pct_change(k) for k in (1, 3, 6, 9, 12)}
    spy_trail6 = spy_m.pct_change(6)
    vol = mclose.pct_change().rolling(6).std()
    mret = mclose.pct_change()
    trail_12_1 = mclose.shift(1) / mclose.shift(12) - 1          # classic 12-1 (skip most recent month)
    risk_adj = trail[9] / vol.replace(0, np.nan)                 # risk-adjusted momentum
    spy_sma10 = spy_m.rolling(10).mean()                         # SPY 10-mo trend for a risk-off filter

    def _sortino(x):
        dd = np.sqrt(np.mean(np.minimum(x, 0) ** 2))
        return x.mean() / dd if dd > 1e-9 else np.nan
    sortino = mret.rolling(12).apply(_sortino, raw=True)         # 12-mo Sortino per sector

    bull = pd.DataFrame({e: _bullish_monthend(df) for e, df in dfs.items() if len(df) >= 60}) \
        .reindex(mclose.index).fillna(False)
    # RSI strength (rsi - its sma) at month-end, to cap the dashboard rule to the strongest BULLISH sectors
    spread = pd.DataFrame({
        e: (lambda r: (r - r.rolling(10).mean()).resample("ME").last())(ta.momentum.rsi(df["Close"], 10))
        for e, df in dfs.items() if len(df) >= 60}).reindex(mclose.index)
    # 200-day MA trend: month-end distance of Close above its 200d SMA (>0 = above = uptrend).
    sma200 = pd.DataFrame({
        e: (df["Close"] / df["Close"].rolling(200).mean() - 1).resample("ME").last()
        for e, df in dfs.items() if len(df) >= 200}).reindex(mclose.index)
    cols = list(mclose.columns)

    def _run(selector, hold=1, start_i=12):
        """Monthly loop; reselect holdings every `hold` months (carry between) — hold=3 = quarterly."""
        rets, spy_rets, idx, cur = [], [], [], []
        for i in range(start_i, len(mclose.index) - 1):
            sp = spy_fwd.iloc[i]
            if pd.isna(sp):
                continue
            if (i - start_i) % hold == 0:
                cur = selector(i)
            held = [e for e in cur if pd.notna(fwd.iloc[i].get(e))]
            pr = float(np.mean([fwd.iloc[i][e] for e in held])) if held else 0.0  # empty -> cash (~0)
            rets.append(pr); spy_rets.append(float(sp)); idx.append(mclose.index[i + 1])
        return rets, spy_rets, idx

    def _by_year(rets, spy_rets, idx):
        d = pd.DataFrame({"r": rets, "s": spy_rets}, index=pd.to_datetime(idx))
        o = {}
        for yr, g in d.groupby(d.index.year):
            sr = float(np.prod(1 + g["r"].values) - 1) * 100
            sp = float(np.prod(1 + g["s"].values) - 1) * 100
            o[str(int(yr))] = {"strat": round(sr, 1), "spy": round(sp, 1), "beats": sr > sp}
        return o

    def topn(series_i, n, ascending=False):
        return series_i.dropna().sort_values(ascending=ascending).head(n).index.tolist()

    def rsi_capN(i, n):
        b = [e for e in cols if bool(bull.iloc[i].get(e, False))]
        sr = spread.iloc[i]
        return sorted(b, key=lambda e: (sr.get(e) if pd.notna(sr.get(e)) else -9), reverse=True)[:n]

    # (label, selector, hold_months)
    specs = [
        ("RSI+Omega, all BULLISH (dashboard rule)", lambda i: [e for e in cols if bool(bull.iloc[i].get(e, False))], 1),
        ("RSI+Omega, capped top 10 by strength", lambda i: rsi_capN(i, 10), 1),
        ("RSI+Omega, capped top 5 by strength", lambda i: rsi_capN(i, 5), 1),
        ("Momentum 12mo, top 20", lambda i: topn(trail[12].iloc[i], 20), 1),
        ("Momentum 12mo, top 20 (quarterly rebal)", lambda i: topn(trail[12].iloc[i], 20), 3),
        ("Sortino-ranked 12mo, top 20", lambda i: topn(sortino.iloc[i], 20), 1),
        ("Sortino-ranked 12mo, top 20 (quarterly)", lambda i: topn(sortino.iloc[i], 20), 3),
        ("Momentum 9mo, top 20", lambda i: topn(trail[9].iloc[i], 20), 1),
        ("Momentum 9mo, top 20 (quarterly rebal)", lambda i: topn(trail[9].iloc[i], 20), 3),
        ("Momentum 6mo, top 20", lambda i: topn(trail[6].iloc[i], 20), 1),
        ("Momentum 3mo, top 20", lambda i: topn(trail[3].iloc[i], 20), 1),
        ("Momentum 12-1 (skip recent mo), top 20", lambda i: topn(trail_12_1.iloc[i], 20), 1),
        ("Risk-adjusted momentum (9mo/vol), top 20", lambda i: topn(risk_adj.iloc[i], 20), 1),
        ("Relative strength vs SPY (6mo), top 20", lambda i: [
            e for e in topn(trail[6].iloc[i], 40)
            if trail[6].iloc[i].get(e, -9) > (spy_trail6.iloc[i] if pd.notna(spy_trail6.iloc[i]) else 0)][:20], 1),
        ("Dual momentum (12mo>0), top 20", lambda i: [
            e for e in topn(trail[12].iloc[i], 20) if trail[12].iloc[i].get(e, -9) > 0], 1),
        ("Momentum 9mo + SPY-trend filter (cash in bear)", lambda i: (
            topn(trail[9].iloc[i], 20)
            if (pd.notna(spy_sma10.iloc[i]) and spy_m.iloc[i] > spy_sma10.iloc[i]) else []), 1),
        ("Above 200-day MA (trend filter), all", lambda i: [
            e for e in cols if pd.notna(sma200.iloc[i].get(e)) and sma200.iloc[i].get(e) > 0], 1),
        ("Above 200-day MA, top 20 by distance", lambda i: [
            e for e in topn(sma200.iloc[i], 20) if sma200.iloc[i].get(e, -9) > 0], 1),
        ("Above 200-day MA, top 10 by distance", lambda i: [
            e for e in topn(sma200.iloc[i], 10) if sma200.iloc[i].get(e, -9) > 0], 1),
        ("Low volatility, 20 calmest", lambda i: topn(vol.iloc[i], 20, ascending=True), 1),
        ("Mean reversion (1mo losers), bottom 20", lambda i: topn(trail[1].iloc[i], 20, ascending=True), 1),
        ("All sectors (equal-weight baseline)", lambda i: cols, 1),
    ]
    out = []
    for label, sel, hold in specs:
        r, s, idx = _run(sel, hold=hold)
        out.append({"label": label, "hold_months": hold, "summary": _stats(r, s),
                    "by_year": _by_year(r, s, idx), "curve": _curve(r, s, idx)})
    out.sort(key=lambda x: x["summary"]["total_return"], reverse=True)
    return out


def _etf_trades(dfs, sig_fn, exit_fn):
    """(entry_date, exit_date, ret_fraction, hold_days) for every fire across the ETF universe."""
    out = []
    for e, df in dfs.items():
        if len(df) < 60:
            continue
        try:
            sig = sig_fn(df).fillna(False)
        except Exception:
            continue
        close = df["Close"].values
        n = len(close)
        for d in sig[sig].index:
            i = df.index.get_loc(d)
            ex = exit_fn(df, i)
            if ex is None or ex <= i or ex >= n:
                continue
            ep = float(close[i])
            if ep <= 0:
                continue
            out.append((df.index[i], df.index[ex], (float(close[ex]) - ep) / ep, ex - i))
    return out


def _top_signals(k=5):
    out = []
    for r in (StockStudy.objects.filter(total_trades__gte=1000)
              .order_by("-avg_return").values("signal_key", "exit_key")):
        sk, ek = r["signal_key"], r["exit_key"]
        if sk in [s for s, _ in out] or sk not in SIGNALS or ek not in EXITS:
            continue
        out.append((sk, ek))
        if len(out) >= k:
            break
    return out


def phase2(dfs, top):
    per, alld = {}, []
    for sk, ek in top:
        t = _etf_trades(dfs, SIGNALS[sk][1], EXITS[ek][1])
        per[(sk, ek)] = t
        alld += [d for d, _, _, _ in t]
    if not alld:
        return {"signals": [], "cutoff_date": None}
    cutoff = pd.Series(alld).sort_values().iloc[int(len(alld) * 0.7)]

    def _ag(rr):
        rr = [x * 100 for x in rr]
        t = _tstat_from_returns(rr)
        return {"n": len(rr), "avg": round(float(np.mean(rr)), 2) if rr else None,
                "wr": round(sum(1 for x in rr if x > 0) / len(rr) * 100, 1) if rr else None,
                "t": round(t, 2) if t is not None else None}
    rows = []
    for (sk, ek), t in per.items():
        is_r = [ret for d, _, ret, _ in t if d <= cutoff]
        oos_r = [ret for d, _, ret, _ in t if d > cutoff]
        rows.append({"signal": SIGNALS[sk][0], "signal_key": sk, "exit": EXITS[ek][0],
                     "is": _ag(is_r), "oos": _ag(oos_r)})
    rows.sort(key=lambda x: (x["oos"]["t"] is not None, x["oos"]["t"] or -9), reverse=True)
    return {"cutoff_date": str(cutoff.date()), "signals": rows}


def phase3(dfs, spy_df, top):
    trades = []
    for sk, ek in top:
        trades += _etf_trades(dfs, SIGNALS[sk][1], EXITS[ek][1])
    if not trades:
        return None
    _, spy_m = _monthly_frame(dfs, spy_df)
    months = spy_m.index
    # each month: mean per-trade monthly-ized return over trades OPEN that month (equal weight)
    open_by_month = {mo: [] for mo in months}
    for ed, xd, ret, hold in trades:
        if hold <= 0:
            continue
        months_held = max(1, hold / 21.0)
        g = (1 + ret) ** (1.0 / months_held) - 1
        for mo in months:
            if ed < mo <= xd:
                open_by_month[mo].append(g)
    spy_fwd = spy_m.pct_change().shift(-1)
    rets, spy_rets, idx = [], [], []
    for i in range(len(months) - 1):
        mo = months[i]
        sp = spy_fwd.iloc[i]
        if pd.isna(sp):
            continue
        opens = open_by_month.get(mo, [])
        pr = float(np.mean(opens)) if opens else 0.0
        rets.append(pr); spy_rets.append(float(sp)); idx.append(months[i + 1])
    return {"label": f"Equal-weight portfolio of top {len(top)} signals (sector universe)",
            "signals": [SIGNALS[sk][0] for sk, _ in top],
            "summary": _stats(rets, spy_rets), "curve": _curve(rets, spy_rets, idx)}


def _save_db(kind, payload):
    """Persist the payload to Postgres (BacktestResult) so results live in the DB like every other
    study, not only in the .data JSON. JSON write stays as a cache/fallback."""
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind=kind, defaults={"payload": payload, "computed_at": timezone.now()})
        print(f"Saved BacktestResult[{kind}] to DB", flush=True)
    except Exception as e:
        print(f"DB save failed for {kind}: {e}", flush=True)


def main():
    import sys as _sys
    phase1_only = "--phase1" in _sys.argv
    etfs = [s.etf for s in Sector.objects.all()]
    dfs = _load(etfs + [BENCH])
    spy_df = dfs.pop(BENCH, None)
    if spy_df is None:
        raise SystemExit("No SPY candles")
    lab = rotation_lab(dfs, spy_df)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "universe": len(dfs),
        "phase1": {"strategies": lab},   # the sector-rotation lab (ranked)
    }
    if not phase1_only:                  # heavy signal-loop phases (skip while iterating on rules)
        top = _top_signals(5)
        print("Top signals:", top, flush=True)
        payload["phase2"] = phase2(dfs, top)
        payload["phase3"] = phase3(dfs, spy_df, top)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    _save_db("rotation_lab", payload)
    print("Rotation lab (ranked by total return):", flush=True)
    for s in lab:
        print(f"  {s['label']:44} {s['summary']['total_return']:>8.1f}%  vs SPY "
              f"{s['summary']['spy_total']:>6.1f}%  Sharpe {s['summary']['sharpe']:>5.2f}  "
              f"t={s['summary']['t_stat']}", flush=True)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
