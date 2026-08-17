#!/usr/bin/env python3
"""H4-on-daily-signals: gate the H4 engine by point-in-time candidate windows from daily systems
A (pure dip), B (capitulation), C (div_2x flagship). See
docs/superpowers/specs/2026-08-16-h4-on-daily-signals-design.md. Reuses h4_study + intraday_data;
modifies nothing. Pure functions import without Django; main() calls django.setup()."""
import numpy as np
import pandas as pd
import h4_study as H

B_WINDOW_DAYS = 10
LOOKBACK, TOP_N = 6, 10


def backtest_ticker_masked(df, allowed_dates=None):
    """Like h4_study.backtest_ticker but only counts entries whose bar date is in `allowed_dates`
    (a set of datetime.date). allowed_dates=None allows every bar (== h4_study.backtest_ticker)."""
    close = df["Close"].values
    n = len(close)
    dates = df.index.normalize()
    out = {}
    for sig, meta in H.SIGNALS.items():
        entry, mag = meta["fn"](df)
        buckets = meta["buckets"]
        cand = [i for i in range(n) if entry[i] and
                (allowed_dates is None or dates[i].date() in allowed_dates)]
        idxs = sorted(H._episode_starts(cand, gap=H.GAP))
        flat = H._empty_exit_pool(sig)
        by_bucket = {b[0]: H._empty_exit_pool(sig) for b in buckets}
        dn_exit = H._rsi_x_dn_exit(df) if meta.get("exit_fn") == "rsi_x_dn" else None
        for i in idxs:
            ep = float(close[i])
            if ep <= 0:
                continue
            blab = H.bucket_of(mag[i], buckets)
            for k, bars, _ in H.EXITS:
                j = i + bars
                if j < n:
                    r = (close[j] - ep) / ep * 100
                    flat[k].append(r)
                    if blab is not None:
                        by_bucket[blab][k].append(r)
            if dn_exit is not None:
                j = next((q for q in range(i + 1, n) if dn_exit[q]), None)
                if j is not None:
                    r = (close[j] - ep) / ep * 100
                    flat["rsi_x_dn"].append(r)
                    if blab is not None:
                        by_bucket[blab]["rsi_x_dn"].append(r)
        out[sig] = {"flat": flat, "by_bucket": by_bucket}
    return out


def _month_dates(daily_index, start, end):
    """Trading dates in [start, end) from a daily DatetimeIndex."""
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    return {d.date() for d in daily_index if s <= d < e}


def _windows_C():
    """C = div_2x flagship monthly picks from saved rotation_history. Each pick is a candidate for
    its holding month [date_i, date_{i+1})."""
    from core.models import BacktestResult
    from seq_fundamental_study import load_candles
    p = BacktestResult.objects.get(kind="rotation_history").payload
    months = p["months"]
    picks_by_name = {}
    for i, m in enumerate(months):
        start = m["date"]
        end = months[i + 1]["date"] if i + 1 < len(months) else None
        for pk in m["picks"]:
            picks_by_name.setdefault(pk["pick"], []).append((start, end))
    daily = load_candles(sorted(picks_by_name))
    allowed = {}
    nwin = 0
    for tk, spans in picks_by_name.items():
        df = daily.get(tk)
        if df is None:
            continue
        s = set()
        for start, end in spans:
            end = end or (pd.Timestamp(start) + pd.Timedelta(days=31))
            s |= _month_dates(df.index, start, end)
            nwin += 1
        if s:
            allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def _windows_A():
    """A = pure-dip (rsi10<45) monthly value pick in top-momentum sectors. Candidate for its holding
    month. Reconstructs arm3_lowpb selection (equal to the entry_signal study's selection)."""
    import ta
    import config, price_basis, sector_holdings
    from seq_fundamental_study import load_candles, load_financial_reports
    from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
    from backtest_lowpb import _monthly_close, BENCH
    etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
    sector_map, all_holds = {}, set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
        sector_map[etf] = h; all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_daily = load_candles(sorted(set(etfs.values()) | {BENCH}))
    etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etfs.values()})
    midx = etf_monthly.index
    stock_daily = load_candles(all_holds)
    stock_monthly = _monthly_close(stock_daily).reindex(midx)
    reports = load_financial_reports(all_holds)
    shares_p = _pit_monthly_panel(reports, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reports, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    pb = (price_basis.as_traded_close(stock_monthly[common]) * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)
    etf_trail = etf_monthly.pct_change(LOOKBACK)
    dip = {}
    for tk, df in stock_daily.items():
        if len(df) < 210:
            continue
        dip[tk] = (ta.momentum.rsi(df["Close"], window=10) < 45).reindex(midx, method="ffill")
    dip = pd.DataFrame(dip).reindex(midx)
    allowed, nwin = {}, 0
    for i in range(LOOKBACK, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if date not in etf_trail.index:
            continue
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N)
        for etf in ranks.index:
            holds = sector_map.get(etf, [])
            cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
            if not cands or date not in pb.index:
                continue
            row = pb.loc[date, [c for c in cands if c in pb.columns]].dropna()
            row = row[row > 0]
            if not len(row):
                continue
            pick = row.idxmin()
            if pick in dip.columns and bool(dip.loc[date, pick]):
                df = stock_daily.get(pick)
                if df is None:
                    continue
                allowed.setdefault(pick, set())
                allowed[pick] |= _month_dates(df.index, date, ndate)
                nwin += 1
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def _stock_universe():
    """Stocks = Fundamental tickers minus sector ETFs. Small-table queries only — avoids the slow
    Candle-hypertable DISTINCT in seq_fundamental_study.build_universe (the /dev/shm hazard).
    load_candles() downstream naturally keeps only names that actually have candles."""
    from core.models import Fundamental, Sector
    funda = set(Fundamental.objects.values_list("ticker", flat=True))
    etfs = set(Sector.objects.values_list("etf", flat=True))
    return sorted(t for t in funda if t not in etfs)


def _windows_B(limit=None):
    """B = capitulation seq_rsi20_ad_rising_rsi fires -> candidate for the next B_WINDOW_DAYS trading days.
    `limit` caps the universe (first N tickers) for fast verification; None = full universe (the study run)."""
    from seq_fundamental_study import load_candles
    from studies import SIGNALS as STUDY_SIGNALS   # seq signal lives in the daily studies engine
    name, fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
    uni = _stock_universe()
    if limit:
        uni = uni[:limit]
    daily = load_candles(uni)
    allowed, nwin = {}, 0
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        sig = fn(df).fillna(False)
        idx = df.index
        fires = [i for i, v in enumerate(sig.values) if v]
        if not fires:
            continue
        s = set()
        for i in fires:
            for j in range(i, min(i + B_WINDOW_DAYS, len(idx))):
                s.add(idx[j].date())
            nwin += 1
        allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


def candidate_windows(selector, b_limit=None):
    """selector in {A,B,C,union} -> ({ticker: set[date]}, meta). b_limit caps B's universe (fast verify)."""
    if selector == "C":
        return _windows_C()
    if selector == "A":
        return _windows_A()
    if selector == "B":
        return _windows_B(limit=b_limit)
    if selector == "union":
        merged, nwin = {}, 0
        for sel in ("A", "B", "C"):
            a, m = candidate_windows(sel, b_limit=b_limit)
            nwin += m["n_windows"]
            for tk, s in a.items():
                merged.setdefault(tk, set()).update(s)
        return merged, {"n_windows": nwin, "n_names": len(merged)}
    raise ValueError(selector)
