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


def _windows_A_plus():
    """★ FLAGSHIP A (short-horizon momentum finder). GAP-UP CONTINUATION on a HIGH-VOLATILITY LIQUID universe —
    the validated short-term edge (vol is the driver; gap-ups continue). Trigger = a >=5% 1-day GAP-UP on a name
    whose 20d trailing vol is in [50%,300%] annualized and >=$5M/day dollar-volume; candidate for the next 8
    trading days. No value/quality frame (that HURTS the bounce — bake-off). Sweep winner: gap-up>=5% beat the
    2-day burst. PORTFOLIO (a_beat.py, the return spec): run WIDE — K>=40 concurrent, <=2% each (concentration
    = RUIN); ENTER on the gap-up, HOLD 8 bars — hold-8 beat hold-5 ~50x (+1959% vs +38% net-20bps over ~5y).
    Do NOT magnitude-rank or uptrend-gate (both HURT). Survives fees, unlike the mean-reversion dip-buy."""
    from seq_fundamental_study import load_candles
    GAP, VOL_LO, VOL_HI, DVOL_FLOOR, WIN = 0.05, 0.50, 3.00, 5e6, 8    # WIN=8 covers the 8-bar hold
    uni = _stock_universe()
    daily = load_candles(uni)
    allowed, nwin = {}, 0
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        c = df["Close"]; idx = df.index
        vol = c.pct_change().rolling(20).std() * (252 ** 0.5)        # annualized trailing vol
        dvol = (c * df["Volume"]).rolling(20).mean()                 # trailing $ volume (liquidity)
        gap = c / c.shift(1) - 1.0                                   # 1-day gap-up (sweep winner vs 2d burst)
        s = set()
        for i in range(len(idx)):
            if (gap.iloc[i] >= GAP and VOL_LO <= vol.iloc[i] <= VOL_HI and dvol.iloc[i] >= DVOL_FLOOR):
                for j in range(i, min(i + WIN, len(idx))):
                    s.add(idx[j].date())
                nwin += 1
        if s:
            allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed)}


B_PLUS_WIN = 15                 # flagship-B watch window (sweep: 15 bars best — capitulation gap-up needs time to appear)


# B / B-2 / B-3 = three DISTINCT profitable entries inside the capitulation (seq w15) windows below
# (b2_overlap.py, verified 0-2% overlap, all significant; run together = ~900 trades vs 172 for B alone):
#   B   = gap-up     @3b  +1.54%/63%/t4.5  (strongest single). IV-skew fear-filter is a CONVICTION DIAL on B
#         (a SUBSET, NOT a separate book) -> +2.18%/84% on fear-priced entries.
#   B-2 = st_ad_div  @4b  +0.57%/57%/t2.5  (A/D-divergence continuation; orthogonal to B, 0% overlap).
#   B-3 = mr_ndown   @6b  +0.65%/51%/t2.3  (n-down reversal; orthogonal, 2% overlap).
def _windows_B_plus(limit=None):
    """★ FLAGSHIP B (short-horizon capitulation alert). B's capitulation trigger (seq_rsi20_ad_rising_rsi) +
    the ONE proven keeper — a $5M/day LIQUIDITY floor so the bounce is tradeable. NO fundamental/quality gate
    (value/profit HURT the mean-reversion bounce — bake-off). 15-trading-day WATCH WINDOW after each fire (sweep
    winner) — then the H4 gap-up entry, exit ~3 bars = +1.96%/68%win/t5.31. RARE (~4/month) across liquid
    large-caps (UAL/NFLX/MAR/ABBV…) so it's a high-quality ALERT/supplement, not a full standalone book."""
    from seq_fundamental_study import load_candles
    from studies import SIGNALS as STUDY_SIGNALS
    name, fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
    uni = _stock_universe()
    if limit:
        uni = uni[:limit]
    daily = load_candles(uni)
    DVOL_FLOOR = 5e6
    allowed, nwin, dropped = {}, 0, 0
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        sig = fn(df).fillna(False)
        idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).mean()       # trailing $ volume (liquidity)
        s = set()
        for i, v in enumerate(sig.values):
            if not v:
                continue
            if dvol.iloc[i] < DVOL_FLOOR:                            # liquidity floor (the proven keeper)
                dropped += 1; continue
            for j in range(i, min(i + B_PLUS_WIN, len(idx))):
                s.add(idx[j].date())
            nwin += 1
        if s:
            allowed[tk] = s
    return allowed, {"n_windows": nwin, "n_names": len(allowed), "dropped_fires": dropped}


def candidate_windows(selector, b_limit=None):
    """selector in {A,A_plus,B,B_plus,C,union} -> ({ticker: set[date]}, meta). b_limit caps B's universe."""
    if selector == "C":
        return _windows_C()
    if selector == "A":
        return _windows_A()
    if selector == "A_plus":
        return _windows_A_plus()
    if selector == "B":
        return _windows_B(limit=b_limit)
    if selector == "B_plus":
        return _windows_B_plus(limit=b_limit)
    if selector == "union":
        merged, nwin = {}, 0
        for sel in ("A", "B", "C"):
            a, m = candidate_windows(sel, b_limit=b_limit)
            nwin += m["n_windows"]
            for tk, s in a.items():
                merged.setdefault(tk, set()).update(s)
        return merged, {"n_windows": nwin, "n_names": len(merged)}
    raise ValueError(selector)


def _merge(dst, src):
    for k, v in src.items():
        dst.setdefault(k, []).extend(v)


def _base_rate(frames, horizon_bars=3):
    pool = []
    for df in frames.values():
        c = df["Close"].values
        if len(c) > horizon_bars:
            r = (c[horizon_bars:] - c[:-horizon_bars]) / c[:-horizon_bars] * 100
            pool.extend([x for x in r if np.isfinite(x)])
    return round(float(np.mean(pool)), 3) if pool else None


def run_selector(selector, years, allow_fetch, b_limit=None):
    from intraday_data import get_4h
    allowed, cwmeta = candidate_windows(selector, b_limit=b_limit)
    names = sorted(allowed)
    flat = {s: {} for s in H.SIGNALS}
    bucket = {s: {b[0]: {} for b in H.SIGNALS[s]["buckets"]} for s in H.SIGNALS}
    frames = {}
    got = dropped = 0
    for tk in names:
        df = get_4h(tk, years, allow_fetch)
        if df is None or len(df) < 120:
            dropped += 1
            continue
        got += 1
        frames[tk] = df
        res = backtest_ticker_masked(df, allowed[tk])
        for s in H.SIGNALS:
            _merge(flat[s], res[s]["flat"])
            for b, d in res[s]["by_bucket"].items():
                for k, v in d.items():
                    bucket[s][b].setdefault(k, []).extend(v)
    standalone = {}
    try:
        from core.models import BacktestResult
        h4 = BacktestResult.objects.get(kind="h4_study").payload
        standalone = {s: h4["signals"][s]["all"] for s in H.SIGNALS if s in h4.get("signals", {})}
    except Exception:
        pass
    base3 = _base_rate(frames, 3)
    sigs = {}
    for s in H.SIGNALS:
        eks = H.exit_keys_for(s)
        border = [b[0] for b in H.SIGNALS[s]["buckets"]]
        sigs[s] = {"name": H.SIGNALS[s]["name"], "family": H.SIGNALS[s]["family"],
                   "all": H.agg_rows(flat[s], eks),
                   "by_bucket": {b: H.agg_rows(bucket[s][b], eks) for b in border},
                   "bucket_order": border,
                   "standalone_h4": standalone.get(s, [])}
    return {"selector": selector, "families": H.FAMILIES, "signals": sigs,
            "candidates": {**cwmeta, "n_with_4h": got, "n_dropped_no_4h": dropped},
            "base_rate_3b": base3}


def run(selectors, years, allow_fetch, b_limit=None):
    return {sel: run_selector(sel, years, allow_fetch, b_limit=b_limit) for sel in selectors}


def main():
    import os, json, argparse
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="all", help="A|A_plus|B|B_plus|C|union|all")
    ap.add_argument("--no-fetch", action="store_true", help="fast pass: cached 4h only")
    ap.add_argument("--b-limit", type=int, default=None, help="cap B universe (fast verify)")
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()
    sels = ["C", "A", "A_plus", "B", "B_plus", "union"] if args.selector == "all" else [args.selector]
    print(f"selectors={sels} fetch={'off' if args.no_fetch else 'on'}", flush=True)
    payload = {"by_selector": {}}
    for sel in sels:
        print(f"[{sel}] building candidate windows + masked backtest...", flush=True)
        payload["by_selector"][sel] = run_selector(sel, args.years, allow_fetch=not args.no_fetch, b_limit=args.b_limit)
        c = payload["by_selector"][sel]["candidates"]
        print(f"[{sel}] done: {c}", flush=True)
    payload["computed_at"] = pd.Timestamp.utcnow().isoformat()
    payload["note"] = ("Daily A/B/C signal selects the name; the H4 engine times a 0-3 day entry inside the "
                       "candidate window. Bucketed by magnitude; benchmarked vs standalone H4 (broad universe) "
                       "and the candidate base rate. Gross of fees; EODHD 1h depth varies (dropped names counted); "
                       "current-membership candidate universe = survivorship bias; small/mid-cap H4 liquidity thinner.")
    STUD = Path(__file__).resolve().parent / ".data" / "studies"
    STUD.mkdir(parents=True, exist_ok=True)
    (STUD / "h4_on_signals.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="h4_on_signals",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_on_signals]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    for sel, p in payload["by_selector"].items():
        print(f"\n=== {sel}  ({p['candidates']}) base3b={p['base_rate_3b']} ===", flush=True)
        rows = []
        for s, d in p["signals"].items():
            best3 = next((r for r in d["all"] if r["exit"] == "3b"), None)
            if best3:
                sa = next((r for r in d["standalone_h4"] if r["exit"] == "3b"), None)
                rows.append((s, best3, sa))
        for s, b, sa in sorted(rows, key=lambda x: -x[1]["avg_pct"]):
            saj = f"(standalone {sa['avg_pct']:+.2f}%)" if sa else ""
            print(f"  {s:16} 3b avg {b['avg_pct']:+.2f}% win {b['win_pct']}% t={b['t']} n={b['trades']} {saj}", flush=True)


if __name__ == "__main__":
    main()
