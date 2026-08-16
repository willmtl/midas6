#!/usr/bin/env python3
"""Mixed engine: SECTOR MOMENTUM ROTATION x STOCK-PICKING.

Combines Trend Studies (rank the 93 sector ETFs by trailing return, hold the top N,
rebalance every hold_months) with Stock Drilldown (hold a STOCK from each winning sector
instead of the ETF). Three hold modes:
  etf      - hold the sector ETF (baseline == original Trend Studies)
  momentum - hold the highest trailing-return stock in the sector (point-in-time)
  hibeta   - hold the highest-beta stock in the sector (Fundamental.beta_5y snapshot)

Answers: does swapping the ETF for a stock inside the winning sectors add or destroy alpha?

Run: docker compose run --rm backend python -u trend_stock_studies.py --db
⚠️ hibeta uses a CURRENT beta snapshot (mild selection lookahead); momentum is fully PIT.
⚠️ Survivorship bias: holdings = today's constituents.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import pandas as pd

import config
import sector_holdings

CRYPTO = {"IBIT", "ETHA", "BTC-USD", "ETH-USD", "BLOK"}
RULES = ["etf", "momentum", "hibeta", "insider", "lowpb"]
# rule -> (panel key, direction). 'momentum' is special (depends on lookback, computed
# per combo). 'etf' needs no panel. min = pick smallest (value); max = pick largest.
_RULE_PANEL = {"hibeta": ("beta", "max"), "insider": ("insider", "max"), "lowpb": ("pb", "min")}


def _monthly(daily_df):
    return daily_df["Close"].resample("ME").last()


def _load_monthly(tickers):
    from seq_fundamental_study import load_candles
    daily = load_candles(tickers)
    cols = {t: _monthly(df) for t, df in daily.items() if len(df) > 60}
    return pd.DataFrame(cols).sort_index()


def _load_daily_close(tickers):
    from seq_fundamental_study import load_candles
    daily = load_candles(tickers)
    return pd.DataFrame({t: df["Close"] for t, df in daily.items() if len(df) > 260}).sort_index()


def _point_in_time_beta(daily_close, holdings, spy_daily, window=252):
    """Monthly panel of trailing-`window`-day beta vs SPY, point-in-time (only past data)."""
    cols = [c for c in holdings if c in daily_close.columns]
    if not cols:
        return pd.DataFrame()
    # Snap to SPY's trading calendar (the union index has junk dates from other stocks'
    # bad data → scattered NaN that would nuke every rolling window). ffill small gaps.
    cal = spy_daily.dropna().index
    px = daily_close[cols].reindex(cal).ffill(limit=5)
    spy_ret = spy_daily.reindex(cal).ffill().pct_change()
    hold_ret = px.pct_change()
    # Explicit rolling cov/var (DataFrame.rolling().cov(Series) returns all-NaN here).
    # cov(x,spy)=E[x*spy]-E[x]E[spy]; beta=cov/var(spy). var(spy) is common across stocks
    # at each date, so the beta *ranking* used for selection is exact regardless of ddof.
    mp = window // 2
    m_spy = spy_ret.rolling(window, min_periods=mp).mean()
    v_spy = spy_ret.rolling(window, min_periods=mp).var()
    prod_mean = hold_ret.mul(spy_ret, axis=0).rolling(window, min_periods=mp).mean()
    cov = prod_mean.sub(hold_ret.rolling(window, min_periods=mp).mean().mul(m_spy, axis=0))
    beta_daily = cov.div(v_spy, axis=0)
    return beta_daily.resample("ME").last()


def _pit_monthly_panel(reports_map, field, monthly_index):
    """Month x ticker panel of a FinancialReport field, forward-filled by avail_date (PIT)."""
    out = {}
    for tk, r in reports_map.items():
        if field not in r.columns:
            continue
        s = pd.Series(r[field].values, index=pd.to_datetime(r["avail_date"])).dropna()
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[tk] = s.reindex(s.index.union(monthly_index)).ffill().reindex(monthly_index)
    return pd.DataFrame(out)


def _insider_monthly_panel(tickers, monthly_index):
    """Month x ticker panel of trailing-3-month insider open-market buy $ (PIT)."""
    from seq_fundamental_study import load_insider
    ins = load_insider(tickers)
    out = {}
    for tk, s in ins.items():
        m = s.resample("ME").sum().rolling(3, min_periods=1).sum()
        out[tk] = m.reindex(m.index.union(monthly_index)).ffill().reindex(monthly_index)
    return pd.DataFrame(out)


def run_mixed(etf_monthly, stock_monthly, panels, sector_map, lookback, hold, top_n, rule):
    trail = etf_monthly.pct_change(lookback)
    stock_trail = stock_monthly.pct_change(lookback)
    if len(etf_monthly) < lookback + hold + 1:
        return None
    reb = list(trail.index[lookback::hold])
    equity = spy_equity = 1.0
    ec, sc, trades = [], [], []

    for i, date in enumerate(reb):
        ranks = trail.loc[date].dropna().drop("SPY", errors="ignore").sort_values(ascending=False)
        if len(ranks) < top_n:
            continue
        top = ranks.head(top_n).index.tolist()
        end = reb[i + 1] if i + 1 < len(reb) else etf_monthly.index[-1]

        picks, port_ret = [], 0.0
        for etf in top:
            name, holds = sector_map.get(etf, (etf, []))
            inst, r = _pick_return(etf, holds, date, end, rule, etf_monthly, stock_monthly, stock_trail, panels)
            picks.append(inst)
            port_ret += r / top_n
        equity *= (1 + port_ret)
        ec.append({"date": str(date)[:10], "equity": round(equity, 4), "return": round(port_ret * 100, 2)})

        if "SPY" in etf_monthly.columns:
            sp = etf_monthly["SPY"]
            if date in sp.index and end in sp.index and sp.loc[date] > 0:
                spy_equity *= (1 + sp.loc[end] / sp.loc[date] - 1)
        sc.append({"date": str(date)[:10], "spy_equity": round(spy_equity, 4)})
        trades.append({"date": str(date)[:10], "end_date": str(end)[:10],
                       "sectors": top, "picks": picks, "return_pct": round(port_ret * 100, 2)})

    if not ec:
        return None
    total = (equity - 1) * 100
    n_years = len(ec) * hold / 12
    annual = ((equity ** (1 / max(n_years, 0.1))) - 1) * 100 if n_years > 0 else 0
    spy_total = (spy_equity - 1) * 100
    peak = 1.0; max_dd = 0.0
    for pt in ec:
        peak = max(peak, pt["equity"])
        max_dd = min(max_dd, (pt["equity"] - peak) / peak * 100)
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    # Significance of the average rebalance-period return vs 0. Periods are NON-overlapping
    # (each starts where the last ended), so this is a legit one-sample t — no overlap correction
    # needed. It's what separates a real edge from a lucky single-name run (esp. top_n=1, where a
    # "+525%" can be ~8 sequential single-stock bets).
    period_rets = [t["return_pct"] for t in trades]
    t_stat = None
    if len(period_rets) >= 3:
        ser = pd.Series(period_rets, dtype=float)
        sd = ser.std(ddof=1)
        if sd and sd > 0:
            t_stat = round(float(ser.mean() / (sd / len(ser) ** 0.5)), 2)
    return {"lookback_months": lookback, "hold_months": hold, "top_n": top_n, "hold_mode": rule,
            "total_return": round(total, 2), "annual_return": round(annual, 2),
            "spy_total": round(spy_total, 2), "alpha": round(total - spy_total, 2),
            "max_drawdown": round(max_dd, 2), "trades": len(trades), "t_stat": t_stat,
            "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
            "equity_curve": ec, "spy_curve": sc, "trade_log": trades}


def _ret(series, date, end):
    if date in series.index and end in series.index:
        s, e = series.loc[date], series.loc[end]
        if pd.notna(s) and pd.notna(e) and s > 0:
            return e / s - 1
    return None


def _available_at(series, date):
    """Is this name tradeable AT `date` — the only information available at selection time."""
    if date not in series.index:
        return False
    s = series.loc[date]
    return bool(pd.notna(s) and s > 0)


def _ret_delist(series, date, end):
    """Hold-period return date->end. If the name has no valid price at `end` (delisted/halted mid-
    hold), realize at the LAST traded price on or before `end` — a delisting is a real exit, not a
    reason to drop the trade. Requiring a price AT `end` (old `_ret`) was survivorship LOOKAHEAD: it
    filtered selection on which names survived the holding period."""
    if not _available_at(series, date):
        return None
    s = series.loc[date]
    e = series.loc[end] if end in series.index else None
    if pd.notna(e) and e and e > 0:
        return e / s - 1
    win = series.loc[date:end].dropna()
    win = win[win > 0]
    if len(win) == 0:            # no valid price at all in the window (entry non-positive) -> can't realize
        return None
    # Delisting = exit at the LAST TRADED price (deal price for an M&A/going-private, last print for a fade).
    # The old blanket -100% here was a BUG: it treated EVERY acquisition/rename/data-gap as a total loss
    # (e.g. STMP/Stamps.com acquired @ $330 was booked at -100%), biasing delist-aware backtests badly
    # pessimistic. When only the entry price survives (data ends at entry, len==1), this returns 0% = a flat
    # exit at last-known price. LIMITATION: genuine gap-to-zero bankruptcies with no intermediate monthly
    # print are now under-penalized (booked flat, not -100%); without a delisting-reason/daily feed this is
    # the least-biased default. See memory delisted-survivorship / verify_survivorship.py.
    return win.iloc[-1] / s - 1


def _pick_return(etf, holds, date, end, rule, etf_monthly, stock_monthly, stock_trail, panels):
    """Return (instrument, hold-period return). Falls back to the ETF if no stock qualifies.
    All stock-pick rules select POINT-IN-TIME using only data available at `date`:
    momentum=highest trailing return, hibeta=highest 252d beta, insider=most trailing-3mo
    insider buying, lowpb=cheapest P/B."""
    etf_r = _ret(etf_monthly[etf], date, end) if etf in etf_monthly.columns else None
    if rule == "etf" or not holds:
        return etf, (etf_r or 0.0)
    # Qualify candidates on availability AT `date` only (no peeking at the future `end` price).
    cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
    if not cands:
        return etf, (etf_r or 0.0)

    if rule == "momentum":
        panel, direction = stock_trail, "max"
    else:
        pk, direction = _RULE_PANEL.get(rule, (None, None))
        panel = panels.get(pk) if pk else None
    if panel is not None and date in panel.index:
        pcands = [c for c in cands if c in panel.columns]
        row = panel.loc[date, pcands].dropna()
        if direction == "min":
            row = row[row > 0]          # cheapest positive P/B (ignore negative-equity)
        else:
            row = row[row > 0] if rule == "insider" else row  # insider: require actual buying
        if len(row):
            pick = row.idxmax() if direction == "max" else row.idxmin()
            r = _ret_delist(stock_monthly[pick], date, end)
            return pick, (r if r is not None else (etf_r or 0.0))
    return etf, (etf_r or 0.0)


def run(save_db=False):
    etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
    etf_tickers = list(etfs.values()) + ["SPY"]
    print(f"Loading {len(etf_tickers)} ETFs + holdings from DB...")
    etf_monthly = _load_monthly(etf_tickers)

    sector_map = {}                      # etf -> (name, [holdings])
    all_holds = set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, "SPY")]
        sector_map[etf] = (name, h)
        all_holds.update(h)
    daily_close = _load_daily_close(sorted(all_holds) + ["SPY"])
    stock_monthly = daily_close[[c for c in daily_close.columns if c in all_holds]].resample("ME").last()
    spy_daily = daily_close["SPY"] if "SPY" in daily_close.columns else _load_daily_close(["SPY"]).get("SPY")
    midx = etf_monthly.index

    # Point-in-time selection panels (month x ticker).
    from seq_fundamental_study import load_financial_reports
    reps = load_financial_reports(sorted(all_holds))
    shares_p = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reps, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    # P/B on AS-TRADED price (undo future-split back-adjustment look-ahead, finding #2); returns keep adj close.
    import price_basis
    px_at = price_basis.as_traded_close(stock_monthly[common], price_basis.refresh_splits(list(common)))
    pb_panel = (px_at * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)
    panels = {
        "beta": _point_in_time_beta(daily_close, sorted(all_holds), spy_daily),
        "insider": _insider_monthly_panel(sorted(all_holds), midx),
        "pb": pb_panel,
    }
    print(f"ETF months {len(etf_monthly)} | holdings {len(stock_monthly.columns)} | "
          f"beta {panels['beta'].shape} insider {panels['insider'].shape} pb {pb_panel.shape}")

    # FULL grid (matches original Trend Studies) x all hold modes.
    grid_lb, grid_h, grid_n = [1, 2, 3, 4, 5, 6, 9, 12], [1, 2, 3, 4, 6], [1, 2, 3, 5, 7, 10, 15, 20]
    results = []
    for rule in RULES:
        for lb in grid_lb:
            for h in grid_h:
                for n in grid_n:
                    r = run_mixed(etf_monthly, stock_monthly, panels, sector_map, lb, h, n, rule)
                    if r:
                        results.append(r)
    # Adequately-sampled AND statistically-supported combos are "robust". Below MIN_PERIODS
    # non-overlapping rebalances a headline "+525%" is usually one lucky (often single-name) run.
    MIN_PERIODS = 12
    for r in results:
        r["robust"] = bool(r["trades"] >= MIN_PERIODS and r["t_stat"] is not None and abs(r["t_stat"]) >= 2)
    # Rank robust first, then by total_return — a thin single-name fluke can no longer sit at the top.
    results.sort(key=lambda x: (x["robust"], x["total_return"]), reverse=True)

    print(f"\n{'MODE':9} {'LB':>3} {'H':>2} {'TopN':>4} {'Total':>8} {'Annual':>7} {'Alpha':>7} {'MaxDD':>7} {'WR':>5} {'t':>5} {'rob':>4}")
    for r in results[:20]:
        print(f"{r['hold_mode']:9} {r['lookback_months']:>3} {r['hold_months']:>2} {r['top_n']:>4} "
              f"{r['total_return']:>7.1f}% {r['annual_return']:>6.1f}% {r['alpha']:>6.1f}% {r['max_drawdown']:>6.1f}% "
              f"{r['win_rate']:>4.0f}% {(r['t_stat'] if r['t_stat'] is not None else 0):>5.1f} {('Y' if r['robust'] else '-'):>4}")

    # Best per mode for a clean head-to-head.
    print("\nBest per mode:")
    for rule in RULES:
        rr = [r for r in results if r["hold_mode"] == rule]
        if rr:
            b = rr[0]
            print(f"  {rule:9} lb{b['lookback_months']}/h{b['hold_months']}/top{b['top_n']}: "
                  f"total {b['total_return']:+.1f}% alpha {b['alpha']:+.1f}% DD {b['max_drawdown']:.1f}% WR {b['win_rate']:.0f}%")

    if save_db:
        from core.models import TrendStudy
        from django.utils import timezone
        now = timezone.now()
        for r in results:
            TrendStudy.objects.update_or_create(
                lookback_months=r["lookback_months"], hold_months=r["hold_months"],
                top_n=r["top_n"], hold_mode=r["hold_mode"],
                defaults={k: r[k] for k in ("total_return", "annual_return", "spy_total", "alpha",
                          "max_drawdown", "win_rate")} | {"num_trades": r["trades"], "t_stat": r["t_stat"],
                          "equity_curve": r["equity_curve"], "spy_curve": r["spy_curve"],
                          "trade_log": r["trade_log"], "computed_at": now})
        print(f"\nSaved {len(results)} rows to TrendStudy (with hold_mode).")
    return results


if __name__ == "__main__":
    if "--db" in sys.argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(save_db="--db" in sys.argv)
