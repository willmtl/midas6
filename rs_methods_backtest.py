#!/usr/bin/env python3
"""RS-TREND METHOD SWEEP — backtest ~20 ways to read the ETF/SPY "relative-strength bar", each feeding
the SAME winning pick (cheapest positive-P/B large-cap holding in the selected sectors), so the ONLY
thing that varies is the sector-selection rule.

Context: pure ETF rotation loses to SPY; the alpha is rotation-as-a-FILTER -> value stock pick
(arm3_lowpb, +154% vs SPY t2.09). We already know a handful of trend reads all cluster ~+210-250% vs
SPY. This sweeps MANY reads at once so they can be compared side-by-side on the dashboard.

HONESTY: 20+ methods on ~62 monthly periods is a multiple-comparisons hazard. Read the TABLE, not the
winner — rank by t-stat / Sharpe / drawdown, and treat any single eye-popping vs-SPY as sample luck if
its neighbours (same family, nearby params) don't agree. The payload keeps t/Sharpe/DD/coverage so the
UI can show that. Directional, no fees; stock-universe survivorship applies.

-> BacktestResult[rs_methods] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/rs_methods_backtest.py
"""
import os, sys, json, warnings
sys.argv = ["rs_methods"]
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd, ta
import config, sector_holdings, indicators
from backtest_lowpb import _arm, _monthly_close, BENCH, CRYPTO
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at

LARGECAP, TOP_N, WARMUP = 10e9, 10, 12


def build():
    # ---- data prep (mirror backtest_lowpb / rs_trend_methods) ----------------
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
        sector_map[etf] = (name, h); all_holds.update(h)
    all_holds = sorted(all_holds); etf_tickers = list(etfs.values())
    etf_daily = load_candles(etf_tickers + [BENCH])
    etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tickers})
    midx = etf_monthly.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    stock_monthly = _monthly_close(load_candles(all_holds)).reindex(midx)
    stock_fwd = stock_monthly.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)
    reps = load_financial_reports(all_holds)
    shares_p = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reps, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    pb_panel = (stock_monthly[common] * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)
    mktcap_panel = stock_monthly[common] * shares_p[common]

    # ---- the RS "dynamic bar" (etf/spy) + every trend read on it -------------
    rs = etf_monthly.div(spy_m, axis=0)
    etf_mom3, etf_mom6, etf_mom12 = etf_monthly.pct_change(3), etf_monthly.pct_change(6), etf_monthly.pct_change(12)
    rs_mom3, rs_mom6, rs_mom12 = rs.pct_change(3), rs.pct_change(6), rs.pct_change(12)
    rs_ma6, rs_ma10, rs_ma12 = rs.rolling(6).mean(), rs.rolling(10).mean(), rs.rolling(12).mean()
    rs_ma3 = rs.rolling(3).mean()
    rs_sma2, rs_sma5 = rs.rolling(2).mean(), rs.rolling(5).mean()
    rs_ema3 = rs.ewm(span=3, adjust=False).mean()
    rs_ema5 = rs.ewm(span=5, adjust=False).mean()
    rs_ema10 = rs.ewm(span=10, adjust=False).mean()
    rs_ema12 = rs.ewm(span=12, adjust=False).mean()
    rs_std12 = rs.rolling(12).std()
    rs_z = (rs - rs_ma12) / rs_std12.where(rs_std12 > 0)
    rs_rsi10 = pd.DataFrame({c: ta.momentum.rsi(rs[c], window=10) for c in rs.columns})
    rs_rsi14 = pd.DataFrame({c: ta.momentum.rsi(rs[c], window=14) for c in rs.columns})
    rs_high6, rs_high12 = rs.rolling(6).max(), rs.rolling(12).max()
    rs_vol6 = rs.pct_change().rolling(6).std()
    rs_volmom = rs_mom6 / rs_vol6.where(rs_vol6 > 0)
    rs_accel = rs_mom6 - rs_mom6.shift(3)

    def _slope(col):
        return col.rolling(6).apply(lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean()) if x.mean() else np.nan, raw=True)
    rs_slope6 = pd.DataFrame({c: _slope(rs[c]) for c in rs.columns})

    def _slope3(col):
        return col.rolling(3).apply(lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean()) if x.mean() else np.nan, raw=True)
    rs_slope3 = pd.DataFrame({c: _slope3(rs[c]) for c in rs.columns})

    # weekly RS bar (for the fresh-turn cross)
    etf_weekly = pd.DataFrame({t: etf_daily[t]["Close"].resample("W").last() for t in etf_tickers})
    spy_weekly = etf_daily[BENCH]["Close"].resample("W").last()
    rs_w = etf_weekly.div(spy_weekly, axis=0)
    rsi_w = pd.DataFrame({c: ta.momentum.rsi(rs_w[c], window=10) for c in rs_w.columns})
    sma_w = rsi_w.rolling(10).mean()
    cross_w = (rsi_w > sma_w) & (rsi_w.shift(1) <= sma_w.shift(1)) & (rsi_w < 50)

    # ---- classic trend indicators on a WEEKLY RS OHLC bar --------------------
    # 14-period indicators need more than 62 monthly points, so compute them on the WEEKLY relative-
    # strength bar (etf/spy daily -> weekly OHLC) and read the last COMPLETED weekly value at each
    # month-end rebalance (reindex+ffill => strictly prior data, no lookahead).
    spy_close = etf_daily[BENCH]["Close"]

    def _wohlc(e):
        s = (etf_daily[e]["Close"] / spy_close.reindex(etf_daily[e].index)).dropna()
        if len(s) < 80:
            return None
        d = pd.DataFrame({"o": s.resample("W").first(), "h": s.resample("W").max(),
                          "l": s.resample("W").min(), "c": s.resample("W").last()}).dropna()
        return d if len(d) > 40 else None
    WK = {e: d for e in etf_tickers if (d := _wohlc(e)) is not None}

    def _mpanel(fn):
        """weekly boolean signal per ETF -> month-end-aligned 0/1 panel (ffill = last completed week)."""
        cols = {}
        for e, d in WK.items():
            try:
                b = fn(d).astype(float)
                cols[e] = b.reindex(b.index.union(midx)).ffill().reindex(midx)
            except Exception:
                pass
        return pd.DataFrame(cols)

    tt = ta.trend

    def _f_macd(d):     m = tt.MACD(d["c"]); return m.macd_diff() > 0
    def _f_adx(d):      a = tt.ADXIndicator(d["h"], d["l"], d["c"]); return (a.adx_pos() > a.adx_neg()) & (a.adx() > 20)
    def _f_aroon(d):
        try: ar = tt.AroonIndicator(high=d["h"], low=d["l"])
        except TypeError: ar = tt.AroonIndicator(close=d["c"])
        return ar.aroon_up() > ar.aroon_down()
    def _f_vortex(d):   v = tt.VortexIndicator(d["h"], d["l"], d["c"]); return v.vortex_indicator_pos() > v.vortex_indicator_neg()
    def _f_cci(d):      return tt.CCIIndicator(d["h"], d["l"], d["c"]).cci() > 0
    def _f_trix(d):     return tt.TRIXIndicator(d["c"]).trix() > 0
    def _f_psar(d):     return d["c"] > tt.PSARIndicator(d["h"], d["l"], d["c"]).psar()
    def _f_ichi(d):     ic = tt.IchimokuIndicator(d["h"], d["l"]); return ic.ichimoku_conversion_line() > ic.ichimoku_base_line()
    def _f_kst(d):      k = tt.KSTIndicator(d["c"]); return k.kst() > k.kst_sig()
    def _f_stc(d):      return tt.STCIndicator(d["c"]).stc() > 50

    IND_PANELS = {
        "MACD > signal (RS)":     _f_macd,
        "ADX/DMI +DI>-DI, ADX>20": _f_adx,
        "Aroon up > down (RS)":   _f_aroon,
        "Vortex VI+ > VI- (RS)":  _f_vortex,
        "CCI > 0 (RS)":           _f_cci,
        "TRIX > 0 (RS)":          _f_trix,
        "PSAR: RS above SAR":     _f_psar,
        "Ichimoku tenkan>kijun":  _f_ichi,
        "KST > signal (RS)":      _f_kst,
        "STC > 50 (RS)":          _f_stc,
    }
    panels = {name: _mpanel(fn) for name, fn in IND_PANELS.items()}

    # ---- OUR ACTUAL SIGNAL ENGINE on a synthetic RS candle -------------------
    # Feed a synthetic daily RS candle (Close = etf/spy) into the real indicators.* functions and
    # reproduce trend_analyzer.analyze()'s BULLISH / ROTATE-IN logic, evaluated at each month-end on a
    # PIT slice (df.loc[:date]). This is literally "run our engine's trend read on the synthetic bar".
    spy_df_full = etf_daily[BENCH]
    rs_daily = {}
    for e in etf_tickers:
        c = (etf_daily[e]["Close"] / spy_close.reindex(etf_daily[e].index)).dropna()
        if len(c) > 60:
            rs_daily[e] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c})

    def _engine_state(sub, spy_sub):
        m = indicators.compute_all_risk_metrics(sub, spy_sub, window=config.SORTINO_WINDOW)
        if not m:
            return None
        rx = indicators.compute_rsi_crossover(sub, omega_series=m.get("omega_series"))
        if rx.get("rsi") is None or rx.get("rsi_sma") is None:
            return None
        bullish = bool(rx.get("rsi_above_sma", False)) and (m.get("omega") or 0) > 1.0
        if not bullish:
            return "OTHER"
        return "ROTATE IN" if rx.get("rsi_crossover", False) else "BULLISH"

    engine_panel = pd.DataFrame(index=midx, columns=list(rs_daily), dtype=object)
    _dates = list(midx[WARMUP - 6:-1])   # a little pre-roll so warmup months have state too
    for e, df in rs_daily.items():
        spy_c = spy_df_full["Close"]
        for d in _dates:
            sub = df.loc[:d]
            if len(sub) < 40:
                continue
            try:
                engine_panel.loc[d, e] = _engine_state(sub, spy_df_full.loc[:d])
            except Exception:
                pass

    def _topn(series, filt=None):
        s = series.dropna()
        if filt is not None:
            s = s[[e for e in s.index if e in filt]]
        return list(s.sort_values(ascending=False).head(TOP_N).index)

    def _up(cond_row, rank_row):
        up = set(e for e in etf_tickers if pd.notna(cond_row.get(e)) and bool(cond_row[e]))
        return _topn(rank_row, up)

    # ---- selection methods: date -> [etf, ...] -------------------------------
    def m_absmom3(d):  return _topn(etf_mom3.loc[d])
    def m_absmom6(d):  return _topn(etf_mom6.loc[d])
    def m_absmom12(d): return _topn(etf_mom12.loc[d])
    def m_rsmom3(d):   return _topn(rs_mom3.loc[d])
    def m_rsmom6(d):   return _topn(rs_mom6.loc[d])
    def m_rsmom12(d):  return _topn(rs_mom12.loc[d])
    def m_ma6(d):      return _up(rs.loc[d] > rs_ma6.loc[d], rs_mom6.loc[d])
    def m_ma10(d):     return _up(rs.loc[d] > rs_ma10.loc[d], rs_mom6.loc[d])
    def m_ma12(d):     return _up(rs.loc[d] > rs_ma12.loc[d], rs_mom6.loc[d])
    def m_macross(d):
        gap = (rs_ma3.loc[d] / rs_ma10.loc[d] - 1)
        return _up(gap > 0, gap)
    def m_rsi10(d):    return _up(rs_rsi10.loc[d] > 50, rs_rsi10.loc[d])
    def m_rsi14(d):    return _up(rs_rsi14.loc[d] > 50, rs_rsi14.loc[d])
    def m_zscore(d):   return _up(rs_z.loc[d] > 0, rs_z.loc[d])
    def m_high6(d):    return _up(rs.loc[d] >= 0.99 * rs_high6.loc[d], rs_mom6.loc[d])
    def m_high12(d):   return _up(rs.loc[d] >= 0.99 * rs_high12.loc[d], rs_mom6.loc[d])
    def m_slope6(d):   return _topn(rs_slope6.loc[d])
    def m_slope3(d):   return _topn(rs_slope3.loc[d])
    def m_volmom(d):   return _topn(rs_volmom.loc[d])
    def m_accel(d):    return _up(rs_accel.loc[d] > 0, rs_accel.loc[d])
    def m_dual(d):     return _up((rs_mom6.loc[d] > 0) & (etf_mom6.loc[d] > 0), rs_mom6.loc[d])
    def m_persist(d):
        prev = rs.shift(1).loc[d]
        return _up((rs.loc[d] > prev) & (rs_mom6.loc[d] > 0), rs_mom6.loc[d])

    # MA crossovers on the monthly RS bar (fast MA above slow MA), ranked by RS 6mo momentum
    def m_sma_2_6(d):   return _up(rs_sma2.loc[d] > rs_ma6.loc[d], rs_mom6.loc[d])
    def m_sma_5_10(d):  return _up(rs_sma5.loc[d] > rs_ma10.loc[d], rs_mom6.loc[d])
    def m_sma_6_12(d):  return _up(rs_ma6.loc[d] > rs_ma12.loc[d], rs_mom6.loc[d])
    def m_ema_3_12(d):  return _up(rs_ema3.loc[d] > rs_ema12.loc[d], rs_mom6.loc[d])
    def m_ema_5_10(d):  return _up(rs_ema5.loc[d] > rs_ema10.loc[d], rs_mom6.loc[d])

    # weekly-indicator panels: select ETFs flagged "up", ranked by RS 6mo momentum
    def _sel_panel(panel):
        def f(d):
            if d not in panel.index or panel.empty:
                return []
            row = panel.loc[d]
            return _up(row > 0, rs_mom6.loc[d])
        return f

    # OUR engine's trend read on the synthetic RS candle
    def m_engine_bullish(d):
        if d not in engine_panel.index:
            return []
        row = engine_panel.loc[d]
        up = set(e for e in etf_tickers if e in row.index and row.get(e) in ("BULLISH", "ROTATE IN"))
        return _topn(rs_mom6.loc[d], up)

    def m_engine_rotatein(d):
        if d not in engine_panel.index:
            return []
        row = engine_panel.loc[d]
        up = set(e for e in etf_tickers if e in row.index and row.get(e) == "ROTATE IN")
        return _topn(rs_mom6.loc[d], up)

    def m_weekly_cross8(d):
        lo = d - pd.Timedelta(weeks=8)
        sub = cross_w.loc[(cross_w.index > lo) & (cross_w.index <= d)]
        fired = set(e for e in etf_tickers if e in sub.columns and bool(sub[e].any()))
        return _topn(rs_mom6.loc[d], fired) if fired else []

    # ---- pick layer: cheapest positive-P/B LARGE-CAP holding -----------------
    def cheapest_pb(cands, date):
        cands = [h for h in cands if h in stock_monthly.columns and _available_at(stock_monthly[h], date)
                 and h in mktcap_panel.columns and pd.notna(mktcap_panel.loc[date, h]) and mktcap_panel.loc[date, h] >= LARGECAP]
        if not cands or date not in pb_panel.index:
            return None
        row = pb_panel.loc[date, [c for c in cands if c in pb_panel.columns]].dropna(); row = row[row > 0]
        return row.idxmin() if len(row) else None

    def rot_pick(sel_fn):
        def f(i):
            date = midx[i]; picks = []
            for etf in sel_fn(date):
                _, holds = sector_map.get(etf, (etf, []))
                p = cheapest_pb(holds, date)
                if p is not None:
                    picks.append(p)
            if not picks:
                return None
            fwd = stock_fwd.loc[date, [p for p in picks if p in stock_fwd.columns]].dropna()
            return float(fwd.mean()) if len(fwd) else None
        return f

    def pick_only(i):   # value floor: cheapest-P/B large-cap over the WHOLE universe, no rotation
        date = midx[i]; avail = stock_monthly.loc[date]
        cands = [c for c in avail[(avail.notna()) & (avail > 0)].index
                 if c in mktcap_panel.columns and pd.notna(mktcap_panel.loc[date, c]) and mktcap_panel.loc[date, c] >= LARGECAP]
        if date not in pb_panel.index:
            return None
        row = pb_panel.loc[date, [c for c in cands if c in pb_panel.columns]].dropna(); row = row[row > 0]
        if not len(row):
            return None
        fwd = stock_fwd.loc[date, [p for p in row.nsmallest(TOP_N).index if p in stock_fwd.columns]].dropna()
        return float(fwd.mean()) if len(fwd) else None

    def n_sel(sel_fn):
        cnt = [len(sel_fn(midx[i])) for i in range(WARMUP, len(midx) - 1)]
        return round(float(np.mean(cnt)), 1) if cnt else 0

    # (name, family, selection_fn, description)
    METHODS = [
        ("Abs momentum 3mo",        "absolute",  m_absmom3,  "Top-N sector ETFs by trailing 3-month price return."),
        ("Abs momentum 6mo",        "absolute",  m_absmom6,  "Top-N sector ETFs by trailing 6-month price return."),
        ("Abs momentum 12mo",       "absolute",  m_absmom12, "Top-N sector ETFs by trailing 12-month price return."),
        ("RS momentum 3mo",         "rs-mom",    m_rsmom3,   "Top-N by 3-month change in the ETF/SPY relative-strength bar."),
        ("RS momentum 6mo",         "rs-mom",    m_rsmom6,   "Top-N by 6-month change in ETF/SPY (relative momentum)."),
        ("RS momentum 12mo",        "rs-mom",    m_rsmom12,  "Top-N by 12-month change in ETF/SPY."),
        ("RS > 6mo MA",             "rs-trend",  m_ma6,      "RS bar above its 6-month MA (uptrend), ranked by RS 6mo momentum."),
        ("RS > 10mo MA",            "rs-trend",  m_ma10,     "RS bar above its 10-month MA, ranked by RS 6mo momentum."),
        ("RS > 12mo MA",            "rs-trend",  m_ma12,     "RS bar above its 12-month MA, ranked by RS 6mo momentum."),
        ("RS 3mo-MA x 10mo-MA",     "rs-trend",  m_macross,  "RS fast MA (3mo) above slow MA (10mo), ranked by the gap."),
        ("RS SMA 2 x 6 cross",      "ma-cross",  m_sma_2_6,  "RS 2mo SMA above 6mo SMA (fast>slow), ranked by RS 6mo momentum."),
        ("RS SMA 5 x 10 cross",     "ma-cross",  m_sma_5_10, "RS 5mo SMA above 10mo SMA, ranked by RS 6mo momentum."),
        ("RS SMA 6 x 12 cross",     "ma-cross",  m_sma_6_12, "RS 6mo SMA above 12mo SMA, ranked by RS 6mo momentum."),
        ("RS EMA 3 x 12 cross",     "ma-cross",  m_ema_3_12, "RS 3mo EMA above 12mo EMA, ranked by RS 6mo momentum."),
        ("RS EMA 5 x 10 cross",     "ma-cross",  m_ema_5_10, "RS 5mo EMA above 10mo EMA, ranked by RS 6mo momentum."),
        ("RS RSI(10) > 50",         "rs-osc",    m_rsi10,    "RSI(10) of the RS bar above 50, ranked by RSI."),
        ("RS RSI(14) > 50",         "rs-osc",    m_rsi14,    "RSI(14) of the RS bar above 50, ranked by RSI."),
        ("RS Z-score > 0 (12mo)",   "rs-osc",    m_zscore,   "RS bar above its 12-month mean (z>0), ranked by z-score."),
        ("RS new 6mo high",         "breakout",  m_high6,    "RS bar within 1% of its 6-month high, ranked by RS 6mo momentum."),
        ("RS new 12mo high",        "breakout",  m_high12,   "RS bar within 1% of its 12-month high, ranked by RS 6mo momentum."),
        ("RS linreg slope 6mo",     "slope",     m_slope6,   "Top-N by normalized 6-month linear-regression slope of the RS bar."),
        ("RS linreg slope 3mo",     "slope",     m_slope3,   "Top-N by normalized 3-month linear-regression slope of the RS bar."),
        ("RS vol-adj momentum",     "risk-adj",  m_volmom,   "RS 6mo momentum divided by 6mo RS volatility (Sharpe-like)."),
        ("RS acceleration",         "risk-adj",  m_accel,    "RS 6mo momentum rising vs 3 months ago (2nd derivative > 0)."),
        ("Dual momentum (rel+abs)", "combo",     m_dual,     "RS 6mo momentum AND absolute 6mo momentum both positive."),
        ("RS persistence",          "combo",     m_persist,  "RS bar up month-over-month AND positive 6mo momentum."),
        ("WEEKLY RS RSI-cross<50",  "weekly",    m_weekly_cross8, "Sectors whose WEEKLY RS RSI crossed up from <50 in the last 8 weeks."),
        ("ENGINE BULLISH (RS candle)",  "engine", m_engine_bullish,  "Our live trend engine (RSI(10) cross + Omega>1) run on a synthetic etf/spy candle — BULLISH."),
        ("ENGINE ROTATE-IN (RS candle)","engine", m_engine_rotatein, "Our live trend engine on the synthetic etf/spy candle — fresh ROTATE-IN (RSI just crossed up)."),
    ]
    # weekly-OHLC trend indicators on the RS bar (skipped cleanly if the ta version can't build one)
    _ind_desc = {
        "MACD > signal (RS)": "MACD(12,26,9) histogram of the RS bar positive (fast trend up).",
        "ADX/DMI +DI>-DI, ADX>20": "Directional Movement: +DI above -DI with ADX>20 (a real trend, not chop).",
        "Aroon up > down (RS)": "Aroon up line above down line — recent RS highs newer than recent RS lows.",
        "Vortex VI+ > VI- (RS)": "Vortex positive line above negative line (uptrend).",
        "CCI > 0 (RS)": "Commodity Channel Index of the RS bar above zero.",
        "TRIX > 0 (RS)": "Triple-smoothed RS momentum positive (filters noise).",
        "PSAR: RS above SAR": "RS bar above its Parabolic SAR (trailing-stop trend flip).",
        "Ichimoku tenkan>kijun": "Ichimoku conversion line above base line (RS bar).",
        "KST > signal (RS)": "Know Sure Thing above its signal line (RS bar).",
        "STC > 50 (RS)": "Schaff Trend Cycle of the RS bar above 50.",
    }
    for name in IND_PANELS:
        METHODS.append((name, "indicator", _sel_panel(panels[name]), _ind_desc.get(name, "")))

    results, skipped = [], []
    for name, family, fn, desc in METHODS:
        try:
            s = _arm(rot_pick(fn), midx, spy_fwd, WARMUP)["summary"]
            if not s.get("periods"):
                skipped.append(name); continue
        except Exception as e:
            skipped.append(f"{name} ({e})"); continue
        results.append({
            "method": name, "family": family, "description": desc,
            "total_return": s["total_return"], "vs_spy": s["vs_spy"],
            "t_stat": s["t_stat"], "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
            "periods": s["periods"], "avg_sectors": n_sel(fn),
        })
    if skipped:
        print("skipped:", "; ".join(skipped), flush=True)
    # value floor (no rotation) as a reference row
    sfloor = _arm(pick_only, midx, spy_fwd, WARMUP)["summary"]
    results.append({
        "method": "(ref) value pick-only, no rotation", "family": "reference",
        "description": "Cheapest-P/B large-cap over the WHOLE universe — the value edge WITHOUT any sector rotation filter.",
        "total_return": sfloor["total_return"], "vs_spy": sfloor["vs_spy"], "t_stat": sfloor["t_stat"],
        "sharpe": sfloor["sharpe"], "max_drawdown": sfloor["max_drawdown"], "periods": sfloor["periods"],
        "avg_sectors": None,
    })

    results.sort(key=lambda r: -(r["vs_spy"] if r["vs_spy"] is not None else -1e9))
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n_sectors": TOP_N, "largecap_min": LARGECAP, "warmup_months": WARMUP,
                   "periods": len(midx),
                   "pick": "cheapest positive-P/B holding, market-cap >= $10B, inside each selected sector",
                   "benchmark": "SPY buy-and-hold over the same monthly windows"},
        "methods": results,
        "caveat": ("Every method uses the SAME value pick — only the sector-SELECTION rule changes. "
                   "20+ methods on ~62 monthly periods is a multiple-comparisons hazard: rank by t-stat / "
                   "Sharpe / drawdown, not by the single highest vs-SPY. Trust a read only when its whole "
                   "FAMILY agrees (nearby params give similar results). Directional, no fees; "
                   "stock-universe survivorship applies."),
    }
    return payload


def main():
    from pathlib import Path
    payload = build()
    out = Path(__file__).resolve().parent / ".data" / "studies" / "rs_methods.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="rs_methods",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[rs_methods]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== RS-TREND METHOD SWEEP ({payload['params']['periods']} monthly periods) ===", flush=True)
    print(f"{'method':30} {'family':10} {'tot%':>7} {'vsSPY%':>8} {'t':>6} {'sharpe':>7} {'maxDD%':>7} {'avgSel':>7}", flush=True)
    for r in payload["methods"]:
        print(f"{r['method']:30} {r['family']:10} {r['total_return']:>7} {r['vs_spy']:>8} "
              f"{str(r['t_stat']):>6} {r['sharpe']:>7} {r['max_drawdown']:>7} {str(r['avg_sectors']):>7}", flush=True)


if __name__ == "__main__":
    main()
