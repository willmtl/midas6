#!/usr/bin/env python3
"""SURVIVORSHIP-FREE SMALL-CAP RE-RUN (delisted de-bias, step 4) — the amplifier study found select_smallcap
(cheapest-P/B among <$2B names) = 463% but on a PRESENT-DAY-holdings universe (dead small-caps absent -> biased
UP). Now re-run WITH the 2,705 delisted names included as candidates during their ALIVE window (candle span;
they exit when their candles stop = delisting), mapped to sectors by GICS (survivors + delisted both via GICS,
since we can't get historical ETF membership).

Compares, on the SAME GICS-sector engine (accel top-10 -> pick -> div_2x, monthly):
  survivors_only_all   pick cheapest-P/B (any size)     | survivors_only_small  pick cheapest-P/B <$2B
  with_delisted_all    + delisted candidates            | with_delisted_small   + delisted, <$2B pick
The gap between survivors_only_small and with_delisted_small = the survivorship inflation in the 463% claim.
-> BacktestResult[survivorship_smallcap] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/survivorship_smallcap_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH
from api.tasks import _eodhd_get


def _pit_ttm_panel(reports_map, field, midx):
    """Point-in-time trailing-12-month panel for a FLOW field: rolling sum of the last 4 QUARTERLY values
    (ordered by period_end), forward-filled by avail_date to the monthly index. FinancialReport stores
    per-quarter flows, so a TTM sum is required for standard trailing P/E (=Price/TTM-EPS), P/S, EV/EBIT,
    FCF-yield, ROE — single-quarter figures are ~4x mis-scaled and can't go negative properly."""
    import pandas as pd
    out = {}
    for tk, r in reports_map.items():
        if field not in r.columns:
            continue
        d = r[["period_end", "avail_date", field]].dropna(subset=[field]).copy()
        if len(d) < 4:
            continue
        d = d.sort_values("period_end")
        d["ttm"] = d[field].rolling(4).sum()
        s = pd.Series(d["ttm"].values, index=pd.to_datetime(d["avail_date"])).dropna()
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
    return pd.DataFrame(out)


def _pit_ttm_ni(reports_map, midx):
    """Back-compat alias: TTM net-income panel."""
    return _pit_ttm_panel(reports_map, "net_income", midx)

# exchange-suffix -> reporting/quote currency. Market cap is computed in the QUOTE currency (price*shares) then
# converted to USD so the <$2B small-cap bucket is apples-to-apples. .L (London) quotes in PENCE -> ×0.01.
SUF_CCY = {"SA": "BRL", "MX": "MXN", "KS": "KRW", "HK": "HKD", "WA": "PLN", "TO": "CAD", "V": "CAD",
           "AX": "AUD", "SW": "CHF", "DE": "EUR", "PA": "EUR", "HE": "EUR", "L": "GBP", "MI": "EUR",
           "AS": "EUR", "BR": "EUR", "MC": "EUR", "ST": "SEK", "OL": "NOK", "CO": "DKK", "T": "JPY",
           "TSE": "JPY", "F": "EUR", "VI": "EUR", "LS": "EUR", "IR": "EUR",
           "TW": "TWD", "TWO": "TWD", "SZ": "CNY", "SS": "CNY", "JO": "ZAR"}   # Taiwan / China A-shares / Johannesburg
# fallback USD-per-1-unit if the live FX endpoint is unavailable (approx; live fetch is primary).
FX_FALLBACK = {"USD": 1.0, "BRL": 0.18, "MXN": 0.055, "KRW": 0.00072, "HKD": 0.128, "PLN": 0.25, "CAD": 0.72,
               "AUD": 0.65, "CHF": 1.12, "EUR": 1.08, "GBP": 1.27, "SEK": 0.095, "NOK": 0.093, "DKK": 0.145,
               "JPY": 0.0067, "TWD": 0.031, "CNY": 0.14, "ZAR": 0.055}


def _fx_monthly(cur, midx, frm):
    """Historical monthly USD-per-1-unit-of-`cur`, reindexed to midx (point-in-time, not a single current rate).
    Pulls daily FX history from EODHD (eod/{PAIR}.FOREX), resamples month-end, ffill/bfill. FX_FALLBACK if no data."""
    if cur == "USD":
        return pd.Series(1.0, index=midx)
    for pair, inv in ((f"{cur}USD", False), (f"USD{cur}", True)):
        r = _eodhd_get(f"eod/{pair}.FOREX", **{"from": frm, "period": "d"})
        if isinstance(r, list) and len(r) > 10:
            s = pd.Series({pd.Timestamp(x["date"]): x.get("adjusted_close", x.get("close")) for x in r}).astype(float)
            s = s[s > 0].sort_index()
            if inv:
                s = 1.0 / s
            return s.resample("ME").last().reindex(midx).ffill().bfill()
    v = FX_FALLBACK.get(cur)
    return pd.Series(v, index=midx) if v else pd.Series(np.nan, index=midx)


def _usd_factor_matrix(tickers, midx):
    """(midx x tickers) DataFrame of USD-per-quote-unit, POINT-IN-TIME. US->1.0, .L->×0.01 (pence),
    unknown currency->NaN (name is then not small-cap-classifiable). Returns (matrix, per-currency monthly series)."""
    frm = (midx[0] - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    ccy_ser = {}
    fac = pd.DataFrame(1.0, index=midx, columns=list(tickers))
    for tk in tickers:
        if "." not in tk:
            continue
        suf = tk.rsplit(".", 1)[1]
        ccy = SUF_CCY.get(suf)
        if not ccy:
            fac[tk] = np.nan; continue
        if ccy not in ccy_ser:
            ccy_ser[ccy] = _fx_monthly(ccy, midx, frm)
        fac[tk] = ccy_ser[ccy].values * (0.01 if suf == "L" else 1.0)
    return fac, ccy_ser


def _is_usca(tk):
    """US (no exchange suffix) or Canada (.TO / .V)."""
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")

TOP_N = 10; CONV = 2.0; MIN_DVOL = 5e6; SMALL = 2e9
MIN_PRICE = 0.0   # NO price floor — a genuine low-priced name is FINE as long as it's on a major exchange
                  # (user policy: keep the penny, gate on LISTING not price). See MAJOR_EXCH below.
MIN_PB = 0.1      # P/B SANITY FLOOR: reject sub-0.1 book multiples. These are ~always corrupt fundamentals
                  # (missing shares/equity -> fake ~0 P/B that auto-wins the cheapest-P/B pick, e.g. RTX shown
                  # at $109M mktcap, BKR/PKX/FOXA at $0). Genuine value bottoms out ~0.1-0.4; nothing real is <0.1.
PHARMA_ETFS = {"XLV", "XBI", "ARKG"}   # healthcare / biotech / genomics
MICRO_PHARMA_MIN = 5e7   # bar sub-$50M pharma/biotech = nano shells only (NLSP $3M-type). Keeps ALL legit small-
                         # pharma ($350-470M winners FATE/TVTX/CDNA untouched); removes just the true garbage.
                         # ($500M and $250M were too aggressive — sweep: 0/50M/250M/500M documented.)
# MAJOR-EXCHANGE gate (replaces the price floor): only NASDAQ / NYSE / AMEX-family. Excludes the ~16k OTC/PINK
# delisted names outright. Survivors come from current ETF membership so are inherently major-exchange.
MAJOR_EXCH = {"NASDAQ", "NYSE", "NYSE MKT", "NYSE ARCA", "AMEX", "BATS"}
# for the ETF-PROXY test (does holding raw commodity / bond sectors when they accelerate in ADD return?)
COMMODITY_ETFS = {"GLD", "SLV", "PPLT", "USO", "UNG", "DBA", "WEAT", "CORN", "DBC", "DBO", "BNO", "UGA",
                  "CPER", "PALL", "DBB", "GSG", "PDBC", "FTGC"}
BOND_ETFS = {"TLT", "TLH", "AGG", "BND", "HYG", "JNK", "TIP", "VTIP", "GOVT", "BIL", "SHV", "SHY", "IEI",
             "IEF", "LQD", "FLOT", "MUB", "CWB", "EMB", "BWX", "IGOV", "TIPX"}
GIC_FILE = "/app/.data/delisted_gic.json"
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "survivorship_smallcap.json"

# map EODHD GicSector / ETF names -> our sector-ETF buckets (accel is computed on the ETF; GICS just assigns
# which sector's candidate pool a stock joins). Broad GICS -> the core S&P sector ETF.
GIC_TO_ETF = {
    "Technology": "XLK", "Information Technology": "XLK", "Financials": "XLF", "Financial Services": "XLF",
    "Health Care": "XLV", "Healthcare": "XLV", "Energy": "XLE", "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB", "Materials": "XLB", "Real Estate": "XLRE",
    "Communication Services": "XLC", "Utilities": "XLU",
}


def _f(x):
    """Safe float for the trace dump: NaN/None-tolerant -> Python float or None."""
    try:
        return float(x) if pd.notna(x) else None
    except Exception:
        return None


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), annual=round(ann, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                dd=round(dd, 1), t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    # db /dev/shm is only 64MB (Docker default) -> parallel scans over the Candle hypertable can DiskFull.
    # Force single-threaded query plans (private work_mem, spills to disk temp) for this heavy read. See memory.
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    try:                                     # ticker -> company name, for the flagship-history trace (optional)
        NAMEMAP = json.load(open("/app/.data/ticker_names.json"))
    except Exception:
        NAMEMAP = {}
    # RUNTIME universe drop (A/B a sector removal with no config edit, nothing to revert): DROP_ETFS="ARKK,QTUM"
    _drop = {x.strip() for x in os.environ.get("DROP_ETFS", "").split(",") if x.strip()}
    if _drop:
        etfs = {n: e for n, e in etfs.items() if e not in _drop}
        print(f"DROP_ETFS active: removed {sorted(_drop)} -> {len(etfs)} sectors remain", flush=True)
    etf_name = {e: n for n, e in etfs.items()}   # ETF ticker -> sector display name (for the trace)
    # survivor sector map + GICS map for survivors (via sector_holdings membership -> their ETF)
    surv_sector = {}                                  # ticker -> etf (survivor, by current ETF membership)
    all_holds = set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                surv_sector.setdefault(t, e); all_holds.add(t)

    # delisted GICS map
    try:
        gic_raw = json.load(open(GIC_FILE))
    except Exception:
        gic_raw = {}
    delisted_sector = {}
    from core.models import FinancialReport, Candle, DelistedCompany
    dl_have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    # exchange gate: keep delisted names only if they were on a MAJOR exchange (no OTC/pink). Penny prices OK.
    dl_exch = {d.ticker: (d.exchange or "").strip()
               for d in DelistedCompany.objects.filter(ticker__in=list(gic_raw))}
    n_gic, n_otc = 0, 0
    for tk, gic in gic_raw.items():
        e = GIC_TO_ETF.get((gic or "").strip())
        if e and tk in dl_have:
            n_gic += 1
            if dl_exch.get(tk) not in MAJOR_EXCH:
                n_otc += 1; continue          # drop OTC / pink-sheet delisted names
            delisted_sector[tk] = e
    print(f"survivors {len(surv_sector)} | delisted GIC+fund mapped {n_gic} | dropped OTC/pink {n_otc} | "
          f"kept major-exchange {len(delisted_sector)}", flush=True)

    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)

    universe = sorted(all_holds | set(delisted_sector))
    stock_daily = load_candles(universe)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(universe)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    as_traded = price_basis.as_traded_close(px)
    mktcap = as_traded * sh                       # QUOTE-currency market cap -> used for P/B (ratio, currency cancels)
    pb = mktcap / eq.where(eq != 0)
    de = dt / eq.where(eq != 0)                    # debt-to-equity (PIT) — for the flagship-history trace
    # POINT-IN-TIME FX: we trade in USD, so returns must include FX gain/loss. Convert the price series to USD at
    # each date's historical rate; returns are then computed on the USD series (local_return × fx_return).
    usd_factor_m, ccy_ser = _usd_factor_matrix(list(common), midx)
    ret_factor = usd_factor_m.fillna(1.0)         # unknown-ccy names: keep (no conv) rather than drop; logged below
    px_usd = px * ret_factor                       # <- USD-translated price series (returns include FX P&L)
    as_traded_usd = as_traded * ret_factor
    mktcap_usd = mktcap * usd_factor_m             # size bucket: unknown ccy stays NaN (not classifiable as 'small')
    unknown = [t for t in common if "." in t and usd_factor_m[t].isna().all()]
    print(f"FX point-in-time USD-per-unit ({len(ccy_ser)} currencies); unmapped-ccy names kept-unconverted: "
          f"{len(unknown)} {unknown[:8]}", flush=True)
    for cur, s in sorted(ccy_ser.items()):
        print(f"  FX {cur}->USD  first={s.iloc[0]:.4g} last={s.iloc[-1]:.4g} min={s.min():.4g} max={s.max():.4g}", flush=True)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    # ROE for the PREMIUM-NORMALIZED value test: justified P/B rises with ROE, so raw-cheapest-P/B avoids
    # quality (mega-caps like GOOG never qualify). P/B÷ROE = cheapness PER UNIT of quality (lower = cheaper).
    roe = ni / eq.where(eq != 0)
    pb_roe = pb / roe.where(roe > 0)              # crude heuristic: only meaningful for positive-ROE names
    # DISPLAY-ONLY standard trailing metrics for the flagship-history doc. ni above is a single QUARTER, so
    # pb_roe is a ~4x-inflated quarterly P/E that is masked to NaN for loss-makers (hence blank, never negative).
    # Here we build the STANDARD signed trailing P/E = MktCap / TTM-net-income (= Price / TTM-EPS), negative when
    # loss-making. This does NOT feed any ranking (usca_small ranks on raw pb; pb_roe is unchanged).
    ttm_ni = R(_pit_ttm_ni(reps, midx))
    roe_ttm = ttm_ni / eq.where(eq != 0)                   # trailing-12m ROE (signed)
    pe_ttm = mktcap / ttm_ni.where(ttm_ni != 0)            # signed trailing P/E = Price / TTM-EPS
    # TTM flow panels for the flagship VALUE-METRIC bake-off (does raw P/B still win the small-cap pick vs
    # properly-TTM P/E / P/S / EV-EBIT / FCF-yield? — audit finding #4 + "test everything"). All "lower=better".
    ttm_rev = R(_pit_ttm_panel(reps, "revenue", midx))
    ttm_opinc = R(_pit_ttm_panel(reps, "operating_income", midx))
    ttm_fcf = R(_pit_ttm_panel(reps, "free_cash_flow", midx))
    ttm_cash = R(_pit_monthly_panel(reps, "cash_and_equivalents", midx))   # cash = stock (point-in-time)
    _ev = mktcap + dt.fillna(0) - ttm_cash.fillna(0)
    pe_ttm_pos = pe_ttm.where(pe_ttm > 0)                   # cheapest POSITIVE trailing P/E (loss-makers excluded)
    ps_ttm = mktcap / ttm_rev.where(ttm_rev > 0)            # cheapest P/S (TTM sales)
    evebit_ttm = _ev / ttm_opinc.where(ttm_opinc > 0)      # cheapest EV/EBIT (TTM, positive EBIT)
    fcfy_ttm = -(ttm_fcf / _ev.where(_ev > 0))             # highest FCF/EV -> negate so min = best
    # RIGOROUS #1 — justified P/B from the residual-income / Gordon model: P/B* = (ROE - g)/(r - g).
    # r = assumed cost of equity (fixed 9%); g = sustainable growth proxied by trailing YoY book-equity growth,
    # clamped < r (the model diverges as g->r). Signal = actual P/B / justified P/B  (<1 = cheaper than deserved).
    R_EQ = 0.09
    g_bk = (eq / eq.shift(12) - 1).clip(-0.20, R_EQ - 0.01)
    justified_pb = ((roe - g_bk) / (R_EQ - g_bk)).where(lambda d: d > 0)
    pb_vs_just = pb / justified_pb
    # RIGOROUS #2 — cross-sectional P/B~ROE "fair-value line": each month OLS-fit P/B on ROE across the
    # universe (winsorized 1/99pct), residual = actual - fitted; most-negative residual = cheapest vs the line.
    resid = pd.DataFrame(np.nan, index=midx, columns=common)
    for _date in midx:
        x = roe.loc[_date].astype(float); y = pb.loc[_date].astype(float)
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 20:
            continue
        xv, yv = x[m], y[m]
        xw = xv.clip(xv.quantile(.01), xv.quantile(.99)); yw = yv.clip(yv.quantile(.01), yv.quantile(.99))
        if xw.std() < 1e-9:
            continue
        try:
            b, a = np.polyfit(xw.values, yw.values, 1)
        except Exception:
            continue
        resid.loc[_date] = (y - (a + b * x))

    # ── MORE PREMIUM ALTERNATIVES (2026-08-16, user: "try more alternatives to calculate that premium") ──
    # extra PIT quality panels: gross-profitability (Novy-Marx GP/A), revenue growth
    revp, gpp, tap = (_pit_monthly_panel(reps, f, midx) for f in ("revenue", "gross_profit", "total_assets"))
    revp, gpp, tap = R(revp), R(gpp), R(tap)
    gpa = gpp / tap.where(tap != 0)
    rev_g = revp / revp.shift(12) - 1
    # (A) justified P/B with CAPM cost of equity: r = rf(4%) + beta*ERP(5%), beta = rolling-12mo vs SPY (PIT).
    stk_ret = px_usd.pct_change(); spy_ret = spy_m.pct_change()
    beta = pd.DataFrame(index=midx, columns=common, dtype=float)
    for t in common:
        beta[t] = stk_ret[t].rolling(12).cov(spy_ret) / spy_ret.rolling(12).var()
    beta = beta.clip(0.2, 3.0)
    R_CAPM = 0.04 + beta * 0.05
    g_capm = (eq / eq.shift(12) - 1).clip(-0.20, None)
    g_capm = g_capm.where(g_capm < R_CAPM - 0.01, R_CAPM - 0.01)
    just_capm = ((roe - g_capm) / (R_CAPM - g_capm)).where(lambda d: d > 0)
    pb_vs_capm = pb / just_capm
    # (B) rank-based residual (robust: ranks kill the OLS-outlier fragility that failed the plain residual's H1)
    # (C) multi-factor fair-value line: P/B ~ ROE + GP/A + rev_growth (winsorized OLS), residual = cheap vs quality
    resid_rk = pd.DataFrame(np.nan, index=midx, columns=common)
    resid_mf = pd.DataFrame(np.nan, index=midx, columns=common)
    for _d in midx:
        x = roe.loc[_d].astype(float); y = pb.loc[_d].astype(float)
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) >= 20:
            xr = x[m].rank(pct=True); yr = y[m].rank(pct=True)
            if xr.std() > 1e-9:
                b, a = np.polyfit(xr.values, yr.values, 1)
                res = pd.Series(np.nan, index=common); res.loc[m[m].index] = (yr - (a + b * xr)).values
                resid_rk.loc[_d] = res.values
        Y = pb.loc[_d].astype(float)
        X = pd.DataFrame({"roe": roe.loc[_d], "gpa": gpa.loc[_d], "rev": rev_g.loc[_d]}).astype(float)
        mm = Y.notna() & np.isfinite(Y) & X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
        if int(mm.sum()) >= 30:
            yv = Y[mm].clip(Y[mm].quantile(.01), Y[mm].quantile(.99))
            Xv = X[mm].copy()
            for c in Xv.columns:
                Xv[c] = Xv[c].clip(Xv[c].quantile(.01), Xv[c].quantile(.99))
            try:
                coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xv)), Xv.values]), yv.values, rcond=None)
                pred = coef[0] + X[["roe", "gpa", "rev"]].fillna(0).values @ coef[1:]
                res2 = pd.Series(np.nan, index=common); res2[mm] = (Y - pred)[mm]
                resid_mf.loc[_d] = res2.values
            except Exception:
                pass

    # ── TWO-STAGE panels (2026-08-16, user: top-5 cheapest raw P/B per sector, then a SECONDARY signal picks 1) ──
    # stock momentum / acceleration (monthly, USD); A/D divergence already via accumulating(); P/E=pb_roe, ROE, GP/A, resid ready.
    smom6 = px_usd.pct_change(6)
    _s3 = px_usd.pct_change(3)
    saccel = _s3 - _s3.shift(3)

    def _rsi(c, n=10):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn.replace(0, np.nan))

    rsi_m = pd.DataFrame(np.nan, index=midx, columns=common)
    rsi_bull_m = pd.DataFrame(False, index=midx, columns=common)
    freshcross_m = pd.DataFrame(False, index=midx, columns=common)
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Close" not in d or len(d) < 40:
            continue
        r = _rsi(d["Close"], 10); s = r.rolling(10).mean()
        bull = r > s
        cross = bull & (~bull.shift(1).fillna(False))    # RSI crossed above its SMA (dashboard's fresh signal)
        rsi_m[t] = r.resample("ME").last().reindex(midx)
        rsi_bull_m[t] = bull.resample("ME").last().reindex(midx).fillna(False).astype(bool)
        freshcross_m[t] = (cross.rolling(15).max() > 0).resample("ME").last().reindex(midx).fillna(False).astype(bool)
    print("two-stage secondary-signal panels built", flush=True)
    dvol, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 60:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    dvol_usd = dvol * ret_factor                   # $5M liquidity floor must be in USD (KRW 5M is ~$3.6k, not $5M)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    # sector -> candidate tickers (survivors always; delisted optionally)
    def sector_cands(etf, include_delisted):
        out = [t for t, e in surv_sector.items() if e == etf and t in common]
        if include_delisted:
            out += [t for t, e in delisted_sector.items() if e == etf and t in common]
        return out

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    from collections import Counter

    def pick5(pool5, date, method):
        """STAGE 2: given the 5 cheapest raw-P/B names in a sector (pool5, cheapest first), pick ONE by a
        secondary signal. Every method falls back to pool5[0] (=the flagship cheapest-P/B pick) when its signal
        is unavailable, so the control 'cheapest_pb' reproduces the flagship exactly."""
        if not pool5:
            return None
        if method == "cheapest_pb" or len(pool5) == 1:
            return pool5[0]
        def _min(panel):
            q = [(h, panel.loc[date, h]) for h in pool5 if pd.notna(panel.loc[date, h])]
            return min(q, key=lambda x: x[1])[0] if q else pool5[0]
        def _max(panel):
            q = [(h, panel.loc[date, h]) for h in pool5 if pd.notna(panel.loc[date, h])]
            return max(q, key=lambda x: x[1])[0] if q else pool5[0]
        if method == "pe":         return _min(pb_roe)      # cheapest P/E (=P/B÷ROE) among the 5
        if method == "roe":        return _max(roe)         # highest quality
        if method == "gpa":        return _max(gpa)         # highest gross-profitability
        if method == "resid":      return _min(resid)       # furthest below P/B~ROE line
        if method == "rsi_os":     return _min(rsi_m)       # most oversold (buy the dip — memory: helps)
        if method == "accel_fade": return _min(saccel)      # buy the fading laggard (stock accel: FADE it)
        if method == "accel_ride": return _max(saccel)      # buy the accelerating one (expect: hurts)
        if method == "mom6_hi":    return _max(smom6)        # buy strength (expect: hurts)
        if method == "mom6_lo":    return _min(smom6)        # buy the most-beaten (deep weakness)
        if method == "rsi_cross":                            # first (cheapest) with a fresh RSI>SMA cross
            q = [h for h in pool5 if bool(freshcross_m.loc[date, h])]
            return q[0] if q else pool5[0]
        if method == "rsi_bull":                             # first (cheapest) in RSI>SMA uptrend
            q = [h for h in pool5 if bool(rsi_bull_m.loc[date, h])]
            return q[0] if q else pool5[0]
        if method == "ad_div":                               # first (cheapest) accumulating into weakness
            q = [h for h in pool5 if accumulating(h, date)]
            return q[0] if q else pool5[0]
        return pool5[0]

    def run(include_delisted, small_only, min_price=MIN_PRICE, country_ok=None, proxy_etf=False, value_key="pb",
            top5=None, capaware=None, trace=None, ban_first_loss=False, pb_ceiling=None, drop_sectors=None,
            exclude_tickers=None, start_date=None, end_date=None):
        rets, spies, dl_picks, mrets = [], [], 0, []
        proxy_hold = Counter()          # etf -> # months held as a no-value-stock proxy (the live fallback)
        proxy_contrib = 0.0             # sum of proxy monthly contributions to the basket (weighted)
        mega_picks = 0                  # picks with >$50B USD mktcap (does premium-normalization let mega-caps in?)
        traded = set(); banned = set()  # ban_first_loss: names whose FIRST-ever trade lost -> never buy again
        _sd = pd.Timestamp(start_date) if start_date else None
        _ed = pd.Timestamp(end_date) if end_date else None
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            if (_sd is not None and date < _sd) or (_ed is not None and date > _ed):
                continue                # window restriction (apples-to-apples sub-period walk-forward)
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            _acc = accel.loc[date].dropna()
            if drop_sectors:                      # "remove ARK" etc: drop these ETFs, backfill the slot from #11
                _acc = _acc.drop(labels=[e for e in drop_sectors if e in _acc.index], errors="ignore")
            top = _acc.sort_values(ascending=False).head(TOP_N).index

            def pbceil_ok(h):
                """P/B ceiling gate. pb_ceiling may be None (off), a flat float, or a cap-tiered dict with keys
                'micro' (<$500M), 'small' (<$2B), 'large' (>=$2B) and optional 'default'."""
                if pb_ceiling is None:
                    return True
                v = pb.loc[date, h]
                if not pd.notna(v):
                    return True
                if isinstance(pb_ceiling, dict):
                    mc = mktcap_usd.loc[date, h]
                    key = ("large" if (pd.notna(mc) and mc >= 2e9) else
                           "micro" if (pd.notna(mc) and mc < 5e8) else "small")
                    cap = pb_ceiling.get(key, pb_ceiling.get("default"))
                    return cap is None or v <= cap
                return v <= pb_ceiling

            held = set(); wsum = rr = 0.0
            tr = None
            if trace is not None:
                tr = {"date": str(pd.Timestamp(date).date()), "ndate": str(pd.Timestamp(ndate).date()),
                      "top_sectors": [{"sector": etf_name.get(e, e), "etf": e, "accel": _f(accel.loc[date, e])}
                                      for e in top],
                      "picks": [], "skipped": []}
            for etf in top:
                pharma = etf in PHARMA_ETFS
                cands = [h for h in sector_cands(etf, include_delisted) if h not in held
                         and (not ban_first_loss or h not in banned)
                         and (exclude_tickers is None or h not in exclude_tickers)
                         and pbceil_ok(h)
                         and (country_ok is None or country_ok(h))
                         and _available_at(px_usd[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > MIN_PB
                         and pd.notna(as_traded_usd.loc[date, h]) and as_traded_usd.loc[date, h] >= min_price
                         and not bool(trap.loc[date, h]) and pd.notna(dvol_usd.loc[date, h]) and dvol_usd.loc[date, h] >= MIN_DVOL
                         and not (pharma and (pd.isna(mktcap_usd.loc[date, h]) or mktcap_usd.loc[date, h] < MICRO_PHARMA_MIN))]
                g0 = [x for x in cands if bool(low.loc[date, x])] or cands
                sm = [x for x in g0 if pd.notna(mktcap_usd.loc[date, x]) and mktcap_usd.loc[date, x] < SMALL]
                g = (sm or g0) if (small_only or capaware is not None) else g0
                if not g:
                    # no qualifying value stock. usca_small SKIPS the slot; proxy_etf HOLDS the ETF itself
                    # (the live is_etf_proxy fallback: raw commodities, bonds, foreign/index markets).
                    if tr is not None:
                        raw = sector_cands(etf, include_delisted)
                        n_country = sum(1 for h in raw if country_ok is None or country_ok(h))
                        if not raw:
                            reason = "ETF sleeve has no mapped equity holdings (raw commodity / bond / index)"
                        elif n_country == 0:
                            reason = "no US/Canada-listed holding — foreign / commodity / bond sleeve"
                        elif not cands:
                            reason = "holdings exist but none cleared the filters (P/B>0, price, $5M liquidity, value-trap)"
                        elif small_only and not sm:
                            reason = "only large-caps qualified — no small-cap (<$2B) value name in the sleeve"
                        else:
                            reason = "no qualifying value stock"
                        tr["skipped"].append({"sector": etf_name.get(etf, etf), "etf": etf, "reason": reason,
                                              "n_holdings": len(raw), "n_usca": n_country,
                                              "accel": _f(accel.loc[date, etf]) if etf in accel.columns else None})
                    if proxy_etf:
                        re = etf_m[etf].iloc[i + 1] / etf_m[etf].iloc[i] - 1 if etf in etf_m.columns else np.nan
                        if np.isfinite(re):
                            proxy_hold[etf] += 1; proxy_contrib += float(re)
                            wsum += 1.0; rr += 1.0 * float(re)
                    continue
                if capaware is not None:        # LIVE cap-aware rule + configurable no-profitable-small-cap fallback
                    if sm:                       # small-cap tier: top-5 cheapest P/B -> cheapest P/E among profitable
                        pool5 = sorted(sm, key=lambda h: pb.loc[date, h])[:5]
                        prof = [x for x in pool5 if pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                        if prof:
                            # ADAPT: rank profitable small-caps by TTM P/E (corrected); else the OLD quarterly pb_roe.
                            _pe = pe_ttm_pos if capaware == "adapt" else pb_roe
                            p = min(prof, key=lambda h: (_pe.loc[date, h] if pd.notna(_pe.loc[date, h]) else 9e18))
                        elif capaware == "prof_any":   # NO profitable small -> prefer cheapest-P/B PROFITABLE large-cap
                            profL = [x for x in g0 if pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                            p = min(profL, key=lambda h: pb.loc[date, h]) if profL else pool5[0]
                        elif capaware == "skip":       # NO profitable small -> skip the sector entirely
                            continue
                        else:                          # "loss_small"/"adapt" -> cheapest-P/B loss-making small (keep exposure)
                            p = pool5[0]
                    else:                        # no small-cap in sector -> the large-cap fallback
                        if capaware == "adapt":  # ADAPT: raw P/B is a BAD large-cap selector -> use cheapest TTM P/E
                            profL = [x for x in g0 if pd.notna(pe_ttm_pos.loc[date, x])]
                            p = min(profL, key=lambda h: pe_ttm_pos.loc[date, h]) if profL else min(g0, key=lambda h: pb.loc[date, h])
                        else:                    # current live rule: cheapest raw P/B large-cap
                            p = min(g0, key=lambda h: pb.loc[date, h])
                elif top5 is not None:          # TWO-STAGE: 5 cheapest raw P/B, then a secondary signal picks 1
                    pool5 = sorted(g, key=lambda h: pb.loc[date, h])[:5]
                    p = pick5(pool5, date, top5)
                elif value_key == "pb_roe":     # crude heuristic
                    q = [x for x in g if pd.notna(pb_roe.loc[date, x])]
                    p = min(q, key=lambda h: pb_roe.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key in ("pe_ttm", "ps_ttm", "evebit_ttm", "fcfy_ttm"):  # TTM value-metric bake-off
                    _M = {"pe_ttm": pe_ttm_pos, "ps_ttm": ps_ttm, "evebit_ttm": evebit_ttm, "fcfy_ttm": fcfy_ttm}[value_key]
                    q = [x for x in g if pd.notna(_M.loc[date, x])]
                    p = min(q, key=lambda h: _M.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "justified":  # rigorous #1: P/B vs (ROE-g)/(r-g)
                    q = [x for x in g if pd.notna(pb_vs_just.loc[date, x])]
                    p = min(q, key=lambda h: pb_vs_just.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "residual":   # rigorous #2: furthest below the P/B~ROE line
                    q = [x for x in g if pd.notna(resid.loc[date, x])]
                    p = min(q, key=lambda h: resid.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "justified_capm":     # alt A: justified P/B with CAPM cost of equity
                    q = [x for x in g if pd.notna(pb_vs_capm.loc[date, x])]
                    p = min(q, key=lambda h: pb_vs_capm.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "resid_rk":           # alt B: rank-based (robust) residual
                    q = [x for x in g if pd.notna(resid_rk.loc[date, x])]
                    p = min(q, key=lambda h: resid_rk.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "resid_mf":           # alt C: multi-factor fair-value residual
                    q = [x for x in g if pd.notna(resid_mf.loc[date, x])]
                    p = min(q, key=lambda h: resid_mf.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "pb_prof":   # CONFOUND CHECK: cheapest RAW P/B among PROFITABLE (ROE>0) names.
                    # crude P/B÷ROE == P/E, ranked among ni>0. This isolates whether the win is the P/E RANKING
                    # or just the profitable-only restriction (same restriction, but rank by P/B not P/E).
                    q = [x for x in g if pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                    p = min(q, key=lambda h: pb.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key in ("roe_gate", "gpa_gate"):   # alt D/E: quality GATE (top-half) then cheapest RAW P/B
                    qp = roe if value_key == "roe_gate" else gpa
                    vals = [(x, qp.loc[date, x]) for x in g if pd.notna(qp.loc[date, x])]
                    if vals:
                        med = float(np.median([v for _, v in vals]))
                        keep = [x for x, v in vals if v >= med] or [x for x, _ in vals]
                        p = min(keep, key=lambda h: pb.loc[date, h])
                    else:
                        p = min(g, key=lambda h: pb.loc[date, h])
                else:
                    p = min(g, key=lambda h: pb.loc[date, h])
                held.add(p)
                if pd.notna(mktcap_usd.loc[date, p]) and mktcap_usd.loc[date, p] > 5e10:
                    mega_picks += 1
                r = _ret_delist(px_usd[p], date, ndate)      # return on the USD-translated series -> includes FX P&L
                if r is None or not np.isfinite(r):
                    if tr is not None:
                        tr["picks"].append({"sector": etf_name.get(etf, etf), "etf": etf, "ticker": p,
                                            "company": NAMEMAP.get(p), "pb": _f(pb.loc[date, p]),
                                            "pe": _f(pe_ttm.loc[date, p]), "roe": _f(roe_ttm.loc[date, p]),
                                            "de": _f(de.loc[date, p]), "gpa": _f(gpa.loc[date, p]),
                                            "rev_g": _f(rev_g.loc[date, p]), "ni": _f(ttm_ni.loc[date, p]),
                                            "revenue": _f(revp.loc[date, p]), "mktcap_usd": _f(mktcap_usd.loc[date, p]),
                                            "weight": None, "ret": None, "delisted": p in delisted_sector,
                                            "conviction": bool(accumulating(p, date))})
                    continue
                if p in delisted_sector:
                    dl_picks += 1
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
                if ban_first_loss and p not in traded:   # record FIRST-ever trade; ban if it was a loss
                    traded.add(p)
                    if float(r) < 0:
                        banned.add(p)
                if tr is not None:
                    tr["picks"].append({"sector": etf_name.get(etf, etf), "etf": etf, "ticker": p,
                                        "company": NAMEMAP.get(p), "pb": _f(pb.loc[date, p]),
                                        "pe": _f(pe_ttm.loc[date, p]), "roe": _f(roe_ttm.loc[date, p]),
                                        "de": _f(de.loc[date, p]), "gpa": _f(gpa.loc[date, p]),
                                        "rev_g": _f(rev_g.loc[date, p]), "ni": _f(ttm_ni.loc[date, p]),
                                        "revenue": _f(revp.loc[date, p]), "mktcap_usd": _f(mktcap_usd.loc[date, p]),
                                        "weight": float(w), "ret": float(r), "delisted": p in delisted_sector,
                                        "conviction": bool(accumulating(p, date))})
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp)); mrets.append((str(pd.Timestamp(date).date()), float(rr / wsum)))
            if tr is not None:
                tr["basket_ret"] = float(rr / wsum); tr["spy_ret"] = float(sp); trace.append(tr)
        perf = _perf(rets, spies); perf["delisted_picks"] = dl_picks; perf["mega_picks"] = mega_picks
        perf["monthly"] = mrets
        if proxy_etf:
            comm = {e: n for e, n in proxy_hold.items() if e in COMMODITY_ETFS}
            bond = {e: n for e, n in proxy_hold.items() if e in BOND_ETFS}
            other = {e: n for e, n in proxy_hold.items() if e not in COMMODITY_ETFS and e not in BOND_ETFS}
            perf["proxy_months_total"] = int(sum(proxy_hold.values()))
            perf["proxy_commodity"] = {"months": int(sum(comm.values())), "etfs": dict(sorted(comm.items(), key=lambda x: -x[1]))}
            perf["proxy_bond"] = {"months": int(sum(bond.values())), "etfs": dict(sorted(bond.items(), key=lambda x: -x[1]))}
            perf["proxy_other"] = {"months": int(sum(other.values())), "etfs": dict(sorted(other.items(), key=lambda x: -x[1])[:12])}
        return perf

    # ── FLAGSHIP HISTORY TRACE (opt-in): run ONLY the usca_small arm, recording every sector/stock pick per
    # month, dump to flagship_history.json, and EXIT before the full sweep so the stored BacktestResult is
    # untouched. Trigger: FLAGSHIP_TRACE=1 docker exec ... python /app/survivorship_smallcap_study.py ──
    if os.environ.get("WEEKLY_ACCEL"):
        # ── WEEKLY-BAR SECTOR RANKING test (no save): the live engine ranks sectors by ACCELERATION on
        # MONTHLY bars (accel = pct_change(3) − shift(3), a 6mo two-sided window). Here we recompute the
        # ranking on WEEKLY (W-FRI) bars at several lookbacks — both acceleration and pure momentum —
        # reindexed to the SAME month-end rebalance dates (ffill = most recent weekly value as of the buy,
        # so it stays point-in-time). Everything downstream (pick, weight, monthly hold) is IDENTICAL, so
        # this isolates ONLY 'does measuring sector momentum on weekly bars change the picks / the return'.
        import sys
        etf_w = pd.DataFrame({t: etf_daily[t]["Close"].resample("W-FRI").last()
                              for t in etf_tk if t in etf_daily})
        variants = {"monthly_base 3-3 (6mo)": accel}          # the LIVE ranking (run same-universe for A/B)
        for win, lab in [(4, "~2mo"), (8, "~4mo"), (13, "~6mo"), (26, "~12mo")]:
            wa = etf_w.pct_change(win) - etf_w.pct_change(win).shift(win)   # two-sided accel on weekly bars
            variants[f"wk_accel {win}w ({lab})"] = wa.reindex(midx, method="ffill")
        for win, lab in [(13, "3mo"), (26, "6mo")]:
            wm = etf_w.pct_change(win)                                       # pure momentum (no acceleration)
            variants[f"wk_mom {win}w ({lab})"] = wm.reindex(midx, method="ffill")
        base = {"usca_small": 790.4, "usca_small_norm": 1024.0, "usca_small_pbprof": 911.5}
        arms = [("usca_small", dict(country_ok=_is_usca)),
                ("usca_small_norm", dict(country_ok=_is_usca, value_key="pb_roe")),
                ("usca_small_pbprof", dict(country_ok=_is_usca, value_key="pb_prof"))]
        print("\n=== WEEKLY_ACCEL (no save): sector ranking on WEEKLY bars vs the live monthly accel ===", flush=True)
        print(f"  stored monthly baselines -> small {base['usca_small']} / norm {base['usca_small_norm']} "
              f"/ pbprof {base['usca_small_pbprof']}", flush=True)
        for vname, acc in variants.items():
            accel = acc                          # reassign enclosing var -> run()'s closure uses the weekly ranking
            cells = []
            for aname, kw in arms:
                r = run(True, True, **kw)
                tag = aname.split("_")[-1]
                cells.append(f"{tag} {r['total']:7.1f}% (b{base[aname]:6.1f} {r['total']-base[aname]:+6.1f}) "
                             f"Sh{r['sharpe']:.2f} DD{r['dd']:.0f}% t{r['t_stat']}")
            print(f"  {vname:24} | " + "  ||  ".join(cells), flush=True)
        sys.exit(0)

    if os.environ.get("ADAPT_TEST"):
        # ── "adapt accordingly": cap-aware EARNINGS selector (small: cheapest TTM P/E among profitable in the
        # top-5 cheapest-P/B; large fallback: cheapest TTM P/E, NOT raw P/B which loses all-cap). Walk-forward
        # vs current flagship (raw pb) + pb_prof across windows — only believe it if it holds ex-2020 AND both halves. ──
        import sys
        wins = [("FULL 2019→now", None, None), ("ex-2020 (2021→now)", "2021-01-31", None),
                ("H1 2019→2022", None, "2022-12-31"), ("H2 2023→2026", "2023-01-31", None)]
        arms = [("pb (flagship)", dict(value_key="pb")), ("pb_prof", dict(value_key="pb_prof")),
                ("ADAPT (cap-aware P/E)", dict(capaware="adapt"))]
        print("\n=== ADAPT_TEST (no save): cap-aware earnings selector, walk-forward (total% / Sharpe) ===", flush=True)
        print(f"  {'window':22}" + "".join(f"{a[0]:>24}" for a in arms), flush=True)
        for lab, sd, ed in wins:
            cells = []
            for _, kw in arms:
                r = run(True, True, country_ok=_is_usca, start_date=sd, end_date=ed, **kw)
                cells.append(f"{r['total']:>10.0f}% Sh{r['sharpe']:.2f} t{r['t_stat']}")
            print(f"  {lab:22}" + "".join(f"{c:>24}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("WINDOW_WF"):
        # ── APPLES-TO-APPLES + WALK-FORWARD: raw pb vs pb_prof across sub-windows. The full window now spans
        # 2019-06->2026-08 (candle backfill extended history); the OLD "790%" flagship started ~2021. Does the
        # pb_prof edge hold on the SAME 2021-start window as before, and on each half — or only with 2020 in? ──
        import sys
        wins = [("FULL 2019-06→2026-08", None, None),
                ("OLD window 2021-07→now", "2021-07-31", None),
                ("H1 2019→2022-12", None, "2022-12-31"),
                ("H2 2023→2026", "2023-01-31", None),
                ("ex-2020 (2021→now)", "2021-01-31", None)]
        print("\n=== WINDOW_WF (no save): raw pb vs pb_prof across sub-windows (apples-to-apples) ===", flush=True)
        print(f"  {'window':26}{'pb':>12}{'pb_prof':>12}{'edge(pp)':>10}{'pb Sh':>7}{'prof Sh':>8}", flush=True)
        for lab, sd, ed in wins:
            a = run(True, True, country_ok=_is_usca, value_key="pb", start_date=sd, end_date=ed)
            b = run(True, True, country_ok=_is_usca, value_key="pb_prof", start_date=sd, end_date=ed)
            print(f"  {lab:26}{a['total']:>11.1f}%{b['total']:>11.1f}%{b['total']-a['total']:>+10.1f}"
                  f"{a['sharpe']:>7.2f}{b['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("FLAGSHIP_VALUE_TEST"):
        # ── does raw cheapest-P/B still win the SMALL-CAP FLAGSHIP pick vs properly-TTM P/E / P/S / EV-EBIT /
        # FCF-yield? (the all-cap value_ranking_lab now says P/B LOSES; re-test inside usca_small on current
        # data with TTM-correct flows). Same usca_small harness, only value_key varies. No save. ──
        import sys
        B = 790.4
        arms = [("pb (raw, FLAGSHIP)", "pb"), ("pe_ttm (cheapest +P/E)", "pe_ttm"),
                ("pb_prof (P/B, profitable)", "pb_prof"), ("ps_ttm (P/S)", "ps_ttm"),
                ("evebit_ttm (EV/EBIT)", "evebit_ttm"), ("fcfy_ttm (FCF yield)", "fcfy_ttm"),
                ("pb_roe (OLD quarterly P/E)", "pb_roe")]
        print("\n=== FLAGSHIP_VALUE_TEST (no save): value metric inside usca_small small-cap (baseline pb=790.4) ===", flush=True)
        rows = []
        for lab, vk in arms:
            r = run(True, True, country_ok=_is_usca, value_key=vk)
            rows.append((lab, r))
            print(f"  {lab:28} {r['total']:8.1f}% ({r['total']-B:+7.1f} vs pb)  Sh{r['sharpe']:.2f}  DD{r['dd']:.0f}%  t{r['t_stat']}", flush=True)
        best = max(rows, key=lambda x: x[1]['total'])
        print(f"\n  BEST small-cap value metric = {best[0]} ({best[1]['total']:.1f}%). "
              f"{'raw P/B still wins.' if best[1]=='pb' or 'pb (raw' in best[0] else 'raw P/B does NOT win the small-cap pick either.'}", flush=True)
        sys.exit(0)

    if os.environ.get("MISCAT_TEST"):
        # ── does excluding the audited MISCATEGORIZED picks (RIO=iron in Lithium&Battery, etc.) help? A first
        # proxy for a GICS-sector-consistency gate: exclude the tickers the audit flagged as sitting in a sleeve
        # whose theme they don't match, and A/B vs baseline. Hindsight (like ARK removal), but it measures the
        # cost of the mismatches and whether a real gate is worth building. ──
        import sys
        B = 790.4
        # (ticker, why) from the miscategorization audit; RIO is the worst (iron-ore miner, 9 months held)
        MISCAT = ["RIO", "PENN", "SIRI", "P", "HASI", "LRN", "TME", "FFIV", "TRIP"]
        CLEAREST = ["RIO", "PENN", "SIRI", "P", "HASI"]  # the 5 most clearly wrong

        def R1(label, **kw):
            r = run(True, True, country_ok=_is_usca, **kw)
            print(f"  {label:34} {r['total']:8.1f}% ({r['total']-B:+7.1f})  Sh{r['sharpe']:.2f}  DD{r['dd']:.0f}%  t{r['t_stat']}", flush=True)

        print("\n=== MISCAT_TEST (no save): exclude audit-flagged miscategorized picks (baseline 790.4) ===", flush=True)
        R1("baseline")
        R1("exclude RIO only (worst, 9mo)", exclude_tickers={"RIO"})
        R1("exclude 5 clearest mismatches", exclude_tickers=set(CLEAREST))
        R1("exclude all 9 flagged", exclude_tickers=set(MISCAT))
        sys.exit(0)

    if os.environ.get("CEILING_TEST"):
        # ── cap-tiered P/B ceiling sweep (no save). User: big-caps like GOOG legitimately trade at P/B>5, so
        # the ceiling must be LOOSER for large-caps and TIGHTER for micro/small (where high P/B = junk/traps).
        # dict keys micro(<$500M)/small(<$2B)/large(>=$2B); large=None means NO cap on big-caps. Paired with
        # remove-ALL-ARK (the user's chosen combo). ──
        import sys
        B = 790.4
        ARK = {"ARKK", "ARKG"}

        def R1(label, **kw):
            r = run(True, True, country_ok=_is_usca, **kw)
            print(f"  {label:42} {r['total']:8.1f}% ({r['total']-B:+7.1f})  Sh{r['sharpe']:.2f}  DD{r['dd']:.0f}%  t{r['t_stat']}", flush=True)

        tiers = [("3/5/6", {"micro": 3, "small": 5, "large": 6}),
                 ("3/5/8", {"micro": 3, "small": 5, "large": 8}),
                 ("3/5/10", {"micro": 3, "small": 5, "large": 10}),
                 ("3/5/15", {"micro": 3, "small": 5, "large": 15}),
                 ("3/5/none (no cap on big-caps)", {"micro": 3, "small": 5, "large": None}),
                 ("4/6/10", {"micro": 4, "small": 6, "large": 10}),
                 ("5/5/none", {"micro": 5, "small": 5, "large": None})]
        print("\n=== CEILING_TEST (no save): cap-tiered P/B ceiling, big-caps loosened (baseline 790.4) ===", flush=True)
        R1("reference baseline")
        R1("reference remove ALL ARK", drop_sectors=ARK)
        R1("reference flat pb<=5", pb_ceiling=5.0)
        print("  -- cap-tiered (micro/small tight, large loose) --", flush=True)
        for lab, cd in tiers:
            R1(f"tiered {lab}", pb_ceiling=cd)
        print("  -- same tiers + remove ALL ARK (your chosen combo) --", flush=True)
        for lab, cd in tiers:
            R1(f"ARKout + tiered {lab}", drop_sectors=ARK, pb_ceiling=cd)
        sys.exit(0)

    if os.environ.get("SELECTION_TEST"):
        # ── user selection rules A/B (no save, no ranking change): ban names whose FIRST trade lost, remove
        # ARK sleeves, and a P/B ceiling (flat + cap-tiered micro/small/large). All are candidate FILTERS on the
        # usca_small flagship — picks/returns of the base arm unchanged; each rule only removes eligible names. ──
        import sys
        B = 790.4

        def R1(label, **kw):
            r = run(True, True, country_ok=_is_usca, **kw)
            print(f"  {label:36} {r['total']:8.1f}% ({r['total']-B:+7.1f} vs base)  Sh{r['sharpe']:.2f}  "
                  f"DD{r['dd']:.0f}%  t{r['t_stat']}  dl{r['delisted_picks']}", flush=True)

        print("\n=== SELECTION_TEST (no save): user rules on the usca_small flagship (baseline 790.4) ===", flush=True)
        R1("baseline (sanity=790.4)")
        R1("ban_first_loss (loss on 1st trade -> banned)", ban_first_loss=True)
        R1("remove ARKK (Ark Innovation)", drop_sectors={"ARKK"})
        R1("remove ALL ARK (ARKK+ARKG)", drop_sectors={"ARKK", "ARKG"})
        print("  -- P/B ceiling, flat --", flush=True)
        for c in (3, 5, 8, 10):
            R1(f"pb <= {c}", pb_ceiling=float(c))
        print("  -- P/B ceiling, cap-tiered (micro<$500M / small<$2B / large>=$2B) --", flush=True)
        R1("tiered hi-micro 8/5/3", pb_ceiling={"micro": 8, "small": 5, "large": 3})
        R1("tiered lo-micro 3/5/8", pb_ceiling={"micro": 3, "small": 5, "large": 8})
        R1("tiered 10/6/4", pb_ceiling={"micro": 10, "small": 6, "large": 4})
        print("  -- stacked (the two winners, NO ban) --", flush=True)
        R1("remove ALL ARK + pb<=5", drop_sectors={"ARKK", "ARKG"}, pb_ceiling=5.0)
        R1("remove ALL ARK + pb<=10", drop_sectors={"ARKK", "ARKG"}, pb_ceiling=10.0)
        R1("remove ALL ARK + tiered lo-micro 3/5/8", drop_sectors={"ARKK", "ARKG"},
           pb_ceiling={"micro": 3, "small": 5, "large": 8})
        R1("remove ALL ARK + pb<=5 + lo-micro floor {micro:3,small:5,large:5}",
           drop_sectors={"ARKK", "ARKG"}, pb_ceiling={"micro": 3, "small": 5, "large": 5})
        print("  -- with ban (ban is the poison) --", flush=True)
        R1("ban + remove ARKK + pb<=5", ban_first_loss=True, drop_sectors={"ARKK"}, pb_ceiling=5.0)
        sys.exit(0)

    if os.environ.get("THEME_TEST"):
        # Quick A/B: run the 4 headline small-cap arms, print, EXIT (no DB save). Pair with DROP_ETFS=... to
        # measure a universe change vs the stored baseline (usca_small 790.4 / norm 1024.0 / pbprof 911.5 / all 723.0).
        import sys
        arms = [("usca_small", dict(country_ok=_is_usca)),
                ("usca_small_norm", dict(country_ok=_is_usca, value_key="pb_roe")),
                ("usca_small_pbprof", dict(country_ok=_is_usca, value_key="pb_prof")),
                ("usca_all", dict(country_ok=_is_usca, _all=True))]
        print(f"\n=== THEME_TEST (no save) DROP_ETFS={sorted(_drop) or 'none'} ===", flush=True)
        base = {"usca_small": 790.4, "usca_small_norm": 1024.0, "usca_small_pbprof": 911.5, "usca_all": 723.0}
        for name, kw in arms:
            small = not kw.pop("_all", False)
            r = run(True, small, **kw)
            b = base[name]
            print(f"  {name:20} {r['total']:8.1f}%  (baseline {b:7.1f}%  {r['total']-b:+7.1f}pp)  "
                  f"Sharpe {r['sharpe']:.2f}  DD {r['dd']:.1f}%  t {r['t_stat']}", flush=True)
        sys.exit(0)

    if os.environ.get("FLAGSHIP_TRACE"):
        import sys
        tr = []
        perf = run(True, True, country_ok=_is_usca, trace=tr)
        out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "arm": "usca_small",
               "perf": {k: perf.get(k) for k in ("total", "annual", "vs_spy", "sharpe", "dd", "t_stat", "months",
                                                 "delisted_picks")},
               "params": {"top_n": TOP_N, "small_cap_max": SMALL, "min_dvol": MIN_DVOL, "conv_weight": CONV},
               "months": tr}
        fp = Path("/app/.data/studies/flagship_history.json")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(out, indent=2, default=str))
        print(f"FLAGSHIP_TRACE written: {fp}  months={len(tr)}  total={perf.get('total')}%  "
              f"(matches stored 790.4% if identical)", flush=True)
        sys.exit(0)

    results = {
        "survivors_only_all": run(False, False),
        "survivors_only_small": run(False, True),
        "with_delisted_all": run(True, False),
        "with_delisted_small": run(True, True),
        "usca_all": run(True, False, country_ok=_is_usca),      # US + Canada only, all-cap
        "usca_small": run(True, True, country_ok=_is_usca),     # US + Canada only, small-cap (FLAGSHIP; skips non-equity sectors)
        # USER-REQUESTED value-trap ceiling arms (2026-08-17): cap-tiered P/B ceiling (micro<=4 / small<=6 /
        # large<=10 so big-caps like GOOG aren't wrongly capped) + remove both ARK sleeves. 4/6/10 tested best
        # of the ceiling shapes; ARKout+4/6/10 = ~1044% in the no-save sweep (treat the +254pp as optimistic —
        # the ARK+ceiling stack is non-monotonic/data-snoopy). Persisted so both the arm and its caveat are visible.
        "usca_small_ceil": run(True, True, country_ok=_is_usca,
                               pb_ceiling={"micro": 4, "small": 6, "large": 10}),
        "usca_small_ceil_noark": run(True, True, country_ok=_is_usca, drop_sectors={"ARKK", "ARKG"},
                                     pb_ceiling={"micro": 4, "small": 6, "large": 10}),
        # ETF-PROXY TEST: same flagship, but when a top-accel sector has NO qualifying value stock (raw
        # commodities like USO/UNG, bonds like TLT/IEF, foreign/index markets) HOLD THE ETF itself — the
        # live is_etf_proxy fallback. Does buying straight commodities/bonds when they accelerate ADD return?
        "usca_small_proxy": run(True, True, country_ok=_is_usca, proxy_etf=True),
        # PREMIUM-NORMALIZED VALUE: rank by P/B÷ROE (cheapness per unit of quality) instead of raw P/B, so
        # high-ROE names (incl. mega-caps like GOOG) can qualify when cheap RELATIVE to the premium they justify.
        "usca_all_norm": run(True, False, country_ok=_is_usca, value_key="pb_roe"),
        "usca_small_norm": run(True, True, country_ok=_is_usca, value_key="pb_roe"),
        # RIGOROUS validation of the crude +233pp: justified P/B and regression-residual, both cap tiers.
        "usca_small_just": run(True, True, country_ok=_is_usca, value_key="justified"),
        "usca_all_just": run(True, False, country_ok=_is_usca, value_key="justified"),
        "usca_small_resid": run(True, True, country_ok=_is_usca, value_key="residual"),
        "usca_all_resid": run(True, False, country_ok=_is_usca, value_key="residual"),
        # MORE premium alternatives
        "usca_small_capm": run(True, True, country_ok=_is_usca, value_key="justified_capm"),
        "usca_all_capm": run(True, False, country_ok=_is_usca, value_key="justified_capm"),
        "usca_small_residrk": run(True, True, country_ok=_is_usca, value_key="resid_rk"),
        "usca_all_residrk": run(True, False, country_ok=_is_usca, value_key="resid_rk"),
        "usca_small_residmf": run(True, True, country_ok=_is_usca, value_key="resid_mf"),
        "usca_all_residmf": run(True, False, country_ok=_is_usca, value_key="resid_mf"),
        "usca_small_roegate": run(True, True, country_ok=_is_usca, value_key="roe_gate"),
        "usca_all_roegate": run(True, False, country_ok=_is_usca, value_key="roe_gate"),
        "usca_small_gpagate": run(True, True, country_ok=_is_usca, value_key="gpa_gate"),
        "usca_all_gpagate": run(True, False, country_ok=_is_usca, value_key="gpa_gate"),
        # CONFOUND: is the P/B÷ROE(=P/E) win the P/E RANKING or just the PROFITABLE-only screen?
        "usca_small_pbprof": run(True, True, country_ok=_is_usca, value_key="pb_prof"),
        # ── P/E-vs-P/B RECONCILIATION MATRIX (why P/E>P/B here but value_ranking_lab found P/B>P/E?) ──
        # 3 metrics x 4 universe cells (survivors-only / +delisted  x  small / all), US+CA. Isolates whether the
        # P/E win needs (a) small-cap, (b) delisted names, or (c) is just the profitable-only screen.
        "usca_all_pbprof": run(True, False, country_ok=_is_usca, value_key="pb_prof"),
        "recon_surv_small_pb": run(False, True, country_ok=_is_usca, value_key="pb"),
        "recon_surv_small_pe": run(False, True, country_ok=_is_usca, value_key="pb_roe"),
        "recon_surv_small_pbp": run(False, True, country_ok=_is_usca, value_key="pb_prof"),
        "recon_surv_all_pb": run(False, False, country_ok=_is_usca, value_key="pb"),
        "recon_surv_all_pe": run(False, False, country_ok=_is_usca, value_key="pb_roe"),
        "recon_surv_all_pbp": run(False, False, country_ok=_is_usca, value_key="pb_prof"),
    }
    # ── TWO-STAGE: top-5 cheapest raw P/B per sector, then a secondary signal picks 1 (small-cap flagship tier) ──
    T5_METHODS = ["cheapest_pb", "pe", "roe", "gpa", "resid", "rsi_os", "rsi_cross", "rsi_bull",
                  "accel_fade", "accel_ride", "mom6_hi", "mom6_lo", "ad_div"]
    for _m in T5_METHODS:
        results[f"t5_{_m}"] = run(True, True, country_ok=_is_usca, top5=_m)
    # CAP-AWARE fallback ordering: when a sector has NO profitable small-cap, take loss-making small (current live
    # rule) vs profitable large-cap vs skip? (loss-makers hurt, so prof-large or skip may beat the loss-small default)
    for _fb in ("loss_small", "prof_any", "skip"):
        results[f"capaware_{_fb}"] = run(True, True, country_ok=_is_usca, capaware=_fb)
    # INFORMATIONAL: what would a price floor COST? (policy is major-exchange gate, NO price floor -> floor_$0 is live)
    price_sweep = {f"floor_${p:g}": run(True, True, min_price=p) for p in (0.0, 1.0, 3.0, 5.0)}
    print(f"\n=== SURVIVORSHIP-FREE SMALL-CAP (GICS engine; USD returns incl. FX P&L; point-in-time FX) ===", flush=True)
    print(f"  {'variant':<22}{'total':>9}{'annual':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}{'dlPicks':>8}", flush=True)
    for k in ("survivors_only_all", "with_delisted_all", "survivors_only_small", "with_delisted_small",
              "usca_all", "usca_small"):
        r = results[k]
        print(f"  {k:<22}{r['total']:>8}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{r['delisted_picks']:>8}", flush=True)

    so = results["survivors_only_small"]["total"]; wd = results["with_delisted_small"]["total"]
    infl = so - wd
    verdict = (
        f"Small-cap pick: survivors-only {so}% vs with-delisted {wd}% -> survivorship inflation "
        f"{infl:+.0f}pp ({results['with_delisted_small']['delisted_picks']} delisted picks taken). "
        + (f"The 463%-class small-cap edge is REAL but inflated by ~{infl:.0f}pp; honest figure ~{wd}%."
           if infl > 20 else
           f"Survivorship inflation is modest (~{infl:.0f}pp) — the small-cap edge largely holds at ~{wd}%.")
        + " (GICS-mapped universe differs from the live ETF-membership universe, so compare within this study.)"
    )
    print("\n" + verdict, flush=True)
    print("\n=== INFO: cost of a price floor (policy = NO floor, keep penny if major-exchange -> floor_$0 is live) ===", flush=True)
    for k, r in price_sweep.items():
        print(f"  {k:<12}{r['total']:>8}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{r['delisted_picks']:>8}", flush=True)

    # ── ETF-PROXY TEST: does holding raw commodity/bond/index sectors (when they accelerate in) ADD return? ──
    base = results["usca_small"]; prox = results["usca_small_proxy"]
    print("\n=== ETF-PROXY TEST: buy the ETF when a top sector has no value stock (commodities/bonds/indices) ===", flush=True)
    print(f"  usca_small        (SKIP non-equity sectors)  total {base['total']:>7}%  Sharpe {base['sharpe']:>5}  DD {base['dd']:>6}%  t {base['t_stat']}", flush=True)
    print(f"  usca_small_proxy  (HOLD the ETF instead)      total {prox['total']:>7}%  Sharpe {prox['sharpe']:>5}  DD {prox['dd']:>6}%  t {prox['t_stat']}", flush=True)
    print(f"  --> proxy delta: {prox['total']-base['total']:+.1f}pp total, {prox['sharpe']-base['sharpe']:+.2f} Sharpe, {prox['dd']-base['dd']:+.1f}pp DD", flush=True)
    print(f"  proxy months held: {prox.get('proxy_months_total')} total  |  commodity {prox['proxy_commodity']['months']} {prox['proxy_commodity']['etfs']}", flush=True)
    print(f"    bond {prox['proxy_bond']['months']} {prox['proxy_bond']['etfs']}", flush=True)
    print(f"    other(index/foreign) {prox['proxy_other']['months']} {prox['proxy_other']['etfs']}", flush=True)
    # ── PREMIUM-NORMALIZED VALUE TEST: does P/B÷ROE (quality-adjusted cheapness) beat raw P/B? ──
    print("\n=== PREMIUM-NORMALIZED VALUE (rank by P/B÷ROE, not raw P/B — lets quality/mega-caps in when cheap-per-quality) ===", flush=True)
    print(f"  {'variant':<20}{'total':>9}{'annual':>8}{'Sharpe':>8}{'DD':>8}{'t':>6}{'mega>50B':>9}", flush=True)
    for k in ("usca_all", "usca_all_norm", "usca_small", "usca_small_norm"):
        r = results[k]
        print(f"  {k:<20}{r['total']:>8}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>7}%{str(r['t_stat']):>6}{r.get('mega_picks',0):>9}", flush=True)
    d_all = results["usca_all_norm"]["total"] - results["usca_all"]["total"]
    d_sm = results["usca_small_norm"]["total"] - results["usca_small"]["total"]
    norm_verdict = (
        f"Premium-normalized P/B÷ROE vs raw P/B: all-cap {d_all:+.1f}pp ({results['usca_all']['total']}%->{results['usca_all_norm']['total']}%, "
        f"mega-cap picks {results['usca_all']['mega_picks']}->{results['usca_all_norm']['mega_picks']}), "
        f"small-cap {d_sm:+.1f}pp ({results['usca_small']['total']}%->{results['usca_small_norm']['total']}%). "
        + ("Quality-adjusting the book multiple HELPS — normalization is return-additive."
           if max(d_all, d_sm) > 0 else
           "Quality-adjusting HURTS — raw cheapest-P/B stays best (the premium is NOT worth paying for; "
           "buying the genuinely-cheapest name beats buying the cheap-relative-to-quality name).")
    )
    print("\n" + norm_verdict, flush=True)

    # ── RIGOROUS validation: do PRINCIPLED premium metrics reproduce the crude +233pp, AND hold in both halves? ──
    def half_split(key):
        mo = results[key].get("monthly") or []
        if not mo:
            return None, None
        mid = len(mo) // 2
        h1 = float(np.prod([1 + v for _, v in mo[:mid]]) - 1) * 100
        h2 = float(np.prod([1 + v for _, v in mo[mid:]]) - 1) * 100
        return round(h1, 1), round(h2, 1)

    METHODS = [("raw P/B", "pb", "usca_small", "usca_all"),
               ("P/B÷ROE (crude)", "pb_roe", "usca_small_norm", "usca_all_norm"),
               ("justified (ROE-g)/(r-g)", "justified", "usca_small_just", "usca_all_just"),
               ("residual P/B~ROE line", "residual", "usca_small_resid", "usca_all_resid")]
    print("\n=== RIGOROUS PREMIUM-NORMALIZATION — principled metrics + split-half robustness (r=9%) ===", flush=True)
    print(f"  {'metric':<26}{'SMALL tot':>10}{'H1':>8}{'H2':>8}{'mega':>6}   {'ALL tot':>9}{'mega':>6}", flush=True)
    sm_raw_h1, sm_raw_h2 = half_split("usca_small")
    for label, _k, sm, al in METHODS:
        h1, h2 = half_split(sm)
        r_sm, r_al = results[sm], results[al]
        print(f"  {label:<26}{r_sm['total']:>9}%{h1:>8}{h2:>8}{r_sm['mega_picks']:>6}   {r_al['total']:>8}%{r_al['mega_picks']:>6}", flush=True)

    def robust(sm_key):
        h1, h2 = half_split(sm_key)
        return (results[sm_key]["total"] > results["usca_small"]["total"]
                and h1 is not None and h1 > sm_raw_h1 and h2 > sm_raw_h2)
    just_ok, resid_ok, crude_ok = robust("usca_small_just"), robust("usca_small_resid"), robust("usca_small_norm")
    robust_verdict = (
        f"SMALL-CAP raw halves: H1 {sm_raw_h1}% / H2 {sm_raw_h2}%. Beats-raw-in-BOTH-halves? "
        f"crude P/B÷ROE={crude_ok}, justified={just_ok}, residual={resid_ok}. "
        + ("A PRINCIPLED metric reproduces the edge in both halves -> the premium-normalization is REAL, not an "
           "artifact of the crude ratio. Candidate to wire (still fee/live-universe caveated)."
           if (just_ok or resid_ok) else
           "NEITHER principled metric beats raw in both halves -> the crude +233pp is likely an ARTIFACT "
           "(ratio noise / a few high-ROE small-cap winners), NOT a robust premium effect. Do NOT wire; "
           "raw cheapest-P/B stays the flagship selector.")
        + " ALL-CAP: every normalization HURTS (waves mega-caps in) -> normalization is small-cap-only at best."
    )
    print("\n" + robust_verdict, flush=True)

    # ── MORE PREMIUM ALTERNATIVES: report + robustness (beat raw in BOTH halves) ──
    ALT = [("justified CAPM (r=rf+βERP)", "usca_small_capm", "usca_all_capm"),
           ("residual RANK (robust)", "usca_small_residrk", "usca_all_residrk"),
           ("residual MULTI (ROE+GP/A+g)", "usca_small_residmf", "usca_all_residmf"),
           ("ROE-gate -> raw P/B", "usca_small_roegate", "usca_all_roegate"),
           ("GP/A-gate -> raw P/B", "usca_small_gpagate", "usca_all_gpagate")]
    print(f"\n=== MORE PREMIUM ALTERNATIVES (raw P/B small {results['usca_small']['total']}% H1 {sm_raw_h1}/H2 {sm_raw_h2}) ===", flush=True)
    print(f"  {'metric':<28}{'SMALL tot':>10}{'H1':>8}{'H2':>8}{'mega':>6}   {'ALL tot':>9}{'robust?':>9}", flush=True)
    alt_winners = []
    for label, sm, al in ALT:
        h1, h2 = half_split(sm); r_sm, r_al = results[sm], results[al]
        ok = (r_sm['total'] > results['usca_small']['total'] and h1 is not None and h1 > sm_raw_h1 and h2 > sm_raw_h2)
        if ok:
            alt_winners.append(label)
        print(f"  {label:<28}{r_sm['total']:>9}%{h1:>8}{h2:>8}{r_sm['mega_picks']:>6}   {r_al['total']:>8}%{('ROBUST' if ok else '-'):>9}", flush=True)
    alt_verdict = (
        f"More premium alternatives vs raw P/B (790.4%, H1 {sm_raw_h1}/H2 {sm_raw_h2}): "
        + (f"ROBUST (beat raw in BOTH halves): {alt_winners}. Best candidate to walk-forward + consider wiring."
           if alt_winners else
           "NONE beats raw cheapest-P/B in both halves. Every principled premium calc (justified/CAPM/residual/"
           "multi-factor) and every quality-gate variant fails to robustly improve the flagship selector. "
           "CONCLUSION: the quality-premium normalization does not add robust return — raw cheapest-P/B is the "
           "selector. The crude P/B÷ROE 1024% remains an ROE-tilt artifact, not reproducible by principled means.")
    )
    print("\n" + alt_verdict, flush=True)

    # ── P/E CONFOUND: crude P/B÷ROE == P/E. Is the win the P/E RANKING or just the PROFITABLE-only screen? ──
    pe_h1, pe_h2 = half_split("usca_small_norm"); pbp_h1, pbp_h2 = half_split("usca_small_pbprof")
    print("\n=== P/E DECOMPOSITION (P/B÷ROE == P/E; isolate ranking vs profitable-only screen) ===", flush=True)
    print(f"  {'variant':<34}{'total':>9}{'DD':>8}{'Shp':>7}{'H1':>8}{'H2':>8}", flush=True)
    print(f"  {'raw P/B (all names)':<34}{results['usca_small']['total']:>8}%{results['usca_small']['dd']:>7}%{results['usca_small']['sharpe']:>7}{sm_raw_h1:>8}{sm_raw_h2:>8}", flush=True)
    print(f"  {'cheapest P/E among profitable':<34}{results['usca_small_norm']['total']:>8}%{results['usca_small_norm']['dd']:>7}%{results['usca_small_norm']['sharpe']:>7}{pe_h1:>8}{pe_h2:>8}", flush=True)
    print(f"  {'cheapest P/B among profitable':<34}{results['usca_small_pbprof']['total']:>8}%{results['usca_small_pbprof']['dd']:>7}%{results['usca_small_pbprof']['sharpe']:>7}{pbp_h1:>8}{pbp_h2:>8}", flush=True)
    pe_verdict = (
        f"P/B÷ROE == P/E (algebraic). cheapest-P/E-among-profitable {results['usca_small_norm']['total']}% vs "
        f"cheapest-P/B-among-profitable {results['usca_small_pbprof']['total']}% vs raw-P/B {results['usca_small']['total']}%. "
        + ("The win is the PROFITABLE-ONLY screen, NOT P/E ranking (pb_prof ~= P/E)."
           if abs(results['usca_small_pbprof']['total'] - results['usca_small_norm']['total']) < 80
           else "The win is P/E RANKING specifically (pb_prof != P/E) — cheapest earnings-multiple genuinely beats "
                "cheapest book-multiple in this small-cap engine, CONTRADICTING the ETF-universe value_ranking (P/B>P/E). "
                "Needs reconciliation before wiring.")
    )
    print("\n" + pe_verdict, flush=True)

    # ── P/E-vs-P/B RECONCILIATION MATRIX ──
    CELLS = [("survivors  small", "recon_surv_small_pb", "recon_surv_small_pe", "recon_surv_small_pbp"),
             ("survivors  all", "recon_surv_all_pb", "recon_surv_all_pe", "recon_surv_all_pbp"),
             ("+delisted  small", "usca_small", "usca_small_norm", "usca_small_pbprof"),
             ("+delisted  all", "usca_all", "usca_all_norm", "usca_all_pbprof")]
    print("\n=== P/E vs P/B RECONCILIATION (why P/E>P/B here but value_ranking_lab found P/B>P/E?) ===", flush=True)
    print(f"  {'universe cell':<18}{'cheapP/B':>10}{'cheapP/E':>10}{'P/B|prof':>10}   {'P/E - P/B':>10}", flush=True)
    recon = {}
    for lab, kpb, kpe, kpbp in CELLS:
        pbt, pet, pbpt = results[kpb]['total'], results[kpe]['total'], results[kpbp]['total']
        recon[lab] = {"pb": pbt, "pe": pet, "pb_prof": pbpt, "pe_minus_pb": round(pet - pbt, 1)}
        print(f"  {lab:<18}{pbt:>9}%{pet:>9}%{pbpt:>9}%   {pet-pbt:>+9.1f}", flush=True)
    sm_surv = recon["survivors  small"]["pe_minus_pb"]; sm_del = recon["+delisted  small"]["pe_minus_pb"]
    al_del = recon["+delisted  all"]["pe_minus_pb"]
    recon_verdict = (
        f"P/E−P/B by cell: surv-small {sm_surv:+.0f}pp, +delisted-small {sm_del:+.0f}pp, +delisted-ALL {al_del:+.0f}pp. "
        + ("P/E beats P/B ONLY in small-cap (all-cap P/E LOSES big) -> the value_ranking_lab contradiction "
           "DISSOLVES (that test was all-cap/ETF, where P/B wins here too). " if al_del < 0 else "")
        + ("P/E's small-cap edge SURVIVES survivors-only (not a delisted artifact). "
           if sm_surv > 20 else
           "P/E's small-cap edge NEEDS the delisted names (survivors-only gap is small) -> it's largely "
           "P/E dodging delisted bankruptcy value-traps that raw-P/B walks into. ")
        + f"Profitable-screen alone (P/B|prof) explains {recon['+delisted  small']['pb_prof']-recon['+delisted  small']['pb']:+.0f}pp "
        f"of the small-cap gap; P/E ranking adds the rest."
    )
    print("\n" + recon_verdict, flush=True)

    # ── TWO-STAGE: top-5 cheapest raw P/B, then a secondary signal picks 1 ──
    ctrl = results["t5_cheapest_pb"]; c_h1, c_h2 = half_split("t5_cheapest_pb")
    print(f"\n=== TWO-STAGE: top-5 cheapest raw P/B per sector -> secondary signal picks 1 (control=cheapest_pb={ctrl['total']}%) ===", flush=True)
    print(f"  {'stage-2 method':<16}{'total':>9}{'DD':>8}{'Shp':>7}{'t':>6}{'H1':>8}{'H2':>8}{'robust?':>9}", flush=True)
    t5_winners = []
    for _m in T5_METHODS:
        r = results[f"t5_{_m}"]; h1, h2 = half_split(f"t5_{_m}")
        ok = (_m != "cheapest_pb" and r['total'] > ctrl['total'] and h1 is not None and h1 > c_h1 and h2 > c_h2)
        if ok:
            t5_winners.append(_m)
        print(f"  {_m:<16}{r['total']:>8}%{r['dd']:>7}%{r['sharpe']:>7}{str(r['t_stat']):>6}{h1:>8}{h2:>8}{('ROBUST' if ok else ('-' if _m!='cheapest_pb' else 'ctrl')):>9}", flush=True)
    t5_verdict = (
        f"Two-stage (top-5 raw P/B -> secondary pick). Control cheapest_pb={ctrl['total']}% (== flagship, sanity check). "
        + (f"Beats control in BOTH halves: {t5_winners}. Best secondary signal to walk-forward + wire."
           if t5_winners else
           "NO secondary signal beats just taking the cheapest of the 5 in both halves -> the cheapest raw P/B IS "
           "the right pick within the value-5; RSI/momentum/quality tie-breakers don't robustly add. Confirms "
           "[[entry-signal-value-pick]]/[[tail-not-average]]: the deepest-value name is the pick, don't second-guess it.")
    )
    print("\n" + t5_verdict, flush=True)

    # ── CAP-AWARE no-profitable-small-cap FALLBACK ordering ──
    print("\n=== CAP-AWARE fallback when a sector has NO profitable small-cap (loss_small=current live rule) ===", flush=True)
    print(f"  {'fallback':<14}{'total':>9}{'DD':>8}{'Shp':>7}{'t':>6}{'H1':>8}{'H2':>8}", flush=True)
    for _fb in ("loss_small", "prof_any", "skip"):
        r = results[f"capaware_{_fb}"]; h1, h2 = half_split(f"capaware_{_fb}")
        print(f"  {_fb:<14}{r['total']:>8}%{r['dd']:>7}%{r['sharpe']:>7}{str(r['t_stat']):>6}{h1:>8}{h2:>8}", flush=True)
    ls, pa, sk = (results[f"capaware_{f}"]['total'] for f in ("loss_small", "prof_any", "skip"))
    best_fb = max((("loss_small", ls), ("prof_any", pa), ("skip", sk)), key=lambda x: x[1])
    capaware_verdict = (
        f"No-profitable-small fallback: loss_small(current live)={ls}%, prof_any(profitable large-cap)={pa}%, "
        f"skip(drop sector)={sk}%. Best = {best_fb[0]} ({best_fb[1]}%). "
        + ("Preferring a PROFITABLE LARGE-CAP over a loss-making small-cap ADDS return -> switch the live fallback."
           if best_fb[0] == "prof_any" else
           "SKIPPING the sector when no profitable small-cap ADDS return -> hold fewer names those months."
           if best_fb[0] == "skip" else
           "The current loss-making-small-cap fallback is already best -> no change; a beaten-down cheap small-cap "
           "in a hot sector still carries the value edge even when unprofitable.")
    )
    print("\n" + capaware_verdict, flush=True)

    proxy_verdict = (
        f"Holding raw commodity/bond/index sectors as ETF proxies {'ADDS' if prox['total']>base['total'] else 'SUBTRACTS'} "
        f"{prox['total']-base['total']:+.1f}pp total return vs the flagship's skip-them rule "
        f"(commodity {prox['proxy_commodity']['months']}mo, bond {prox['proxy_bond']['months']}mo, "
        f"index/foreign {prox['proxy_other']['months']}mo held). "
        + ("Confirms the flagship is right to SKIP them — the live is_etf_proxy fallback should be disabled for these."
           if prox['total'] <= base['total'] else
           "The proxy fallback HELPS — worth keeping/enabling live.")
    )
    print("\n" + proxy_verdict, flush=True)
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "small_cap_max": SMALL, "min_dvol": MIN_DVOL, "min_price": MIN_PRICE,
                       "months": int(results['survivors_only_all']['months']),
                       "n_survivors": len(surv_sector), "n_delisted_mapped": len(delisted_sector),
                       "objective": "MAX TOTAL RETURN, survivorship-free, no-penny"},
            "results": results, "price_sweep": price_sweep, "survivorship_inflation_pp": round(infl, 1), "verdict": verdict,
            "proxy_verdict": proxy_verdict, "norm_verdict": norm_verdict, "robust_verdict": robust_verdict,
            "alt_verdict": alt_verdict, "pe_verdict": pe_verdict, "t5_verdict": t5_verdict,
            "recon_verdict": recon_verdict, "recon_matrix": recon, "capaware_verdict": capaware_verdict,
            "cost_of_equity_r": R_EQ,
            "caveat": "GICS-sector universe (survivors by current ETF membership, delisted by GicSector) -> NOT the "
                      "live ETF-membership universe, so numbers differ from the 463% amplifier run; the SURVIVORS-vs-"
                      "WITH-DELISTED GAP within this study is the survivorship estimate. Delisted names exit when "
                      "candles stop (=delisting, _ret_delist-aware). Only 2705 major-exchange delisted w/ EODHD data "
                      "included (still misses pre-2020 deaths & OTC). PIT, no fees."}


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="survivorship_smallcap", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[survivorship_smallcap]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
