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


def _option_panels(midx, cols):
    """Monthly PIT panels of option-derived signals from core.OptionSnapshot (daily; 2022-09->now): ATM IV,
    IV skew (put-call), put/call OI & volume ratios, dealer GEX. Resampled to month-end (last)."""
    from core.models import OptionSnapshot
    metrics = ["atm_iv", "iv_skew", "pc_oi", "pc_vol", "gex"]
    qs = OptionSnapshot.objects.filter(ticker__in=list(cols)).values_list("ticker", "date", *metrics)
    df = pd.DataFrame.from_records(list(qs), columns=["ticker", "date"] + metrics)
    out = {m: pd.DataFrame(index=midx, columns=cols) for m in metrics}
    if df.empty:
        return out
    df["date"] = pd.to_datetime(df["date"])
    for m in metrics:
        piv = df.pivot_table(index="date", columns="ticker", values=m, aggfunc="last")
        mm = piv.resample("ME").last()
        mm = mm.reindex(mm.index.union(midx)).sort_index().ffill(limit=2).reindex(midx)
        out[m] = mm.reindex(columns=cols)
    return out


def _short_interest_panel(midx, cols, stale_days=45):
    """PIT monthly short-interest (days-to-cover) panel from .data/short_interest.jsonl (Polygon/FINRA bi-monthly,
    dated by settlement_date). Value = latest days_to_cover as of each month-end, if within `stale_days`."""
    import json
    from collections import defaultdict
    p = Path("/app/.data/short_interest.jsonl")
    if not p.exists():
        return pd.DataFrame(index=midx, columns=cols)
    byt = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("days_to_cover") is not None and r.get("settlement_date") and r.get("ticker") in cols:
            byt[r["ticker"]].append((pd.Timestamp(r["settlement_date"]), float(r["days_to_cover"])))
    out = {}
    midx_ser = pd.Series(midx, index=midx)
    for tk, pts in byt.items():
        s = pd.Series({d: v for d, v in pts}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        val = s.reindex(s.index.union(midx)).sort_index().ffill().reindex(midx)
        li = s.index
        last_date = pd.Series([li[li <= d][-1] if len(li[li <= d]) else pd.NaT for d in midx], index=midx)
        age = (midx_ser - last_date).dt.days
        out[tk] = val.where(age <= stale_days)
    return pd.DataFrame(out).reindex(index=midx, columns=cols)


def _analyst_upside_panel(midx, px_panel, stale_days=90):
    """PIT monthly analyst implied-upside panel = (latest price_target within `stale_days` as of month-end) /
    (month-end close) − 1, per ticker. Source: .data/analyst_ratings.jsonl (Benzinga, backfilled). Targets
    older than stale_days are dropped (not 'current'). Ratio cancels currency (both quote-ccy)."""
    import json
    from collections import defaultdict
    p = Path("/app/.data/analyst_ratings.jsonl")
    if not p.exists():
        return pd.DataFrame(index=midx, columns=px_panel.columns)
    byt = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("price_target") and r.get("date") and r.get("ticker") in px_panel.columns:
            byt[r["ticker"]].append((pd.Timestamp(r["date"]), float(r["price_target"])))
    out = {}
    midx_ser = pd.Series(midx, index=midx)
    for tk, pts in byt.items():
        s = pd.Series({d: v for d, v in pts}).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        tgt = s.reindex(s.index.union(midx)).sort_index().ffill().reindex(midx)
        li = s.index
        last_date = pd.Series([li[li <= d][-1] if len(li[li <= d]) else pd.NaT for d in midx], index=midx)
        age = (midx_ser - last_date).dt.days
        out[tk] = tgt.where(age <= stale_days)
    tgt_panel = pd.DataFrame(out).reindex(index=midx, columns=px_panel.columns)
    return (tgt_panel / px_panel.where(px_panel > 0)) - 1

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

# ── SECTOR PLAYBOOK (user: "people don't invest in each sector the same way") — map each sector ETF to the
# way that TYPE of company is really valued. Grounded in valuation common-sense, NOT fit to the data. ──
PLAY_MINERS = {"GLD", "SLV", "PPLT", "USO", "UNG", "URA", "LIT", "COPX", "SLX", "REMX", "XLE", "XLB",
               "AMLP", "WOOD"}                              # asset/reserve-heavy -> cheapest P/B, large-cap OK
PLAY_GROWTH = {"XLK", "SMH", "IGV", "SKYY", "FDN", "BOTZ", "CIBR", "SOCL", "HERO", "PRNT", "UFO", "DRIV",
               "IPO", "FINX", "IBUY", "MAGS", "QQQ", "IWC", "TINY", "ICLN", "TAN", "PAVE", "SRVR"}  # -> momentum leader
PLAY_FIN = {"KRE", "XLF", "IAK", "REM"}                     # banks/insurers -> cheapest P/B among profitable
PLAY_CYCLICAL = {"XLY", "XLI", "XRT", "JETS", "IYT", "XHB", "PEJ", "ITA"}   # cyclicals -> cheapest trailing P/E
PLAY_DEFENSIVE = {"XLP", "XLU", "PBJ", "MOO", "PHO"}        # defensives -> cheapest P/E among profitable (stable)
# everything else (healthcare, biotech, genomics, communications, broad, foreign...) -> the analyst-upside blend
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
    _geno = os.environ.get("GENO_ETF")           # swap the Genomics sleeve's accel-driving ETF (ARKG/GNOM/IDNA);
    if _geno and "Genomics" in etfs:             # holdings are keyed by the sector NAME, so the candidate pool is unchanged
        etfs["Genomics"] = _geno
        print(f"GENO_ETF: Genomics sleeve accel now driven by {_geno}", flush=True)
    # DEACTIVATED sleeves: accel still COMPUTED (kept in the price/accel panel) but never traded — dropped
    # from the pickable ranking below and shown as "⊘ deactivated". Excluded from the candidate/holdings maps.
    _deact = {n: e for n, e in getattr(config, "DEACTIVATED_ETFS", {}).items()
              if e not in CRYPTO and e not in _drop and e not in etfs.values()}
    DEACT_TK = set(_deact.values())
    if DEACT_TK:
        print(f"DEACTIVATED (calculated, not traded): {sorted(DEACT_TK)}", flush=True)
    etf_name = {e: n for n, e in {**etfs, **_deact}.items()}   # ETF ticker -> sector display name (for the trace)
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

    etf_tk = list(etfs.values()) + sorted(DEACT_TK)      # include deactivated so their accel is still computed
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    mom6 = etf_m.pct_change(6)          # 6-month momentum LEVEL (trend), for sector-state scenarios
    mom3 = etf_m.pct_change(3)          # 3-month momentum LEVEL
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    # PIT market-stress regime: SPY drawdown from its trailing-12-month high (past-only). Deep drawdown = stressed
    # -> deep-value/junk names rip on the recovery (raw P/B best, H1/2020); near highs = calm -> quality screen wins.
    spy_dd = (spy_m / spy_m.rolling(12, min_periods=3).max() - 1)
    # SPY 200-DAY MA regime (PIT): bull if month-end SPY >= its trailing 200-trading-day mean, else bear/risk-off.
    _spy_d = etf_daily[BENCH]["Close"]
    _spy200 = _spy_d.rolling(200).mean().resample("ME").last().reindex(midx)
    bull_200 = (spy_m >= _spy200)          # True = above 200d MA (bull); the classic trend filter
    _qqq_d = load_candles(["QQQ"]).get("QQQ")   # for the risk-off QQQ hedge (short growth when value holds up)
    qqq_close_m = (_qqq_d["Close"].resample("ME").last().reindex(midx) if _qqq_d is not None else spy_m)

    # ── REGIME DETECTOR from the rotation system's OWN signal (user: "detect it with the same sector rotation
    # system"): is VALUE / SMALL-CAP leading (favorable — our regime) or is MEGA-CAP GROWTH leading (hostile,
    # like 2017/2018/2023)? Same 6-month momentum the accel engine uses, on the style/size ETFs. PIT. ──
    def _rm(t, w):
        return etf_m[t].pct_change(w) if t in etf_m.columns else pd.Series(np.nan, index=midx)
    # regime-detection SPEED matters: 6mo lags regime turns by months (we eat losses in the wrong regime before
    # it confirms); 1mo reacts fast but whipsaws. Precompute the favorable/hostile signal at several lookbacks.
    regime_fav_by_w, regime_favboth_by_w = {}, {}
    for _w in (1, 2, 3, 6, 12):
        _vg = _rm("VTV", _w) - _rm("VUG", _w)      # value minus growth
        _sl = _rm("IWM", _w) - _rm("IWB", _w)      # small minus large
        regime_fav_by_w[_w] = ((_vg > 0) | (_sl > 0)).reindex(midx)
        regime_favboth_by_w[_w] = ((_vg > 0) & (_sl > 0)).reindex(midx)
    regime_fav = regime_fav_by_w[6]        # default 6mo
    regime_fav_both = regime_favboth_by_w[6]

    universe = sorted(all_holds | set(delisted_sector))
    stock_daily = load_candles(universe)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    smom6 = stock_m.pct_change(6)     # per-stock 6-month price momentum (for the growth-sector 'buy the winner' rule)

    def _pick_mae(tk, d0, d1):
        """Max ADVERSE excursion of a holding during its hold month: the worst intra-window drawdown from
        the buy-date close (how deep it sank before the sell date), as a negative fraction. For the trace's
        'how risky was this month' number — a name can end +44% yet have been −30% mid-hold."""
        df = stock_daily.get(tk)
        if df is None or "Close" not in df:
            return None
        try:
            seg = df["Close"].loc[(df.index > d0) & (df.index <= d1)].dropna()
            base = df["Close"].asof(d0)
            if not (len(seg) and pd.notna(base) and base > 0):
                return None
            return float(seg.min() / base - 1.0)
        except Exception:
            return None
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
    # ANALYST implied-upside panel (PIT): (latest target within 90d) / month-close − 1. For mixing the Benzinga
    # signal into the flagship pick (tie-breaker / gate). Higher = more analyst upside.
    upside_m = _analyst_upside_panel(midx, px).reindex(index=midx, columns=common)
    _up_cov = float(upside_m.notna().mean().mean())
    print(f"analyst implied-upside panel: {upside_m.notna().any().sum()} names ever covered, "
          f"{100*_up_cov:.0f}% cell coverage", flush=True)
    # ── UNTESTED FUNDAMENTAL FACTORS (quality gates for the blend). TTM flows + PIT balance-sheet; all "higher=better". ──
    _ttm_ocf = R(_pit_ttm_panel(reps, "operating_cash_flow", midx))
    _ttm_rd = R(_pit_ttm_panel(reps, "rd_expense", midx))
    _ttm_cogs = R(_pit_ttm_panel(reps, "cost_of_revenue", midx))
    _ca = R(_pit_monthly_panel(reps, "current_assets", midx))
    _cl = R(_pit_monthly_panel(reps, "current_liabilities", midx))
    _inv = R(_pit_monthly_panel(reps, "inventory", midx))
    _cashbs = R(_pit_monthly_panel(reps, "cash_and_equivalents", midx))
    _ta = R(_pit_monthly_panel(reps, "total_assets", midx))
    # trailing realized VOLATILITY of each stock (6-month daily-return std, PIT), for the low/high-vol criterion
    _svol = {}
    for tk in common:
        dd = stock_daily.get(tk)
        if dd is not None and len(dd) > 60:
            _svol[tk] = dd["Close"].pct_change().rolling(126).std().resample("ME").last().reindex(midx)
    stock_vol = pd.DataFrame(_svol).reindex(index=midx, columns=common)
    si_days = _short_interest_panel(midx, common)                      # PIT short interest (days-to-cover)
    print(f"short-interest panel: {si_days.notna().any().sum()} names covered, "
          f"{100*float(si_days.notna().mean().mean()):.0f}% cell coverage", flush=True)
    QFACTORS = {
        "stock_vol": stock_vol,                                        # trailing 6mo realized vol (higher = more volatile)
        "si_days": si_days,                                            # days-to-cover (higher = more shorted)
        "accruals": (_ttm_ocf - ttm_ni) / _ta.where(_ta != 0),          # OCF exceeds earnings = high earnings quality
        "op_margin": ttm_opinc / ttm_rev.where(ttm_rev != 0),
        "asset_turn": ttm_rev / _ta.where(_ta != 0),
        "current_ratio": _ca / _cl.where(_cl != 0),
        "net_cash": (_cashbs - dt) / mktcap.where(mktcap != 0),         # net cash / market cap
        "inv_turn": _ttm_cogs / _inv.where(_inv != 0),
        "fcf_margin": ttm_fcf / ttm_rev.where(ttm_rev != 0),
        "rd_intensity": _ttm_rd / ttm_rev.where(ttm_rev != 0),
    }
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
    ttm_rev_g = ttm_rev / ttm_rev.shift(12) - 1          # YoY TTM revenue growth (hypergrowth LEVEL, RKLB thesis)
    rev_accel = ttm_rev_g - ttm_rev_g.shift(3)           # revenue-growth RE-ACCELERATION (the RKLB tell)
    _REVSEL = {"rev_g": ttm_rev_g, "rev_accel": rev_accel}
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
        if method == "upside":       # highest analyst implied-upside among the 5 cheapest-P/B (covered names)
            return _max(upside_m)
        if method == "upside_lo":    # control: LOWEST analyst upside (should underperform if signal is real)
            return _min(upside_m)
        if method == "upside_gate":  # first (cheapest-P/B) name with >20% analyst upside; else cheapest-P/B
            q = [h for h in pool5 if pd.notna(upside_m.loc[date, h]) and upside_m.loc[date, h] > 0.20]
            return q[0] if q else pool5[0]
        return pool5[0]

    def run(include_delisted, small_only, min_price=MIN_PRICE, country_ok=None, proxy_etf=False, value_key="pb",
            top5=None, capaware=None, trace=None, ban_first_loss=False, pb_ceiling=None, drop_sectors=None,
            exclude_tickers=None, start_date=None, end_date=None, warmup=9, sector_rule=None, quality_gate=None,
            include_months=None, spy200=None, bear_gate=None, hedge=None, growth_etfs=None, adaptive_growth=False,
            growth_fallback=False, top_n=None, size_mode="conv", cost_bps=0.0, lev=1.0, largecap_mode=None,
            defensive_riskoff=None, largecap_keep=None, sector_playbook=False, regime_switch=None,
            regime_lookback=6):
        rets, spies, dl_picks, mrets = [], [], 0, []
        prev_held = set()          # last month's basket, for turnover-based transaction costs
        _base_lcm = largecap_mode  # base large-cap policy; regime_switch overrides it per-month
        proxy_hold = Counter()          # etf -> # months held as a no-value-stock proxy (the live fallback)
        proxy_contrib = 0.0             # sum of proxy monthly contributions to the basket (weighted)
        mega_picks = 0                  # picks with >$50B USD mktcap (does premium-normalization let mega-caps in?)
        traded = set(); banned = set()  # ban_first_loss: names whose FIRST-ever trade lost -> never buy again
        _sd = pd.Timestamp(start_date) if start_date else None
        _ed = pd.Timestamp(end_date) if end_date else None
        for i in range(max(6, warmup), len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            if (_sd is not None and date < _sd) or (_ed is not None and date > _ed):
                continue                # window restriction (apples-to-apples sub-period walk-forward)
            if include_months is not None and date not in include_months:
                continue                # regime-conditional: only accumulate months in this regime
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            a = accel.loc[date]; m6 = mom6.loc[date]; m3 = mom3.loc[date]
            if drop_sectors:
                for e in drop_sectors:
                    if e in a.index:
                        a = a.drop(e); m6 = m6.drop(e, errors="ignore"); m3 = m3.drop(e, errors="ignore")
            # DEACTIVATED sleeves: note any that would have ranked in the top-N (for the blotter), then drop
            # them from the pickable ranking so they never take a slot (results identical to fully-removed).
            _deact_show = []
            if DEACT_TK:
                _aa = a.dropna().sort_values(ascending=False)
                for e in DEACT_TK:
                    if e in _aa.index:
                        rk = _aa.index.get_loc(e) + 1
                        if rk <= TOP_N:                 # would have been in the pickable top-N -> surface it
                            _deact_show.append((e, float(a[e]), int(rk)))
                _dcols = [e for e in DEACT_TK if e in a.index]
                a = a.drop(_dcols, errors="ignore")
                m6 = m6.drop([e for e in _dcols if e in m6.index], errors="ignore")
                m3 = m3.drop([e for e in _dcols if e in m3.index], errors="ignore")
            # SECTOR-STATE scenarios (2-D: momentum LEVEL × ACCELERATION). Default 'accel' = current flagship.
            _sr = sector_rule
            _tn = top_n or TOP_N                       # concentration: override the number of sectors held
            if _sr == "accel" or _sr is None:          # top-N by acceleration (CURRENT)
                top = a.dropna().sort_values(ascending=False).head(_tn).index
            elif _sr == "mom6":                        # top-10 by 6mo momentum LEVEL (trend-following)
                top = m6.dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "up_and_accel":                # up AND accelerating (mom>0 & accel>0), by accel
                top = a[(m6 > 0) & (a > 0)].dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "up_decel":                    # UP but DECELERATING (mom>0 & accel<0), by momentum
                top = m6[(m6 > 0) & (a < 0)].dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "up_any":                      # anything still UP (mom>0), by momentum (ignore accel)
                top = m6[m6 > 0].dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "down_turning":                # DOWN but turning up (mom<0 & accel>0), by accel — early reversal
                top = a[(m6 < 0) & (a > 0)].dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "accel_pos":                   # accel>0 filter then by accel (drops decel Q even if top-10)
                top = a[a > 0].dropna().sort_values(ascending=False).head(TOP_N).index
            elif _sr == "mom_x_accel":                 # rank blend: momentum-rank + accel-rank
                mr = m6.rank(pct=True); ar = a.rank(pct=True)
                top = (mr + ar).dropna().sort_values(ascending=False).head(TOP_N).index
            else:
                top = a.dropna().sort_values(ascending=False).head(TOP_N).index

            # REGIME SWITCH (user): use the rotation system's own value/small-cap-leadership signal to pick the
            # config — AGGRESSIVE (skip large-cap-only, pure small-cap) when our regime is favorable; CORE (keep
            # large-cap) when mega-cap growth leads (2017/2018/2023). largecap_mode is set per-month here.
            largecap_mode = _base_lcm
            if regime_switch:
                _fw = regime_lookback if regime_lookback in regime_fav_by_w else 6
                _sig = regime_favboth_by_w[_fw] if regime_switch == "both" else regime_fav_by_w[_fw]
                _fav = _sig.get(date, True)
                largecap_mode = "skip" if bool(_fav) else None

            # DEFENSIVE ROTATION in risk-off (user): when SPY < 200d MA, don't de-risk to cash — ROTATE into
            # defensive sleeves (Gold miners / Consumer Staples / Utilities / Healthcare) and buy the cheap value
            # name in each. Stays invested, flees to safety, captures flight-to-quality. Only replaces `top`.
            if defensive_riskoff and not bool(bull_200.get(date, True)):
                _def = defensive_riskoff if isinstance(defensive_riskoff, (list, set, tuple)) else \
                    ["GLD", "XLP", "XLU", "XLV", "SCHD"]
                _dtop = [e for e in _def if e in accel.columns]
                if _dtop:
                    top = pd.Index(_dtop)

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
                def _fwd(e):     # the sleeve ETF's OWN return over the hold month (buy date -> sell date)
                    try:
                        return _f(etf_m[e].iloc[i + 1] / etf_m[e].iloc[i] - 1)
                    except Exception:
                        return None
                # FULL ranking (every sleeve, not just top-10) with its accel + own month return, for the blotter
                _all_rank = a.dropna().sort_values(ascending=False)
                _topset = set(top)
                all_sectors = [{"rank": rk, "sector": etf_name.get(e, e), "etf": e,
                                "accel": _f(a[e]), "etf_ret": _fwd(e), "in_top": (e in _topset)}
                               for rk, e in enumerate(_all_rank.index, 1)]
                tr = {"date": str(pd.Timestamp(date).date()), "ndate": str(pd.Timestamp(ndate).date()),
                      "top_sectors": [{"sector": etf_name.get(e, e), "etf": e, "accel": _f(accel.loc[date, e]),
                                       "etf_ret": _fwd(e)} for e in top],
                      "all_sectors": all_sectors,
                      "deactivated": [{"sector": etf_name.get(e, e), "etf": e, "accel": _f(v), "rank": rk,
                                       "etf_ret": _fwd(e)} for e, v, rk in _deact_show],
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
                _lc_only = (not sm) and bool(g0)       # sector offers ONLY large-caps -> the loss-prone fallback
                _lc_mom = False
                # commodity/miner exemption (user): a large-cap PRODUCER (ALB/SQM/MP) is a real cheap producer,
                # not the disrupted-tech value-trap (VSAT/MU) that made skip-large-cap a win. Keep the large-cap
                # in these sectors instead of skipping.
                # sector playbook keeps the large-cap in MINER sectors (real producers) automatically
                _lc_exempt = bool(largecap_keep and etf in largecap_keep) or (sector_playbook and etf in PLAY_MINERS)
                if largecap_mode and _lc_only and not _lc_exempt:
                    # LARGE-CAP FALLBACK FIX (loser analysis: 58% of big losses were cheap-large-cap fallbacks).
                    if largecap_mode == "skip":
                        sm = []; g0 = []               # skip large-cap-only sectors (concentrate elsewhere)
                    elif largecap_mode == "quality":   # require ROE>0 among the large-caps (drop value traps)
                        _q = [x for x in g0 if pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                        g0 = _q or g0
                    elif largecap_mode == "momentum":  # buy the large-cap WINNER, not the cheapest (VSAT/MU trap)
                        _lc_mom = True
                g = (sm or g0) if (small_only or capaware is not None) else g0
                # active quality gate: bear_gate ONLY when SPY<200d MA (switch selection factor in risk-off); else quality_gate
                _ag = quality_gate
                if bear_gate is not None:
                    _ag = bear_gate if not bool(bull_200.get(date, True)) else None
                if _ag is not None and g:               # keep only the top-half by the factor, then pick within
                    _neg = _ag.startswith("-")
                    _qf = QFACTORS.get(_ag.lstrip("-"))
                    if _qf is not None:
                        vals = [(x, _qf.loc[date, x]) for x in g if pd.notna(_qf.loc[date, x])]
                        if len(vals) >= 4:
                            med = float(np.median([v for _, v in vals]))
                            keep = [x for x, v in vals if (v <= med if _neg else v >= med)]
                            if keep:
                                g = keep
                if not g and growth_fallback:
                    # HYPERGROWTH FALLBACK (user): sector would skip (no qualifying value name) -> instead of
                    # skipping, buy the highest-revenue-growth US/CA name in the sleeve (relax cheap/small-cap/
                    # trap; keep price+liquidity). If no growth name either, still skip. Fills EQUITY skips only.
                    _raw = sector_cands(etf, include_delisted)
                    _rel = [h for h in _raw if (country_ok is None or country_ok(h)) and h not in held
                            and _available_at(px_usd[h], date)
                            and pd.notna(as_traded_usd.loc[date, h]) and as_traded_usd.loc[date, h] >= min_price
                            and pd.notna(dvol_usd.loc[date, h]) and dvol_usd.loc[date, h] >= MIN_DVOL
                            and pd.notna(ttm_rev_g.loc[date, h]) and ttm_rev_g.loc[date, h] > 0.20]  # >20% growth
                    if _rel:
                        g = [max(_rel, key=lambda h: ttm_rev_g.loc[date, h])]   # single hypergrowth pick
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
                        # proxy_etf True = hold ANY skipped ETF; or a set of types {"commodity","bond","foreign"}
                        _typ = "commodity" if etf in COMMODITY_ETFS else ("bond" if etf in BOND_ETFS else "foreign")
                        if (proxy_etf is True) or (_typ in proxy_etf):
                            re = etf_m[etf].iloc[i + 1] / etf_m[etf].iloc[i] - 1 if etf in etf_m.columns else np.nan
                            if np.isfinite(re):
                                proxy_hold[etf] += 1; proxy_contrib += float(re)
                                wsum += 1.0; rr += 1.0 * float(re)
                    continue
                # PRINCIPLED value-vs-momentum switch, decided PER SECTOR PER MONTH (point-in-time, no
                # hand-picked sector list): look at the 5 cheapest small-caps in the sleeve. If their cohort is
                # UNPROFITABLE (median trailing ROE < profit_thr) the 'cheap' is a value TRAP (dying disrupted
                # names like VSAT/DDD) -> buy the MOMENTUM leader instead; if the cheap cohort is cash-generative,
                # keep the VALUE pick. growth_etfs = the old hand-picked override (for the A/B); adaptive_growth
                # = the principled version we walk-forward.
                _use_mom = bool(growth_etfs and etf in growth_etfs) or _lc_mom
                if adaptive_growth and g and not _use_mom:
                    _pool5 = sorted(sm, key=lambda h: pb.loc[date, h])[:5] if sm else g[:5]
                    _r = [roe.loc[date, x] for x in _pool5 if pd.notna(roe.loc[date, x])]
                    _thr = 0.0 if adaptive_growth is True else float(adaptive_growth)
                    _use_mom = (len(_r) >= 2 and float(np.median(_r)) < _thr)
                if sector_playbook and g:
                    # SECTOR PLAYBOOK: value the pick the way that TYPE of company is really valued.
                    if etf in PLAY_GROWTH:                 # growth/tech -> momentum leader (cheap P/B = trap)
                        q = [x for x in g if pd.notna(smom6.loc[date, x])]
                        p = max(q, key=lambda h: smom6.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                    elif etf in PLAY_MINERS:               # miners/commodity -> cheapest P/B (book = reserves)
                        p = min(g, key=lambda h: pb.loc[date, h])
                    elif etf in PLAY_FIN:                  # banks/insurers -> cheapest P/B among PROFITABLE
                        q = [x for x in g if pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                        p = min(q or g, key=lambda h: pb.loc[date, h])
                    elif etf in PLAY_CYCLICAL:             # cyclicals -> cheapest trailing P/E (earnings through cycle)
                        q = [x for x in g if pd.notna(pe_ttm_pos.loc[date, x])]
                        p = min(q, key=lambda h: pe_ttm_pos.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                    elif etf in PLAY_DEFENSIVE:            # defensives -> cheapest P/E among profitable (stable earners)
                        q = [x for x in g if pd.notna(pe_ttm_pos.loc[date, x]) and pd.notna(roe.loc[date, x]) and roe.loc[date, x] > 0]
                        p = min(q, key=lambda h: pe_ttm_pos.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                    else:                                  # default (healthcare/biotech/broad/foreign) -> analyst blend
                        _q = [x for x in g if pd.notna(upside_m.loc[date, x])]
                        if len(_q) >= 3:
                            pr = pd.Series({h: pb.loc[date, h] for h in _q}).rank(pct=True)
                            ur = pd.Series({h: upside_m.loc[date, h] for h in _q}).rank(pct=True, ascending=False)
                            p = (0.6 * ur + 0.4 * pr).idxmin()
                        else:
                            p = min(g, key=lambda h: pb.loc[date, h])
                elif _use_mom and g:
                    q = [x for x in g if pd.notna(smom6.loc[date, x])]
                    p = max(q, key=lambda h: smom6.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif capaware is not None:        # LIVE cap-aware rule + configurable no-profitable-small-cap fallback
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
                elif value_key.startswith("revsel:"):   # REVENUE-GROWTH selector (hypergrowth / re-acceleration)
                    _rp = _REVSEL[value_key.split(":")[1]]
                    q = [x for x in g if pd.notna(_rp.loc[date, x])]
                    p = max(q, key=lambda h: _rp.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key == "upside":     # PRIMARY selector: HIGHEST analyst implied-upside % (target/price−1)
                    q = [x for x in g if pd.notna(upside_m.loc[date, x])]
                    p = max(q, key=lambda h: upside_m.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
                elif value_key.startswith("upside_pb"):  # rank blend: w·upside + (1−w)·pb; w from suffix (_60=0.6), default 0.5
                    _w = (float(value_key.split("_")[2]) / 100) if value_key.count("_") >= 2 else 0.5
                    q = [x for x in g if pd.notna(upside_m.loc[date, x])]
                    if len(q) >= 3:
                        pr = pd.Series({h: pb.loc[date, h] for h in q}).rank(pct=True)          # low P/B = low rank = good
                        ur = pd.Series({h: upside_m.loc[date, h] for h in q}).rank(pct=True, ascending=False)  # high upside = low rank
                        p = (_w * ur + (1 - _w) * pr).idxmin()
                    else:
                        p = min(g, key=lambda h: pb.loc[date, h])
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
                elif value_key.startswith("regime"):   # MARKET-CONDITION SWITCH (PIT): stressed -> raw cheapest-P/B
                    # (deep-value/junk rips off the bottom); calm -> cheapest-P/B among PROFITABLE (quality). Threshold
                    # after underscore, e.g. "regime_10" = -10% SPY drawdown from trailing-12m high. Default -10%.
                    _thr = -(float(value_key.split("_")[1]) / 100) if "_" in value_key else -0.10
                    _dd = spy_dd.loc[date]
                    if pd.notna(_dd) and _dd < _thr:          # STRESSED regime -> raw cheapest P/B
                        p = min(g, key=lambda h: pb.loc[date, h])
                    else:                                      # CALM regime -> cheapest P/B among profitable
                        q = [x for x in g if pd.notna(roe_ttm.loc[date, x]) and roe_ttm.loc[date, x] > 0]
                        p = min(q, key=lambda h: pb.loc[date, h]) if q else min(g, key=lambda h: pb.loc[date, h])
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
                if size_mode == "accel":            # bigger bet on the hotter sector (weight ∝ 1+accel)
                    _ac = accel.loc[date, etf] if etf in accel.columns else np.nan
                    if pd.notna(_ac):
                        w *= max(0.3, min(3.0, 1.0 + 1.5 * float(_ac)))
                elif size_mode in ("upside", "upside_steep", "upside_accel"):   # bigger bet on higher analyst upside
                    _up = upside_m.loc[date, p] if p in upside_m.columns else np.nan
                    if pd.notna(_up):
                        _cap = 5.0 if size_mode == "upside_steep" else 3.0
                        _mult = (1.0 + float(_up)) ** (1.5 if size_mode == "upside_steep" else 1.0)
                        w *= max(0.3, min(_cap, _mult))
                    if size_mode == "upside_accel":      # ALSO tilt by sector accel (stack both sizing signals)
                        _ac = accel.loc[date, etf] if etf in accel.columns else np.nan
                        if pd.notna(_ac):
                            w *= max(0.5, min(2.0, 1.0 + float(_ac)))
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
                                        "mae": _pick_mae(p, date, ndate),
                                        "conviction": bool(accumulating(p, date))})
            if wsum <= 0:
                continue
            _mret = rr / wsum
            if not bool(bull_200.get(date, True)):       # SPY below 200d MA -> risk-off overlays
                if spy200 == "cash":
                    _mret = 0.0                          # sit in cash this month
                elif spy200 == "half":
                    _mret *= 0.5                         # de-risk to half exposure
                if hedge is not None:                    # short `hedge` fraction of QQQ (long value / short growth)
                    _qr = qqq_close_m.iloc[i + 1] / qqq_close_m.iloc[i] - 1
                    if np.isfinite(_qr):
                        _mret -= hedge * float(_qr)
            if cost_bps > 0:            # transaction-cost drag: charge cost_bps on the fraction of the basket
                turnover = len(held ^ prev_held) / max(1, len(held))   # that turned over (symmetric diff ≈ two-way)
                _mret -= (cost_bps / 10000.0) * turnover
            prev_held = set(held)
            if lev != 1.0:              # leverage: scale the monthly return (return-additive; -100% floor = ruin month)
                _mret = max(-0.99, lev * _mret)
            rets.append(_mret); spies.append(float(sp)); mrets.append((str(pd.Timestamp(date).date()), float(_mret)))
            if tr is not None:
                # basket worst intra-month drawdown = weighted mean of each holding's max adverse excursion
                _mw = [(pk.get("mae"), pk.get("weight") or 1.0) for pk in tr["picks"] if pk.get("mae") is not None]
                tr["basket_mae"] = (sum(mv * wv for mv, wv in _mw) / sum(wv for _, wv in _mw)) if _mw else None
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

    if os.environ.get("OPTION_TEST"):
        # ── OPTION-DERIVED signals as pick criteria on the blend (data 2022-09+): ATM IV, IV skew, put/call OI &
        # volume, dealer GEX. Each tested HIGH (top-half) and LOW (bottom-half). Split the covered window. ──
        import sys
        QFACTORS.update(_option_panels(midx, common))
        cov = float(QFACTORS["atm_iv"].notna().mean().mean())
        print(f"\n=== OPTION_TEST (no save): option signals as gates on the blend | IV coverage {100*cov:.0f}% ===", flush=True)
        wins = [("2022-09→now", "2022-09-30", None), ("22-24", "2022-09-30", "2024-06-30"), ("24-26", "2024-07-31", None)]
        sigs = [("blend (no gate)", None), ("LOW IV", "-atm_iv"), ("HIGH IV", "atm_iv"),
                ("LOW skew", "-iv_skew"), ("HIGH skew", "iv_skew"), ("LOW pc_oi", "-pc_oi"), ("HIGH pc_oi", "pc_oi"),
                ("LOW pc_vol", "-pc_vol"), ("HIGH pc_vol", "pc_vol"), ("LOW gex", "-gex"), ("HIGH gex", "gex")]
        print(f"  {'gate':18}" + "".join(f"{w[0]:>16}" for w in wins), flush=True)
        base = {}
        for lab, qg in sigs:
            cells = []
            for wl, s, e in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", quality_gate=qg, start_date=s, end_date=e)
                if qg is None:
                    base[wl] = r["total"]
                dtag = f"({r['total']-base[wl]:+.0f})" if qg is not None else ""
                cells.append(f"{r['total']:>5.0f}%{dtag}")
            print(f"  {lab:18}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("HEDGE_TEST"):
        # ── QQQ HEDGE in risk-off months (SPY<200d MA): short `hedge` fraction of QQQ. In a downturn cheap value
        # holds up while growth/QQQ craters, so long-flagship/short-QQQ may capture the spread. Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("no hedge", None), ("short 50% QQQ", 0.5), ("short 100% QQQ", 1.0),
                ("short 50% QQQ + bear-fcf", 0.5)]
        print("\n=== HEDGE_TEST (no save): short QQQ in risk-off months (SPY<200d MA) ===", flush=True)
        print(f"  {'policy':26}" + "".join(f"{w[0]:>16}" for w in wins), flush=True)
        for lab, hf in arms:
            bg = "fcf_margin" if "bear-fcf" in lab else None
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", hedge=hf, bear_gate=bg, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f} {r['dd']:.0f}%")
            print(f"  {lab:26}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("VOL_TEST"):
        # ── the bought STOCK's own realized volatility (trailing 6mo) as a criterion: does LOW-vol or HIGH-vol
        # predict better picks? Gate the blend; also a bear-only low-vol switch. Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("blend (no vol gate)", dict()), ("LOW vol (keep calm)", dict(quality_gate="-stock_vol")),
                ("HIGH vol (keep wild)", dict(quality_gate="stock_vol")),
                ("bear->LOW vol switch", dict(bear_gate="-stock_vol"))]
        print("\n=== VOL_TEST (no save): stock's own trailing volatility as a pick criterion ===", flush=True)
        print(f"  {'policy':24}" + "".join(f"{w[0]:>15}" for w in wins), flush=True)
        for lab, kw in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", start_date=sd, end_date=ed, **kw)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:24}" + "".join(f"{c:>15}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("REGIME_SWITCH"):
        # ── switch the STOCK-SELECTION factor when SPY < 200d MA (risk-off): apply a quality gate ONLY in bear
        # months, keep the pure blend in bull. Tests the flight-to-quality switch WITHOUT going to cash. WF. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("blend (no switch)", None), ("bear->fcf_margin", "fcf_margin"), ("bear->op_margin", "op_margin"),
                ("bear->current_ratio", "current_ratio"), ("bear->net_cash", "net_cash"),
                ("bear->accruals", "accruals"), ("bear->rd_intensity", "rd_intensity")]
        n_below = int((~bull_200.reindex(midx).fillna(True)).sum())
        print(f"\n=== REGIME_SWITCH (no save): switch selection factor when SPY<200d MA ({n_below} bear mo) ===", flush=True)
        print(f"  {'policy':24}" + "".join(f"{w[0]:>15}" for w in wins), flush=True)
        base = {}
        for lab, bg in arms:
            cells = []
            for wl, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", bear_gate=bg, start_date=sd, end_date=ed)
                if bg is None:
                    base[wl] = r["total"]
                dtag = f"({r['total']-base[wl]:+.0f})" if bg is not None else ""
                cells.append(f"{r['total']:>6.0f}%{dtag}")
            print(f"  {lab:24}" + "".join(f"{c:>15}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("SPY200_TEST"):
        # ── SPY 200-day MA regime switch: go to CASH (or HALF exposure) in months where SPY is below its 200d MA.
        # The classic trend filter. But the flagship earns its HIGHEST Sharpe in bear months (buys cheap) — does
        # de-risking help or forfeit that? Walk-forward. ──
        import sys
        n_below = int((~bull_200.reindex(midx).fillna(True)).sum())
        print(f"\n=== SPY200_TEST (no save): de-risk when SPY < 200d MA ({n_below}/{len(midx)} months below) ===", flush=True)
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("always invested", None), ("CASH when SPY<200MA", "cash"), ("HALF when SPY<200MA", "half")]
        print(f"  {'policy':24}" + "".join(f"{w[0]:>16}" for w in wins), flush=True)
        for lab, sw in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", spy200=sw, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f} {r['dd']:.0f}%")
            print(f"  {lab:24}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("REGIME_FACTOR"):
        # ── does a fundamental factor that's bad ON AVERAGE become GOOD in a specific market regime? Test each
        # quality gate WITHIN bull (SPY>10mo MA = risk-on) vs bear (SPY<10mo MA = risk-off) months. ──
        import sys
        spy_ma = spy_m.rolling(10, min_periods=3).mean()
        bull = set(midx[(spy_m >= spy_ma) & spy_ma.notna()])
        bear = set(midx[(spy_m < spy_ma) & spy_ma.notna()])
        gates = [("no gate", None), ("accruals", "accruals"), ("op_margin", "op_margin"), ("asset_turn", "asset_turn"),
                 ("current_ratio", "current_ratio"), ("net_cash", "net_cash"), ("inv_turn", "inv_turn"),
                 ("fcf_margin", "fcf_margin"), ("rd_intensity hi", "rd_intensity"), ("si LOW", "-si_days"), ("si HIGH", "si_days")]
        print(f"\n=== REGIME_FACTOR (no save): quality gates WITHIN market regimes ===", flush=True)
        print(f"regime months: BULL {len(bull)} | BEAR {len(bear)}", flush=True)
        for rlab, rset in [("BULL / risk-on", bull), ("BEAR / risk-off", bear)]:
            print(f"\n  --- {rlab} (n={len(rset)} months; total% compounded over regime months, Δ vs no-gate) ---", flush=True)
            base = None
            for glab, qg in gates:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", quality_gate=qg, include_months=rset)
                if qg is None:
                    base = r["total"]
                tag = f"({r['total']-base:>+6.0f})" if (base is not None and qg is not None) else ""
                print(f"    {glab:18}{r['total']:>8.0f}%  Sh{r['sharpe']:>5.2f}  {tag}", flush=True)
        sys.exit(0)

    if os.environ.get("REGIME_SPEED_TEST"):
        # ── does the regime switch lag? (user: "bad results because we're not catching the regime quickly
        # enough"). Test detection SPEED: 1/2/3/6/12-month lookback on the value/small leadership signal. Fast
        # = less lag but more whipsaw. Honest 2016-2026, focus on the hostile years (2017/2018/2023). ──
        import sys
        from collections import defaultdict
        print("\n=== REGIME_SPEED_TEST (honest 2016-2026): regime-detection lookback speed ===", flush=True)
        print(f"  {'lookback':16}{'FULL':>12}{'DD':>8}{'pre-2020':>10}{'2017':>8}{'2018':>8}{'2023':>8}{'Sharpe':>8}", flush=True)
        for w in (1, 2, 3, 6, 12):
            r = run(True, True, country_ok=_is_usca, regime_switch="either", regime_lookback=w)
            rp = run(True, True, country_ok=_is_usca, regime_switch="either", regime_lookback=w, end_date="2019-12-31")
            mo = dict(r.get("monthly", []))
            yr = defaultdict(lambda: 1.0)
            for d, rr in r.get("monthly", []):
                yr[d[:4]] *= (1 + rr)
            print(f"  {str(w)+'-month':16}{r['total']:>11.0f}%{r['dd']:>7.1f}%{rp['total']:>9.0f}%"
                  f"{(yr['2017']-1)*100:>7.0f}%{(yr['2018']-1)*100:>7.0f}%{(yr['2023']-1)*100:>7.0f}%{r['sharpe']:>8.2f}", flush=True)
        # reference: static core and aggressive on the same year breakdown
        for lab, kw in [("static CORE", dict()), ("static AGGRESSIVE", dict(largecap_mode="skip"))]:
            r = run(True, True, country_ok=_is_usca, **kw)
            yr = defaultdict(lambda: 1.0)
            for d, rr in r.get("monthly", []):
                yr[d[:4]] *= (1 + rr)
            print(f"  {lab:16}{r['total']:>11.0f}%{r['dd']:>7.1f}%{'':>10}{(yr['2017']-1)*100:>7.0f}%{(yr['2018']-1)*100:>7.0f}%{(yr['2023']-1)*100:>7.0f}%{r['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("REGIME_SWITCH_TEST"):
        # ── REGIME SWITCH (user): detect the regime with the rotation system's OWN value/small-cap leadership
        # signal, run AGGRESSIVE (skip large-cap) when favorable, CORE when mega-cap growth leads. Does it
        # capture the aggressive upside AND dodge the 2017/2018/2023 drawdown? Honest 2016-2026, WF. ──
        import sys
        arms = [("static CORE", dict()),
                ("static AGGRESSIVE (skip all)", dict(largecap_mode="skip")),
                ("REGIME-SWITCH (value OR small leads)", dict(regime_switch="either")),
                ("REGIME-SWITCH (value AND small lead)", dict(regime_switch="both"))]
        print("\n=== REGIME_SWITCH_TEST (honest 2016-2026): detect regime from the rotation system, switch config ===", flush=True)
        print(f"  {'config':40}{'FULL':>13}{'DD':>8}{'pre-2020':>11}{'2020-26':>11}{'Sharpe':>8}", flush=True)
        for lab, kw in arms:
            r = run(True, True, country_ok=_is_usca, **kw)
            rp = run(True, True, country_ok=_is_usca, end_date="2019-12-31", **kw)
            rq = run(True, True, country_ok=_is_usca, start_date="2020-01-31", **kw)
            print(f"  {lab:40}{r['total']:>12.0f}%{r['dd']:>7.1f}%{rp['total']:>10.0f}%{rq['total']:>10.0f}%{r['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("CONFIGS_COMPARE"):
        # ── compute the 3 regime-bet configs (agnostic core / middle / aggressive) + save for the doc. ──
        import sys, json as _j
        MINER = {"GLD", "SLV", "PPLT", "USO", "UNG", "URA", "LIT", "COPX", "SLX", "REMX", "XLE", "XLB"}
        cfgs = [
            ("core", "Regime-agnostic core", "Cheapest-P/B small-cap value, top-10, equal-weight. Works in both worlds.",
             dict()),
            ("middle", "Middle — commodity exemption", "Skip large-cap-only sectors EXCEPT commodity/miners (keep the real producer). Robust + lower DD.",
             dict(largecap_mode="skip", largecap_keep=MINER)),
            ("adaptive", "Adaptive — regime switch (12mo)", "Detects value/small-cap leadership from the rotation system's own 12-month momentum (regimes are multi-year, so a slow signal avoids whipsaw); aggressive in our regime, core when mega-cap growth leads. Best risk-adjusted config.",
             dict(regime_switch="either", regime_lookback=12)),
            ("aggressive", "Aggressive — regime bet", "Skip ALL large-cap-only sectors (pure small-cap). Levered long the post-2020 small-cap/commodity regime.",
             dict(largecap_mode="skip")),
        ]
        # shared SPY equity curve (same benchmark for every config)
        _spycurve = []
        _e = 1.0
        for i in range(max(6, 9), len(midx) - 1):
            _sr = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if np.isfinite(_sr):
                _e *= (1 + float(_sr)); _spycurve.append(round(_e * 100000))
        out = []
        for key, name, desc, kw in cfgs:
            rF = run(True, True, country_ok=_is_usca, **kw)
            rp = run(True, True, country_ok=_is_usca, end_date="2019-12-31", **kw)
            rq = run(True, True, country_ok=_is_usca, start_date="2020-01-31", **kw)
            n_yr = rF["months"] / 12.0
            cagr = ((1 + rF["total"] / 100) ** (1 / n_yr) - 1) * 100 if rF["total"] > -100 else None
            # equity curve + calendar from the monthly returns
            mo = rF.get("monthly", [])
            eq = 1.0; curve = []
            cal = {}
            for d, r in mo:
                eq *= (1 + r); curve.append({"d": d, "f": round(eq * 100000)})
                y = d[:4]; cal[y] = (cal.get(y, 1.0)) * (1 + r)
            calendar = [{"year": y, "ret": round((v - 1) * 100, 1)} for y, v in sorted(cal.items())]
            out.append({"key": key, "name": name, "desc": desc, "total": rF["total"], "sharpe": rF["sharpe"],
                        "dd": rF["dd"], "cagr": round(cagr, 1) if cagr else None, "t_stat": rF.get("t_stat"),
                        "pre2020": rp["total"], "post2020": rq["total"], "months": rF["months"],
                        "curve": curve, "calendar": calendar,
                        "final_100k": round(100000 * (1 + rF["total"] / 100))})
            print(f"  {name}: FULL {rF['total']:.0f}% Sh{rF['sharpe']} DD{rF['dd']}% pre{rp['total']:.0f} post{rq['total']:.0f}", flush=True)
        Path("/app/.data/studies/configs_compare.json").write_text(_j.dumps(
            {"configs": out, "spy_curve": _spycurve}, default=str))
        print("saved configs_compare.json", flush=True)
        sys.exit(0)

    if os.environ.get("PLAYBOOK_TEST"):
        # ── SECTOR PLAYBOOK (user: "common-sense rule per sector — people don't invest the same way"):
        # miners->cheap P/B(+large-cap OK), growth->momentum, banks->P/B-if-profitable, cyclicals->P/E,
        # defensives->P/E-if-profitable, else->analyst blend. vs the uniform flagship. Honest 2016-2026, WF. ──
        import sys
        arms = [("uniform blend + skip-largecap (config)", dict(value_key="upside_pb_60", growth_fallback=True, largecap_mode="skip")),
                ("SECTOR PLAYBOOK", dict(sector_playbook=True, growth_fallback=True)),
                ("SECTOR PLAYBOOK + skip-largecap non-miner", dict(sector_playbook=True, growth_fallback=True, largecap_mode="skip"))]
        print("\n=== PLAYBOOK_TEST (honest 2016-2026): per-sector valuation rules vs uniform ===", flush=True)
        print(f"  {'config':44}{'FULL':>13}{'DD':>8}{'pre-2020':>11}{'2020-26':>11}{'Sharpe':>8}", flush=True)
        for lab, kw in arms:
            r = run(True, True, country_ok=_is_usca, **kw)
            rp = run(True, True, country_ok=_is_usca, end_date="2019-12-31", **kw)
            rq = run(True, True, country_ok=_is_usca, start_date="2020-01-31", **kw)
            print(f"  {lab:44}{r['total']:>12.0f}%{r['dd']:>7.1f}%{rp['total']:>10.0f}%{rq['total']:>10.0f}%{r['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("COMMODITY_LC_TEST"):
        # ── COMMODITY/MINER large-cap exemption (user, from Rare-Earth skip): skip large-cap-only sectors
        # EVERYWHERE except commodity/miner sectors, where the large-cap is a real producer (ALB/SQM/MP), not
        # a tech value-trap. Does keeping their large-cap help? Honest 2016-2026. ──
        import sys
        MINER = {"GLD", "SLV", "PPLT", "USO", "UNG", "URA", "LIT", "COPX", "SLX", "REMX", "XLE", "XLB"}
        base = dict(country_ok=_is_usca, largecap_mode="skip")     # config 2
        arms = [("config2 (skip ALL large-cap-only)", dict()),
                ("keep large-cap in commodity/miner", dict(largecap_keep=MINER))]
        print("\n=== COMMODITY_LC_TEST (honest 2016-2026): keep large-cap producers in miner sectors ===", flush=True)
        print(f"  {'policy':38}{'FULL':>13}{'DD':>8}{'pre-2020':>11}{'2020-26':>11}{'Sharpe':>8}", flush=True)
        for lab, kw in arms:
            r = run(True, True, **base, **kw)
            rp = run(True, True, end_date="2019-12-31", **base, **kw)
            rq = run(True, True, start_date="2020-01-31", **base, **kw)
            print(f"  {lab:38}{r['total']:>12.0f}%{r['dd']:>7.1f}%{rp['total']:>10.0f}%{rq['total']:>10.0f}%{r['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("DEFENSIVE_TEST"):
        # ── DEFENSIVE ROTATION in risk-off (user): when SPY<200MA, rotate into Gold/Staples/Utilities/Healthcare
        # instead of de-risking to cash. Test on config 2 (core + skip-large-cap, DD-42%). Does it cut the DD
        # while keeping return? Honest 2016-2026. ──
        import sys
        base = dict(country_ok=_is_usca, largecap_mode="skip")
        arms = [("config2 (baseline)", dict()),
                ("Gold+Staples+Utilities+HC+Div", dict(defensive_riskoff=True)),
                ("Gold + Staples only", dict(defensive_riskoff=["GLD", "XLP"])),
                ("Gold + Utilities + Staples", dict(defensive_riskoff=["GLD", "XLU", "XLP"])),
                ("Gold only (pure haven)", dict(defensive_riskoff=["GLD"]))]
        print("\n=== DEFENSIVE_TEST (honest 2016-2026): rotate to defensives when SPY<200MA (config 2) ===", flush=True)
        print(f"  {'risk-off policy':34}{'FULL':>13}{'DD':>8}{'pre-2020':>11}{'2020-26':>11}{'Sharpe':>8}", flush=True)
        for lab, kw in arms:
            r = run(True, True, **base, **kw)
            rp = run(True, True, end_date="2019-12-31", **base, **kw)
            rq = run(True, True, start_date="2020-01-31", **base, **kw)
            print(f"  {lab:34}{r['total']:>12.0f}%{r['dd']:>7.1f}%{rp['total']:>10.0f}%{rq['total']:>10.0f}%{r['sharpe']:>8.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("DD_TEST"):
        # ── LOWER THE DRAWDOWN on config 2 (core + skip-large-cap = 38294% but DD-42%), user wants to keep the
        # return + cut the DD. Test regime/trend de-riskers that dodge the 2018/2020 crashes. Honest 2016-2026. ──
        import sys
        wins = [("FULL 16-26", None, None), ("pre-2020", None, "2019-12-31"), ("2020-26", "2020-01-31", None)]
        base = dict(country_ok=_is_usca, largecap_mode="skip")     # config 2
        arms = [("config2 (baseline)", dict()),
                ("+ SPY<200MA -> half exposure", dict(spy200="half")),
                ("+ SPY<200MA -> cash", dict(spy200="cash")),
                ("+ SPY<200MA -> FCF-quality gate", dict(bear_gate="fcf_margin")),
                ("+ only accelerating sectors (accel>0)", dict(sector_rule="accel_pos")),
                ("+ up AND accel sectors", dict(sector_rule="up_and_accel")),
                ("+ SPY<200MA short 50% QQQ", dict(hedge=0.5))]
        print("\n=== DD_TEST (honest 2016-2026): lower config-2's -42% DD, keep the return ===", flush=True)
        print(f"  {'overlay':40}{'FULL':>15}{'DD':>8}{'pre-2020':>11}{'2020-26':>11}", flush=True)
        for lab, kw in arms:
            r = run(True, True, start_date=None, end_date=None, **base, **kw)
            rp = run(True, True, end_date="2019-12-31", **base, **kw)
            rq = run(True, True, start_date="2020-01-31", **base, **kw)
            print(f"  {lab:40}{r['total']:>13.0f}% {r['dd']:>6.1f}% {rp['total']:>9.0f}% {rq['total']:>9.0f}%"
                  f"  Sh{r['sharpe']:.2f}", flush=True)
        sys.exit(0)

    if os.environ.get("LEVER_ABLATION"):
        # ── ABLATE every lever on the HONEST 2016-2026 history (user 'test them all'). Cumulative stack:
        # core -> +skip-largecap -> +blend -> +hypergrowth-fill -> +top7 -> +upside-size. Report total/Sharpe/DD
        # + pre-2020 (most OOS-like) vs post-2020, so we see which levers earn their drawdown. ──
        import sys
        wins = [("FULL 16-26", None, None), ("pre-2020", None, "2019-12-31"), ("2020-26", "2020-01-31", None)]
        configs = [
            ("1. core (raw P/B, top10, eq-wt)", dict()),
            ("2. + skip large-cap-only", dict(largecap_mode="skip")),
            ("3. + analyst-upside blend", dict(largecap_mode="skip", value_key="upside_pb_60")),
            ("4. + hypergrowth-fill", dict(largecap_mode="skip", value_key="upside_pb_60", growth_fallback=True)),
            ("5. + top-7 concentration", dict(largecap_mode="skip", value_key="upside_pb_60", growth_fallback=True, top_n=7)),
            ("6. + upside-sizing (FULL)", dict(largecap_mode="skip", value_key="upside_pb_60", growth_fallback=True, top_n=7, size_mode="upside")),
        ]
        print("\n=== LEVER_ABLATION (honest 2016-2026): cumulative stack — total / Sharpe / DD ===", flush=True)
        print(f"  {'config':34}{'FULL':>16}{'pre-2020':>16}{'2020-26':>16}{'DD':>8}", flush=True)
        for lab, kw in configs:
            cells = []; dd = None
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, start_date=sd, end_date=ed, **kw)
                cells.append(f"{r['total']:>8.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    dd = r.get("dd")
            print(f"  {lab:34}" + "".join(f"{c:>16}" for c in cells) + f"{dd:>7.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("LARGECAP_TEST"):
        # ── LARGE-CAP FALLBACK FIX (loser analysis: 58% of big losses were cheap-large-cap fallbacks like
        # MU/VSAT/RIO). When a sector offers ONLY large-caps, test: skip it, quality-gate (ROE>0), or buy the
        # momentum leader instead of the cheapest. On the full flagship. Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        base = dict(country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True, top_n=7, size_mode="upside")
        arms = [("flagship (cheap large-cap)", None), ("SKIP large-cap-only sectors", "skip"),
                ("quality-gate large-caps (ROE>0)", "quality"), ("momentum on large-caps", "momentum")]
        print("\n=== LARGECAP_TEST (no save): fix the loss-prone large-cap fallback ===", flush=True)
        print(f"  {'variant':34}" + "".join(f"{w[0]:>13}" for w in wins) + f"{'DD':>8}", flush=True)
        for lab, lm in arms:
            cells = []; dd = None
            for _, sd, ed in wins:
                r = run(True, True, start_date=sd, end_date=ed, largecap_mode=lm, **base)
                cells.append(f"{r['total']:>6.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    dd = r.get("dd")
            print(f"  {lab:34}" + "".join(f"{c:>13}" for c in cells) + f"{dd:>7.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("ROUND2_TEST"):
        # ── ROUND 2 (user): push the winning conviction-amplification further — steeper/stacked sizing + leverage,
        # on the new flagship (top-7 + upside-size + hypergrowth-fill). Walk-forward + DD. Costs ignored. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        base = dict(country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True, top_n=7)
        arms = [("flagship (size-upside)", dict(size_mode="upside")),
                ("steeper upside (^1.5, cap5)", dict(size_mode="upside_steep")),
                ("upside × accel (stack sizing)", dict(size_mode="upside_accel")),
                ("size-upside + 1.3× lev", dict(size_mode="upside", lev=1.3)),
                ("size-upside + 1.5× lev", dict(size_mode="upside", lev=1.5)),
                ("steeper + 1.3× lev", dict(size_mode="upside_steep", lev=1.3)),
                ("pure upside select (w100)", dict(size_mode="upside", value_key="upside"))]
        print("\n=== ROUND2_TEST (no save): steeper/stacked sizing + leverage on the flagship ===", flush=True)
        print(f"  {'variant':32}" + "".join(f"{w[0]:>13}" for w in wins) + f"{'DD':>8}", flush=True)
        for lab, kw in arms:
            cells = []; dd = None
            merged = {**base, **kw}
            for _, sd, ed in wins:
                r = run(True, True, start_date=sd, end_date=ed, **merged)
                cells.append(f"{r['total']:>6.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    dd = r.get("dd")
            print(f"  {lab:32}" + "".join(f"{c:>13}" for c in cells) + f"{dd:>7.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("STACK_TEST"):
        # ── stack the two IMPROVE winners (top-7 concentration + size-by-analyst-upside), gross AND net of a
        # realistic 25bps round-trip cost. Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        base = dict(country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True)
        arms = [("flagship (top-10)", dict()),
                ("top-7", dict(top_n=7)),
                ("size-upside", dict(size_mode="upside")),
                ("top-7 + size-upside", dict(top_n=7, size_mode="upside")),
                ("top-7 + size-upside + 25bps", dict(top_n=7, size_mode="upside", cost_bps=25)),
                ("flagship + 25bps (net baseline)", dict(cost_bps=25))]
        print("\n=== STACK_TEST (no save): stack the winners, gross + net of 25bps ===", flush=True)
        print(f"  {'variant':34}" + "".join(f"{w[0]:>13}" for w in wins) + f"{'DD':>8}", flush=True)
        for lab, kw in arms:
            cells = []; dd = None
            for _, sd, ed in wins:
                r = run(True, True, start_date=sd, end_date=ed, **base, **kw)
                cells.append(f"{r['total']:>6.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    dd = r.get("dd")
            print(f"  {lab:34}" + "".join(f"{c:>13}" for c in cells) + f"{dd:>7.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("IMPROVE_TEST"):
        # ── BRAINSTORM batch (user 'try all'): concentration, conviction-sizing, transaction costs. All on the
        # current flagship (blend + hypergrowth-fill). Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        base = dict(country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True)
        arms = [("flagship (top-10, conv2x)", dict()),
                ("CONCENTRATION top-5", dict(top_n=5)),
                ("CONCENTRATION top-3", dict(top_n=3)),
                ("CONCENTRATION top-7", dict(top_n=7)),
                ("SIZE by accel (top-10)", dict(size_mode="accel")),
                ("SIZE by upside (top-10)", dict(size_mode="upside")),
                ("top-5 + size by accel", dict(top_n=5, size_mode="accel")),
                ("COST 10bps round-trip", dict(cost_bps=10)),
                ("COST 25bps round-trip", dict(cost_bps=25)),
                ("COST 50bps round-trip", dict(cost_bps=50)),
                ("top-5 + COST 25bps", dict(top_n=5, cost_bps=25))]
        print("\n=== IMPROVE_TEST (no save): concentration / sizing / costs on the flagship ===", flush=True)
        print(f"  {'variant':30}" + "".join(f"{w[0]:>13}" for w in wins) + f"{'DD':>8}", flush=True)
        for lab, kw in arms:
            cells = []; dd = None
            for _, sd, ed in wins:
                r = run(True, True, start_date=sd, end_date=ed, **base, **kw)
                cells.append(f"{r['total']:>6.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    dd = r.get("dd")
            print(f"  {lab:30}" + "".join(f"{c:>13}" for c in cells) + f"{dd:>7.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("GROWTH_FB"):
        # ── HYPERGROWTH FALLBACK (user): when a sector would SKIP, buy its highest-rev-growth name instead of
        # skipping (else still skip). Fills equity skips with growth rather than concentrating in value. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("SKIP (current)", False), ("hypergrowth-fill on skip", True)]
        print("\n=== GROWTH_FB (no save): hypergrowth fallback on would-be-skipped sectors (blend) ===", flush=True)
        print(f"  {'policy':30}" + "".join(f"{w[0]:>14}" for w in wins), flush=True)
        for lab, gf in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=gf,
                        start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:30}" + "".join(f"{c:>14}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("REVGROWTH_HORIZON"):
        # ── Revenue hypergrowth at the RIGHT HORIZON (buy-and-HOLD, not monthly flip). Rank ALL small-caps by
        # TTM revenue growth / re-acceleration; measure fwd 1/3/6/12-month returns of the top quintile vs
        # bottom vs universe. Tests the pure growth FACTOR the way RKLB actually pays (hold the compounder). ──
        import sys
        ttm_rev_g = ttm_rev / ttm_rev.shift(12) - 1
        rev_accel = ttm_rev_g - ttm_rev_g.shift(3)
        smret = stock_m.pct_change()
        def horizon(sig, H):
            fwd = stock_m.pct_change(H).shift(-H)          # forward H-month return
            topq, botq, uni = [], [], []
            for i in range(12, len(midx) - H):
                d = midx[i]
                s = sig.loc[d].dropna()
                # restrict to small-cap, liquid, priced names (the tradeable set)
                s = s[[t for t in s.index if pd.notna(mktcap_usd.loc[d, t]) and mktcap_usd.loc[d, t] < SMALL
                       and pd.notna(dvol_usd.loc[d, t]) and dvol_usd.loc[d, t] >= MIN_DVOL]]
                if len(s) < 20:
                    continue
                n = max(3, len(s) // 5)
                hi = s.sort_values(ascending=False).head(n).index
                lo = s.sort_values(ascending=True).head(n).index
                fv = fwd.loc[d]
                topq.append(fv[hi].dropna().mean()); botq.append(fv[lo].dropna().mean()); uni.append(fv.dropna().mean())
            ann = 12.0 / H
            def cagr(x):
                m = np.nanmean(x)
                return ((1 + m) ** ann - 1) * 100 if pd.notna(m) else float("nan")
            return cagr(topq), cagr(botq), cagr(uni)
        print("\n=== REVGROWTH_HORIZON (no save): fwd-return by hold length, top-quintile revenue growth ===", flush=True)
        print(f"  {'signal / hold':26}{'top-Q ann%':>12}{'bot-Q ann%':>12}{'universe':>10}{'top-bot':>9}", flush=True)
        for lab, sig in [("rev-growth", ttm_rev_g), ("rev-reaccel", rev_accel)]:
            for H in (1, 3, 6, 12):
                t, b, u = horizon(sig, H)
                print(f"  {lab+' '+str(H)+'mo':26}{t:>11.1f}%{b:>11.1f}%{u:>9.1f}%{t-b:>+8.1f}%", flush=True)
        sys.exit(0)

    if os.environ.get("REVGROWTH_TEST"):
        # ── REVENUE HYPERGROWTH as the selector (user, from the RKLB finding): instead of cheapest value,
        # buy the fastest-growing / re-accelerating REVENUE name in each accelerating sector. RKLB thesis =
        # market pays for the revenue trajectory, not earnings. Test level vs acceleration, walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("value blend (baseline)", None), ("hypergrowth: max rev-growth", "rev_g"),
                ("re-accel: max rev-accel", "rev_accel")]
        print("\n=== REVGROWTH_TEST (no save): revenue-growth selector inside accel sectors ===", flush=True)
        print(f"  {'selector':34}" + "".join(f"{w[0]:>14}" for w in wins), flush=True)
        for lab, key in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca,
                        value_key="upside_pb_60" if key is None else ("revsel:" + key),
                        start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:34}" + "".join(f"{c:>14}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("FOREIGN_VALUE"):
        # ── buy the cheap VALUE STOCK inside foreign sleeves (not the ETF): relax the US/CA filter. Do foreign
        # value small-caps bounce like US ones, or does FX/quality drag ruin it? Walk-forward vs US/CA-only. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        _ca = lambda t: _is_usca(t)
        _all = lambda t: True                                    # allow ALL countries' stocks
        arms = [("US/CA only (CURRENT)", _ca), ("ALL countries (foreign value too)", _all)]
        print("\n=== FOREIGN_VALUE (no save): allow foreign value stocks in the pick (blend) ===", flush=True)
        print(f"  {'universe':36}" + "".join(f"{w[0]:>14}" for w in wins), flush=True)
        for lab, ck in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=ck, value_key="upside_pb_60", start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:36}" + "".join(f"{c:>14}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("COUNTRY_TIMING"):
        # ── 'get into the country ETF BEFORE it pops' (user): for FOREIGN country ETFs (no US/CA stock to buy),
        # test ANTICIPATORY signals vs the trailing accel. Accel arrives AFTER the pop; oversold/reversal aim to
        # catch the bottom. Measure the avg NEXT-month return of the top-K countries under each signal. ──
        import sys
        FOREIGN = {"EWC","EWQ","EWI","EWP","EWL","EWN","EWD","EWK","EWO","EIRL","NORW","EDEN","EWH","EWS","EIS",
                   "ENZL","EZU","EWT","EWM","THD","EIDO","EPHE","VNM","EZA","TUR","KSA","EPOL","ECH","EPU","GXG",
                   "ARGT","GREK","QAT","UAE","PAK","EEM","EFA","MCHI","INDA","EWJ","EWY","ILF","EWZ","EWW","VGK",
                   "EWG","EWU","EWA","AFK","ACWI","VEU","AAXJ","EPP","EMXC","FM"}
        fcols = [e for e in etf_m.columns if e in FOREIGN]
        fm = etf_m[fcols]
        r1 = fm.pct_change(1); r3 = fm.pct_change(3); r6 = fm.pct_change(6)
        facc = fm.pct_change(3) - fm.pct_change(3).shift(3)
        fwd = fm.pct_change().shift(-1)
        # 12-month rolling z-score of price (how far below its own recent range) for a mean-reversion signal
        zpx = (fm - fm.rolling(12, min_periods=6).mean()) / fm.rolling(12, min_periods=6).std()
        def topk_fwd(sig, k=5, ascending=False):
            out = []
            for i in range(9, len(fm.index) - 1):
                d = fm.index[i]
                s = sig.loc[d].dropna()
                if len(s) < k:
                    continue
                picks = s.sort_values(ascending=ascending).head(k).index
                fv = fwd.loc[d, picks].dropna()
                if len(fv):
                    out.append(float(fv.mean()))
            return np.mean(out) * 100, len(out)
        sigs = [("accel (trailing, CURRENT)", facc, False), ("6mo-momentum level", r6, False),
                ("OVERSOLD: worst 6mo (rev)", r6, True), ("OVERSOLD: worst 3mo", r3, True),
                ("REVERSAL: down6 & up1 (rank up1)", r1.where(r6 < 0), False),
                ("price z-score LOW (cheap vs range)", zpx, True),
                ("1mo momentum (fast)", r1, False)]
        print("\n=== COUNTRY_TIMING (no save): avg NEXT-month return of top-5 FOREIGN ETFs by signal ===", flush=True)
        print(f"  {'signal':38}{'next-mo avg':>12}{'n':>6}", flush=True)
        for lab, sig, asc in sigs:
            av, n = topk_fwd(sig, 5, asc)
            print(f"  {lab:38}{av:>11.2f}%{n:>6}", flush=True)
        print(f"  {'(all foreign ETFs, equal-weight)':38}{fwd.mean(axis=1).mean()*100:>11.2f}%", flush=True)
        sys.exit(0)

    if os.environ.get("ADAPTIVE_TEST"):
        # ── PRINCIPLED adaptive rule (user): per sector/month, if the cheap cohort is UNPROFITABLE (value trap)
        # buy the MOMENTUM leader, else the VALUE pick. Sector-agnostic + point-in-time. Walk-forward: it must
        # beat pure-value in BOTH halves to be real, not just full-sample (overfit guard). ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("pure value (baseline)", False),
                ("adaptive: momentum if cheap-cohort ROE<0", True),
                ("adaptive: momentum if cheap-cohort ROE<-5%", -0.05),
                ("adaptive: momentum if cheap-cohort ROE<+5%", 0.05)]
        print("\n=== ADAPTIVE_TEST (no save): principled per-sector value/momentum switch (blend) ===", flush=True)
        print(f"  {'rule':44}" + "".join(f"{w[0]:>13}" for w in wins), flush=True)
        for lab, ag in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", adaptive_growth=ag,
                        start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>6.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:44}" + "".join(f"{c:>13}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("SECTOR_RULE"):
        # ── PER-SECTOR value-vs-momentum analysis (user: 'the rule may depend on the sector'). For every
        # month a sector is in the top-10, compute what the VALUE pick (blend) vs the MOMENTUM leader
        # (highest 6mo return) would have returned. Aggregate per sector -> which sectors want which rule. ──
        import sys
        from collections import defaultdict
        agg = defaultdict(lambda: {"n": 0, "val": [], "mom": []})
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            a = accel.loc[date]
            top = a.dropna().sort_values(ascending=False).head(TOP_N).index
            for etf in top:
                pharma = etf in PHARMA_ETFS
                cands = [h for h in sector_cands(etf, True)
                         if _is_usca(h) and _available_at(px_usd[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > MIN_PB
                         and pd.notna(as_traded_usd.loc[date, h]) and as_traded_usd.loc[date, h] >= MIN_PRICE
                         and not bool(trap.loc[date, h]) and pd.notna(dvol_usd.loc[date, h]) and dvol_usd.loc[date, h] >= MIN_DVOL]
                g0 = [x for x in cands if bool(low.loc[date, x])] or cands
                sm = [x for x in g0 if pd.notna(mktcap_usd.loc[date, x]) and mktcap_usd.loc[date, x] < SMALL] or g0
                if not sm:
                    continue
                # VALUE pick = blend (60% upside + 40% cheap P/B), same as flagship
                q = [x for x in sm if pd.notna(upside_m.loc[date, x])]
                if len(q) >= 3:
                    pr = pd.Series({h: pb.loc[date, h] for h in q}).rank(pct=True)
                    ur = pd.Series({h: upside_m.loc[date, h] for h in q}).rank(pct=True, ascending=False)
                    vp = (0.6 * ur + 0.4 * pr).idxmin()
                else:
                    vp = min(sm, key=lambda h: pb.loc[date, h])
                # MOMENTUM pick = highest trailing 6-month return
                qm = [x for x in sm if pd.notna(smom6.loc[date, x])]
                mp = max(qm, key=lambda h: smom6.loc[date, h]) if qm else vp
                vr = _ret_delist(px_usd[vp], date, ndate)
                mr = _ret_delist(px_usd[mp], date, ndate)
                nm = etf_name.get(etf, etf)
                if vr is not None and np.isfinite(vr):
                    agg[nm]["val"].append(float(vr))
                if mr is not None and np.isfinite(mr):
                    agg[nm]["mom"].append(float(mr)); agg[nm]["n"] += 1
        rows = []
        for nm, d in agg.items():
            if len(d["val"]) >= 4:
                va = float(np.mean(d["val"])) * 100; ma = float(np.mean(d["mom"])) * 100
                rows.append((nm, len(d["val"]), va, ma, ma - va))
        rows.sort(key=lambda x: -x[4])
        print("\n=== PER-SECTOR: value-pick vs momentum-pick avg monthly return (n>=4 picks) ===", flush=True)
        print(f"  {'sector':30}{'n':>4}{'VALUE':>9}{'MOMENTUM':>10}{'mom-val':>9}  wants", flush=True)
        for nm, n, va, ma, diff in rows:
            want = "MOMENTUM" if diff > 1.0 else ("value" if diff < -1.0 else "~tie")
            print(f"  {nm:30}{n:>4}{va:>8.1f}%{ma:>9.1f}%{diff:>+8.1f}%  {want}", flush=True)
        sys.exit(0)

    if os.environ.get("GROWTH_TEST"):
        # ── GROWTH-SECTOR RULE (user, from the RKLB finding): in growth/tech/space sleeves, cheapest-P/B buys
        # dying value traps (VSAT/DDD) while the winner (RKLB) is 'expensive'. Buy the MOMENTUM LEADER there
        # instead. Test which growth-sector set helps, walk-forward. ──
        import sys
        GROWTH_ALL = {"UFO", "PRNT", "DRIV", "SMH", "XLK", "IGV", "SKYY", "BOTZ", "FDN", "CIBR",
                      "SOCL", "HERO", "IDNA", "XBI", "FINX", "IBUY", "SRVR"}
        GROWTH_NEG = {"UFO", "PRNT", "DRIV", "XLK", "SMH"}       # only the sleeves that CONTRIBUTED NEGATIVE
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("baseline (value everywhere)", None),
                ("momentum in NEG growth sleeves", GROWTH_NEG),
                ("momentum in ALL growth sleeves", GROWTH_ALL)]
        print("\n=== GROWTH_TEST (no save): momentum-leader instead of cheapest-P/B in growth sleeves (blend) ===", flush=True)
        print(f"  {'rule':34}" + "".join(f"{w[0]:>14}" for w in wins), flush=True)
        for lab, ge in arms:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", growth_etfs=ge,
                        start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:34}" + "".join(f"{c:>14}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("SI_TEST"):
        # ── SHORT INTEREST at purchase (Polygon/FINRA days-to-cover, PIT): does LOW or HIGH short interest
        # among the sector's value candidates predict better picks? Gate the blend, walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        gates = [("blend (no gate)", None), ("LOW short interest", "-si_days"), ("HIGH short interest", "si_days")]
        print("\n=== SI_TEST (no save): short interest at purchase, gate on the blend flagship ===", flush=True)
        print(f"  {'gate':22}" + "".join(f"{w[0]:>16}" for w in wins), flush=True)
        for lab, qg in gates:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", quality_gate=qg, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:22}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("FACTOR_TEST"):
        # ── untested fundamental factors as a QUALITY GATE (keep top-half by the factor) on the blend flagship,
        # walk-forward. accruals/op_margin/asset_turn/current_ratio/net_cash/inv_turn/fcf_margin/rd_intensity. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        gates = [("blend (no gate)", None), ("accruals (earnings qual)", "accruals"), ("op_margin", "op_margin"),
                 ("asset_turn", "asset_turn"), ("current_ratio", "current_ratio"), ("net_cash", "net_cash"),
                 ("inv_turn", "inv_turn"), ("fcf_margin", "fcf_margin"), ("rd_intensity hi", "rd_intensity"),
                 ("rd_intensity lo", "-rd_intensity")]
        base = None
        print("\n=== FACTOR_TEST (no save): fundamental quality GATE on the blend flagship (total% / vs base) ===", flush=True)
        print(f"  {'gate (keep top-half)':26}" + "".join(f"{w[0]:>16}" for w in wins), flush=True)
        for lab, qg in gates:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", quality_gate=qg, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
                if qg is None and sd is None:
                    base = r["total"]
            print(f"  {lab:26}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("PROXY_TEST"):
        # ── instead of SKIPPING a commodity/bond/country sleeve that accelerated into the top-10, HOLD its ETF
        # (Nat-Gas futures ETF, Treasury ETF, country ETF). Test each type separately on the blend flagship. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        arms = [("SKIP (current)", False), ("proxy COMMODITY", {"commodity"}), ("proxy BOND", {"bond"}),
                ("proxy COUNTRY/foreign", {"foreign"}), ("proxy ALL", True)]
        print("\n=== PROXY_TEST (no save): hold the ETF for skipped commodity/bond/country sleeves (blend flagship) ===", flush=True)
        print(f"  {'policy':24}" + "".join(f"{w[0]:>14}" for w in wins) + f"{'proxy-mo':>10}", flush=True)
        for lab, pe in arms:
            cells = []
            pm = None
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", proxy_etf=pe, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
                if sd is None and ed is None:
                    pm = r.get("proxy_months_total")
            print(f"  {lab:24}" + "".join(f"{c:>14}" for c in cells) + f"{(pm if pm is not None else ''):>10}", flush=True)
        sys.exit(0)

    if os.environ.get("GENO_TEST"):
        # ── run the flagship BLEND under the current GENO_ETF (Genomics accel driven by ARKG/GNOM/IDNA), across
        # windows. Run the process 3× with GENO_ETF=ARKG/GNOM/IDNA to A/B the genomics-sleeve replacement. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        print(f"\n=== GENO_TEST: flagship blend with Genomics accel = {_geno or 'ARKG (default)'} ===", flush=True)
        cells = []
        for lab, sd, ed in wins:
            r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", start_date=sd, end_date=ed)
            cells.append(f"{lab} {r['total']:.0f}% Sh{r['sharpe']:.2f}")
        print("  " + "  |  ".join(cells), flush=True)
        sys.exit(0)

    if os.environ.get("SECTOR_TEST"):
        # ── SECTOR-STATE scenarios: rank/filter sectors by momentum LEVEL × ACCELERATION, not just top-accel.
        # "what if it's not accelerating anymore but still up?" etc. Flagship blend pick within each. Walk-forward. ──
        import sys
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        rules = [("accel (CURRENT)", "accel"), ("mom6 level (trend)", "mom6"),
                 ("up_and_accel", "up_and_accel"), ("up_decel (up,not accel)", "up_decel"),
                 ("up_any (still up)", "up_any"), ("down_turning (early rev)", "down_turning"),
                 ("accel_pos (accel>0)", "accel_pos"), ("mom_x_accel blend", "mom_x_accel")]
        print("\n=== SECTOR_TEST (no save): sector-selection scenarios × flagship blend (total% / Sharpe) ===", flush=True)
        print(f"  {'rule':26}" + "".join(f"{w[0]:>14}" for w in wins), flush=True)
        for lab, sr in rules:
            cells = []
            for _, sd, ed in wins:
                r = run(True, True, country_ok=_is_usca, value_key="upside_pb_60", sector_rule=sr,
                        start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>7.0f}% {r['sharpe']:.2f}")
            print(f"  {lab:26}" + "".join(f"{c:>14}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("UPSIDE_WF"):
        # ── (a) blend-WEIGHT sensitivity + (b) per-YEAR walk-forward for the cheap-P/B × analyst-upside blend. ──
        import sys
        # (a) weight sweep across windows
        wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
                ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
        weights = [("pb (w0)", "pb"), ("w30", "upside_pb_30"), ("w40", "upside_pb_40"),
                   ("w50", "upside_pb_50"), ("w60", "upside_pb_60"), ("w70", "upside_pb_70"), ("upside(w100)", "upside")]
        print("\n=== (a) UPSIDE×P/B BLEND-WEIGHT sensitivity (total% vs-SPY-agnostic, higher=better) ===", flush=True)
        print(f"  {'window':10}" + "".join(f"{w[0]:>14}" for w in weights), flush=True)
        for lab, sd, ed in wins:
            cells = []
            for _, vk in weights:
                r = run(True, True, country_ok=_is_usca, value_key=vk, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>10.0f}%")
            print(f"  {lab:10}" + "".join(f"{c:>14}" for c in cells), flush=True)
        # (b) per-calendar-year walk-forward: does w50 beat raw pb each year? (2020 = bottom-entry, flag it)
        print("\n=== (b) PER-YEAR walk-forward: pb vs upside_pb_50 (total% within year) ===", flush=True)
        print(f"  {'year':8}{'pb':>12}{'upside_pb_50':>16}{'edge(pp)':>10}", flush=True)
        yrs = [("2020*", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
               ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
               ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
               ("2026", "2026-01-01", "2026-12-31")]
        wins_yr = 0; tot_yr = 0
        for y, sd, ed in yrs:
            a = run(True, True, country_ok=_is_usca, value_key="pb", start_date=sd, end_date=ed)
            b = run(True, True, country_ok=_is_usca, value_key="upside_pb_50", start_date=sd, end_date=ed)
            edge = b["total"] - a["total"]
            if "*" not in y:
                tot_yr += 1; wins_yr += 1 if edge > 0 else 0
            print(f"  {y:8}{a['total']:>11.1f}%{b['total']:>15.1f}%{edge:>+10.1f}", flush=True)
        print(f"\n  blend beats raw-P/B in {wins_yr}/{tot_yr} clean years (2020 excluded = bottom-entry artifact)", flush=True)
        sys.exit(0)

    if os.environ.get("UPSIDE_PRIMARY"):
        # ── analyst implied-upside % (target/price−1) as the PRIMARY selection criterion (pick highest-upside
        # name in each sector, ignoring P/B) vs the flagship (cheapest-P/B) vs a 50/50 rank blend. Walk-forward. ──
        import sys
        wins = [("FULL 2019→now", None, None), ("ex-2020 (2021→now)", "2021-01-31", None),
                ("H1 2019→2022", None, "2022-12-31"), ("H2 2023→2026", "2023-01-31", None)]
        arms = [("pb (flagship)", "pb"), ("upside (primary)", "upside"), ("upside_pb (50/50 blend)", "upside_pb")]
        print("\n=== UPSIDE_PRIMARY (no save): analyst-upside % as the SELECTOR, walk-forward (total% / Sh / t) ===", flush=True)
        print(f"  {'window':22}" + "".join(f"{a[0]:>26}" for a in arms), flush=True)
        for lab, sd, ed in wins:
            cells = []
            for _, vk in arms:
                r = run(True, True, country_ok=_is_usca, value_key=vk, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>9.0f}% Sh{r['sharpe']:.2f} t{r['t_stat']}")
            print(f"  {lab:22}" + "".join(f"{c:>26}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("ANALYST_MIX"):
        # ── mix the Benzinga analyst implied-upside into the flagship pick: among the 5 cheapest-P/B small-caps,
        # pick highest analyst-upside (tie-breaker) / lowest (control) / gate >20%. Walk-forward vs the flagship
        # (cheapest_pb) — only real if it beats baseline AND the upside_lo control underperforms, across windows. ──
        import sys
        wins = [("FULL 2019→now", None, None), ("ex-2020 (2021→now)", "2021-01-31", None),
                ("H1 2019→2022", None, "2022-12-31"), ("H2 2023→2026", "2023-01-31", None)]
        arms = [("cheapest_pb (flagship)", "cheapest_pb"), ("+upside (tie-break)", "upside"),
                ("+upside_gate>20%", "upside_gate"), ("upside_lo (control)", "upside_lo")]
        print("\n=== ANALYST_MIX (no save): implied-upside as tie-breaker on the 5 cheapest-P/B, walk-forward ===", flush=True)
        print(f"  {'window':22}" + "".join(f"{a[0]:>24}" for a in arms), flush=True)
        for lab, sd, ed in wins:
            cells = []
            for _, mth in arms:
                r = run(True, True, country_ok=_is_usca, top5=mth, start_date=sd, end_date=ed)
                cells.append(f"{r['total']:>10.0f}% t{r['t_stat']}")
            print(f"  {lab:22}" + "".join(f"{c:>24}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("REGIME_TEST"):
        # ── MARKET-CONDITION SWITCH walk-forward: does "raw-P/B when SPY is in a drawdown, profitability-screen
        # when calm" beat BOTH static selectors across windows? Only adopt if it wins ex-2020 AND in both halves
        # (else it's just re-fitting the H1 crash). ──
        import sys
        wins = [("FULL 2019→now", None, None), ("ex-2020 (2021→now)", "2021-01-31", None),
                ("H1 2019→2022", None, "2022-12-31"), ("H2 2023→2026", "2023-01-31", None)]
        arms = [("pb", dict(value_key="pb")), ("pb_prof", dict(value_key="pb_prof")),
                ("regime_5", dict(value_key="regime_5")), ("regime_10", dict(value_key="regime_10")),
                ("regime_15", dict(value_key="regime_15"))]
        print("\n=== REGIME_TEST (no save): SPY-drawdown selector switch, walk-forward (total% / t) ===", flush=True)
        print(f"  {'window':22}" + "".join(f"{a[0]:>16}" for a in arms), flush=True)
        for lab, sd, ed in wins:
            cells = []
            for _, kw in arms:
                r = run(True, True, country_ok=_is_usca, start_date=sd, end_date=ed, **kw)
                cells.append(f"{r['total']:>8.0f}% t{r['t_stat']}")
            print(f"  {lab:22}" + "".join(f"{c:>16}" for c in cells), flush=True)
        sys.exit(0)

    if os.environ.get("CRASH_TEST"):
        # ── how does the flagship do THROUGH the Feb->Mar 2020 crash? warmup=9 (default) enters end-March = the
        # bottom (captures only recovery); warmup=6 starts trading 2019-12 so we HOLD a position INTO the crash
        # (the 2020-02-29 row = return end-Feb->end-Mar = the crash month). ──
        import sys
        for wu in (9, 6):
            tr = []
            r = run(True, True, country_ok=_is_usca, warmup=wu, trace=tr)
            print(f"\n=== CRASH_TEST warmup={wu}: first trade {tr[0]['date']} | full-window total {r['total']}% | DD {r['dd']}% ===", flush=True)
            print(f"  {'month(buy)':12}{'basket→next':>13}{'SPY→next':>11}", flush=True)
            for t in tr:
                if t["date"][:7] in ("2019-12", "2020-01", "2020-02", "2020-03", "2020-04", "2020-05"):
                    mark = "  <-- CRASH (end-Feb→end-Mar)" if t["date"][:7] == "2020-02" else ""
                    print(f"  {t['date']:12}{t['basket_ret']*100:>+12.1f}%{t['spy_ret']*100:>+10.1f}%{mark}", flush=True)
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
        # FLAGSHIP is now the CHEAP-P/B × ANALYST-UPSIDE blend (w60) — the one additive lever that beat raw-P/B
        # in both halves + across weights + 4/6 years (2026-08-17, user). Raw-P/B remains as usca_small reference.
        tr = []
        # CONFIG-parameterized trace so each setup gets its own full doc (subtabs -> per-config trades).
        # FLAGSHIP default = ADAPTIVE: raw-value core + 12-month regime switch. Best risk-adjusted (28447%).
        _cfgkw = {
            "adaptive": dict(regime_switch="either", regime_lookback=12),
            "core": dict(),
            "middle": dict(largecap_mode="skip", largecap_keep={"GLD", "SLV", "PPLT", "USO", "UNG", "URA", "LIT",
                                                                 "COPX", "SLX", "REMX", "XLE", "XLB"}),
            "aggressive": dict(largecap_mode="skip"),
        }
        _ck = os.environ.get("CONFIG", "adaptive")
        _kw = _cfgkw.get(_ck, _cfgkw["adaptive"])
        perf = run(True, True, country_ok=_is_usca, trace=tr, **_kw)
        out = {"computed_at": pd.Timestamp.utcnow().isoformat(), "arm": f"usca_small_{_ck}", "config": _ck,
               "perf": {k: perf.get(k) for k in ("total", "annual", "vs_spy", "sharpe", "dd", "t_stat", "months",
                                                 "delisted_picks")},
               "params": {"top_n": TOP_N, "small_cap_max": SMALL, "min_dvol": MIN_DVOL, "conv_weight": CONV,
                          "config": _ck, "selector": "cheapest-P/B value in accelerating sectors; " + _ck + " overlay"},
               "months": tr}
        suffix = "" if _ck == "adaptive" else f"_{_ck}"
        fp = Path(f"/app/.data/studies/flagship_history{suffix}.json")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(out, indent=2, default=str))
        print(f"FLAGSHIP_TRACE[{_ck}] written: {fp}  months={len(tr)}  total={perf.get('total')}%", flush=True)
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
        # ⭐ CHEAP-P/B × ANALYST-UPSIDE rank blend (60% analyst implied-upside + 40% cheap-P/B). The one additive
        # lever that beat raw P/B in BOTH halves + across weights 30-70% + 4/6 clean years (2026-08-17). Uses the
        # Benzinga implied-upside panel (67% coverage; falls back to cheapest-P/B when no recent target). Caveats:
        # 2 down years (2021/2024), partial coverage, survivorship/window-inflated absolutes, needs true OOS.
        # ⭐⭐ FLAGSHIP (2026-08-18): ADAPTIVE — raw-value core + 12-month REGIME SWITCH (aggressive skip-large-cap
        # when value/small-cap leads, core when mega-cap growth leads). Best risk-adjusted config: 28447% Sh1.61
        # DD−24.9% (core-level DD, 2.4× the core return), +72% pre-2020. Detects regime from the rotation
        # system's own 12mo value/small leadership signal (slow = matches the multi-year regime, no whipsaw).
        "usca_small_adaptive": run(True, True, country_ok=_is_usca, regime_switch="either", regime_lookback=12),
        # the demoted aggressive stack (kept for reference; overfit the 2020 recovery, DD−42%)
        "usca_small_upside_pb": run(True, True, country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True,
                                    top_n=7, size_mode="upside", largecap_mode="skip"),
        # references to keep the levers' lift visible
        "usca_small_blend_top10": run(True, True, country_ok=_is_usca, value_key="upside_pb_60", growth_fallback=True),
        "usca_small_blend_nofill": run(True, True, country_ok=_is_usca, value_key="upside_pb_60"),
        # ⭐ REGIME SWITCH (2026-08-17, user): the blend, but when SPY < its 200-day MA (risk-off) tilt the pick
        # toward cash-generative names (FCF-margin quality gate); pure blend when SPY above 200d MA. +287pp full,
        # positive in ALL windows (only applies the gate where flight-to-quality helps). Modest, owes OOS.
        "usca_small_bear_fcf": run(True, True, country_ok=_is_usca, value_key="upside_pb_60", bear_gate="fcf_margin"),
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
