#!/usr/bin/env python
"""Faithful port of the user's Pine v6 "DD Downside + Sortino + OHLC4 RSI" indicator (pine.txt) as a
UNIVERSE SCANNER + a CRITERIA-ATTRIBUTION backtest.

Three jobs:
  1. Port the exact Pine BUY/SELL state machine (own-return Sortino-RSI cross of its EMA within a
     confirmation window; DD Downside Score > 0; Ulcer Index > minimum; OHLC4-RSI used only for the
     SELL confirmation; hard DD>=3 override).
  2. For EVERY historical BUY across the universe, record the entry-time value of each gate/criterion
     + forward returns (5/10/21/42/63d) + the realized round-trip return under the Pine exit. Then bin
     each criterion and report forward return / win rate per bin (segmented by cap x sector) so we can
     see WHICH criteria find the best trades and re-tune the gates.
  3. Emit the LIVE scanner: names firing a BUY (or armed and waiting) on the most recent bar.

Follows pine.txt EXACTLY (own-return Sortino, NO SPY ref; DD+Ulcer entry gates; OHLC4 RSI for exit).

Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sortino_dd_scanner.py
Env: SORT_WIN(14) SORT_SMOOTH(14) SORT_RSI(14) SORT_EMA(14) RF(0.02) CONFIRM_WIN(10)
     OHLC_RSI(14) OHLC_EMA(14) DD_RET(14) DD_DOWN(14) DD_DD(60) DD_W(0.10) DD_SMOOTH(3) DD_HARD(3.0)
     ULCER_LEN(14) ULCER_PEAK(14) ULCER_MIN(10.0) MINVOL(5e6) START_DATE(2016-01-01)
     TOP_MCAP(0=all) SECTOR_FILTER("") HORIZONS(5,10,21,42,63)
"""
import os, json, pickle
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

# --- Pine inputs (defaults = pine.txt) ---
SORT_WIN    = int(os.environ.get("SORT_WIN", 14))       # Rolling Sortino window
SORT_SMOOTH = int(os.environ.get("SORT_SMOOTH", 14))    # SMA smoothing of the sortino
SORT_RSI    = int(os.environ.get("SORT_RSI", 14))       # RSI length of the smoothed sortino
SORT_EMA    = int(os.environ.get("SORT_EMA", 14))       # EMA length of the sortino-RSI (the cross line)
RF          = float(os.environ.get("RF", 0.02))         # annual risk-free rate
CONFIRM_WIN = int(os.environ.get("CONFIRM_WIN", 10))    # bars allowed after the bull sortino cross
OHLC_RSI    = int(os.environ.get("OHLC_RSI", 14))       # OHLC4 RSI length (SELL confirm)
OHLC_EMA    = int(os.environ.get("OHLC_EMA", 14))       # EMA of the OHLC4 RSI
DD_RET      = int(os.environ.get("DD_RET", 14))         # DD score: return EMA length
DD_DOWN     = int(os.environ.get("DD_DOWN", 14))        # DD score: downside risk EMA length
DD_DD       = int(os.environ.get("DD_DD", 60))          # DD score: drawdown lookback
DD_W        = float(os.environ.get("DD_W", 0.10))       # DD score: drawdown penalty weight
DD_SMOOTH   = int(os.environ.get("DD_SMOOTH", 3))       # DD score: final EMA smoothing
DD_HARD     = float(os.environ.get("DD_HARD", 3.0))     # hard-sell DD score
ULCER_LEN   = int(os.environ.get("ULCER_LEN", 14))      # ulcer averaging length
ULCER_PEAK  = int(os.environ.get("ULCER_PEAK", 14))     # ulcer peak lookback
ULCER_MIN   = float(os.environ.get("ULCER_MIN", 0.0))   # USER 2026-09-25: REMOVE ulcer -> 0 disables the Pine ulcer entry gate

MINVOL      = float(os.environ.get("MINVOL", 5e6))      # 63d median $ volume floor (scanner realism)
MCAP_FLOOR  = float(os.environ.get("MCAP_FLOOR", 1e9))  # HARD PIT market-cap floor at entry (user: focus on $1B+ = tradeable)
MKT_GATE    = int(os.environ.get("MKT_GATE", 1))        # USER 2026-09-25: only BUY when SPY's RSI is ABOVE its own moving average
MKT_RSI_LEN = int(os.environ.get("MKT_RSI_LEN", 14))    # SPY RSI length
MKT_MA_LEN  = int(os.environ.get("MKT_MA_LEN", 14))     # length of the RSI's moving average
MKT_MA_TYPE = os.environ.get("MKT_MA_TYPE", "sma")      # sma | ema
START_DATE  = os.environ.get("START_DATE", "2016-01-01")
TOP_MCAP    = int(os.environ.get("TOP_MCAP", 0))        # 0 = whole universe; else PIT top-N by mcap
SECTOR_FILTER = os.environ.get("SECTOR_FILTER", "")     # e.g. "XLK,XLC" to restrict
HORIZONS    = [int(x) for x in os.environ.get("HORIZONS", "5,10,21,42,63").split(",")]
PANELS      = os.environ.get("PANELS", "/app/.data/panels.pkl")
OUT_DIR     = "/app/.data/studies"

# --- TUNED scanner = TAIL-TARGETING (user 2026-09-24: "keep the ulcer and find a way to target the big trades").
#     The big trades (round-trip >= +50%) concentrate MONOTONICALLY in HIGH-ULCER x SMALLER-CAP names: ulcer>=30
#     turns 1-in-4 into a +50% trade; mega-cap has NO tails (AMZN max ulcer ever = 11). So: ULCER FLOOR + drop
#     mega. DD ceiling only dodges the dd>=3 one-bar traps (within high-ulcer it barely filters). ---
SCAN_ULCER_MIN = float(os.environ.get("SCAN_ULCER_MIN", 5.0))    # USER 2026-09-25: try ulcer>5 (milder than the 14 tail lever)
SCAN_DD_CEIL   = float(os.environ.get("SCAN_DD_CEIL", 2.5))      # only dodge the near-euphoric dd>=3 one-bar traps (keeps 100% of the tail)
SCAN_MAXBAC    = int(os.environ.get("SCAN_MAXBAC", 0))           # 0 = off (bars_after_cross is NOT a tail lever)
SCAN_EX_MICRO  = int(os.environ.get("SCAN_EX_MICRO", 0))         # 0 = off (MCAP_FLOOR already sets the bottom = $1B)
SCAN_EX_MEGA   = int(os.environ.get("SCAN_EX_MEGA", 0))          # 0 = include mega ("$1B and up" per user); set 1 to drop mega (no tails)
SCAN_SRSI_MAX  = float(os.environ.get("SCAN_SRSI_MAX", 0))       # 0 = off (not a tail lever)
SCAN_NRSI_MAX  = float(os.environ.get("SCAN_NRSI_MAX", 0))       # 0 = off (not a tail lever)
SCAN_MOM_MIN   = float(os.environ.get("SCAN_MOM_MIN", 0.0))     # SKIP-SIDEWAYS: require stock trailing return > this (set -9 to disable)
MOM_WIN        = int(os.environ.get("MOM_WIN", 42))             # lookback (days) for skip-sideways gate. Sweep: <=21d fails, 42d most balanced (both halves +332/+453), 126d higher but 2020-front-loaded. Magnitude window-fragile.


def tuned_pass(dd_score, ulcer, bars_after_cross, cap, srsi, nrsi=None, mom6=None):
    """TAIL-targeting selectivity on top of the faithful Pine BUY: high ulcer + not mega + non-euphoric DD +
    the stock must be TRENDING not sideways (6-mo momentum > SCAN_MOM_MIN)."""
    if SCAN_ULCER_MIN and not (ulcer >= SCAN_ULCER_MIN):
        return False
    if not (0 < dd_score <= SCAN_DD_CEIL):
        return False
    if SCAN_MAXBAC and not (0 <= bars_after_cross <= SCAN_MAXBAC):
        return False
    if SCAN_EX_MICRO and cap == "micro":
        return False
    if SCAN_EX_MEGA and cap == "mega":
        return False
    if SCAN_SRSI_MAX and not (srsi <= SCAN_SRSI_MAX):
        return False
    if SCAN_NRSI_MAX and nrsi is not None and not (nrsi <= SCAN_NRSI_MAX):
        return False
    if SCAN_MOM_MIN > -9 and mom6 is not None and not (np.isfinite(mom6) and mom6 > SCAN_MOM_MIN):
        return False
    return True

# Human-readable sector names for the sleeve ETFs used in surv_sector/delisted_sector
SECTOR_NAME = {
    "XLK": "Technology", "XLC": "Communications", "XLF": "Financials", "XLV": "Healthcare",
    "XLE": "Energy", "XLI": "Industrials", "XLY": "Cons.Disc.", "XLP": "Cons.Staples",
    "XLB": "Materials", "XLRE": "Real Estate", "XLU": "Utilities",
    "SOXX": "Semis", "SMH": "Semis", "IGV": "Software", "SKYY": "Cloud", "CIBR": "Cyber",
    "HACK": "Cyber", "FINX": "Fintech", "BOTZ": "Robotics", "ARKK": "Innovation", "IBUY": "E-comm",
    "UFO": "Space", "QTUM": "Quantum", "AIQ": "AI", "WCLD": "Cloud", "IPAY": "Payments", "MAGS": "MegaTech",
}


def wilder_rsi(x, n):
    d = x.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))


def ema(x, n):
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def compute_indicators(close, ohlc4):
    """All indicator panels, faithful to pine.txt. Returns a dict of DataFrames aligned to `close`."""
    ret = close.pct_change()
    # --- Rolling Sortino (own return, population downside stdev), smoothed, RSI, EMA cross line ---
    mean_r = ret.rolling(SORT_WIN).mean()
    dsd = ret.clip(upper=0.0).rolling(SORT_WIN).std(ddof=0)
    adj_rf = (1.0 + RF) ** (1.0 / 525600.0) - 1.0                  # daily chart: timeframe.multiplier=1 -> ~0
    sortino = (mean_r - adj_rf) / dsd.replace(0.0, np.nan)
    sortino_s = sortino.rolling(SORT_SMOOTH, min_periods=SORT_SMOOTH).mean()
    srsi = wilder_rsi(sortino_s.ffill().fillna(0.0), SORT_RSI)     # match TradingView na-handling (validated on BTC)
    srsi_ema = ema(srsi, SORT_EMA)
    # --- OHLC4 RSI (SELL confirm only) ---
    nrsi = wilder_rsi(ohlc4, OHLC_RSI)
    nrsi_ema = ema(nrsi, OHLC_EMA)
    # --- DD Downside Score (EMA-based) ---
    dret = close / close.shift(1) - 1.0
    mret = ema(dret, DD_RET)
    dsq = dret.clip(upper=0.0) ** 2
    dvar = ema(dsq, DD_DOWN)
    ddev = np.sqrt(dvar)
    hi = close.rolling(DD_DD).max()
    ddraw = (close / hi - 1.0).where(hi > 0, 0.0)
    absdd = (-ddraw).clip(lower=0.0)
    totrisk = ddev + absdd * DD_W
    raw = (mret / totrisk.replace(0.0, np.nan)).where(totrisk > 0, 0.0)
    dd_score = ema(raw, DD_SMOOTH)
    # --- Ulcer Index ---
    upeak = close.rolling(ULCER_PEAK).max()
    udd = (100.0 * (close / upeak - 1.0)).where(upeak > 0, 0.0)
    ulcer = np.sqrt((udd ** 2).rolling(ULCER_LEN).mean())
    return dict(srsi=srsi, srsi_ema=srsi_ema, nrsi=nrsi, nrsi_ema=nrsi_ema,
                dd_score=dd_score, ulcer=ulcer, dist_hi=ddraw)


def run_state_machine(col_idx, close_np, srsi, srsi_ema, nrsi, nrsi_ema, dd, ulcer, valid, warm):
    """Replicate the Pine buy/sell state machine for ONE ticker's arrays. Returns list of
    (entry_i, exit_i, exit_reason, bars_after_cross_at_entry)."""
    n = len(close_np)
    # vectorized cross events for this column
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)          # sortino RSI cross up / down of its EMA
    ncd = np.zeros(n, bool)                                 # OHLC4 RSI cross DOWN of its EMA
    s, se, nr, ne = srsi, srsi_ema, nrsi, nrsi_ema
    for i in range(1, n):
        if s[i - 1] <= se[i - 1] and s[i] > se[i]:
            cu[i] = True
        if s[i - 1] >= se[i - 1] and s[i] < se[i]:
            cd[i] = True
        if nr[i - 1] >= ne[i - 1] and nr[i] < ne[i]:
            ncd[i] = True
    sortino_bull = s > se
    nrsi_above = nr > ne
    nrsi_below = nr < ne
    dd_pos = dd > 0.0
    dd_neg = dd < 0.0
    ulcer_hi = ulcer > ULCER_MIN

    trips = []
    bull_cross_bar = -1
    in_pos = False; entry_bar = -1
    warn = False; waiting = False
    for i in range(n):
        if not (valid[i] and i >= warm):
            # still advance cross memory so warmup crosses are respected
            if cu[i]:
                bull_cross_bar = i
            if cd[i]:
                bull_cross_bar = -1
            continue
        # --- bull cross memory ---
        if cu[i]:
            bull_cross_bar = i
        if cd[i]:
            bull_cross_bar = -1
        if bull_cross_bar >= 0 and (i - bull_cross_bar) > CONFIRM_WIN:
            bull_cross_bar = -1
        bars_after = (i - bull_cross_bar) if bull_cross_bar >= 0 else -1
        valid_bull = bull_cross_bar >= 0 and 0 <= bars_after <= CONFIRM_WIN
        # --- bearish exit state (only while long) ---
        if in_pos and cd[i]:
            warn = True
            waiting = bool(nrsi_above[i])
        if in_pos and warn and waiting and ncd[i]:
            waiting = False
        if in_pos and warn and cu[i]:
            warn = False; waiting = False
        # --- signals ---
        buy = (not in_pos) and valid_bull and bool(sortino_bull[i]) and bool(dd_pos[i]) and bool(ulcer_hi[i])
        normal_exit_ready = warn and (not waiting) and bool(nrsi_below[i])
        normal_sell = in_pos and normal_exit_ready and bool(dd_neg[i])
        hard_sell = in_pos and entry_bar >= 0 and i > entry_bar and dd[i] >= DD_HARD
        sell = normal_sell or hard_sell
        # --- execute ---
        if buy:
            in_pos = True; entry_bar = i
            bull_cross_bar = -1; warn = False; waiting = False
            trips.append([i, -1, "open", bars_after])
        elif sell:
            in_pos = False
            trips[-1][1] = i
            trips[-1][2] = "hard_dd" if hard_sell else "normal"
            entry_bar = -1; bull_cross_bar = -1; warn = False; waiting = False
    return trips


def cap_bucket(mc):
    if mc is None or not np.isfinite(mc):
        return "unknown"
    b = mc / 1e9
    if b >= 200: return "mega"
    if b >= 10: return "large"
    if b >= 2: return "mid"
    if b >= 0.3: return "small"
    return "micro"


def qbins(vals, labels):
    """Quantile bin edges over finite vals into len(labels) buckets; returns (edges, assign_fn)."""
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < len(labels) * 5:
        return None
    qs = np.linspace(0, 1, len(labels) + 1)
    edges = np.unique(np.quantile(v, qs))
    if len(edges) < 3:
        return None
    return edges


def bin_report(entries, feat, edges, horizon_key="fwd21"):
    """Mean/median forward return + win rate per feature bin."""
    rows = []
    fv = np.array([e[feat] for e in entries], float)
    rv = np.array([e[horizon_key] for e in entries], float)
    ok = np.isfinite(fv) & np.isfinite(rv)
    fv, rv = fv[ok], rv[ok]
    idx = np.digitize(fv, edges[1:-1])
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        r = rv[m]
        rows.append((f"{edges[b]:.3g}..{edges[b+1]:.3g}", int(m.sum()),
                     float(np.mean(r)), float(np.median(r)), float(np.mean(r > 0) * 100)))
    return rows


def _equity_export(books, spy, qqq, days):
    """Daily equity+drawdown series for the two headline books + SPY/QQQ (normalized to the book span), for charting."""
    out = {}
    for lab in ("ALL-signals EW", "max5 EW"):
        if lab in books:
            out[lab] = books[lab]["eq"]
    ref = books.get("ALL-signals EW")
    if ref:
        a, z = ref["start"], ref["end"]
        for name, s in (("SPY", spy), ("QQQ", qqq)):
            if s is None:
                continue
            ss = s.reindex(days).ffill()
            ss = ss[(ss.index >= pd.Timestamp(a)) & (ss.index <= pd.Timestamp(z))].dropna()
            if len(ss) < 2:
                continue
            eq = ss / ss.iloc[0]; uw = ((eq - eq.cummax()) / eq.cummax()) * 100
            out[name] = [[str(d.date()), round(float(e), 4), round(float(u), 2)] for d, e, u in zip(eq.index, eq.values, uw.values)]
    return out


def run_book(trips, close_np, days, maxpos=0, weight_by=None):
    """Daily equal-weight (or ulcer-weighted) portfolio of the tail trips. maxpos=0 = hold ALL open signals
    (each sized 1/open-count); maxpos>0 = cap concurrent names, greedy first-come by entry day. Returns
    (metrics_dict, monthly_list)."""
    n = len(days)
    if maxpos > 0:                                                    # capacity: admit in entry-date order while a slot is free
        adm = []; open_ends = []
        for tr in sorted(trips, key=lambda x: x[1]):
            open_ends = [e for e in open_ends if e > tr[1]]
            if len(open_ends) < maxpos:
                adm.append(tr); open_ends.append(tr[2])
        trips = adm
    dsum = np.zeros(n); wsum = np.zeros(n); cnt = np.zeros(n)
    for tr in trips:
        c, ei, xi, isb, ulc = tr[0], tr[1], tr[2], tr[3], tr[4]
        P = tr[5] if len(tr) > 5 else 0.0                             # optional limit/entry price override
        w = (ulc if weight_by == "ulcer" else 1.0)
        if P > 0 and close_np[ei, c] > 0:                            # entered intraday at P on day ei -> book the fill-to-close move
            dsum[ei] += w * (close_np[ei, c] / P - 1.0); wsum[ei] += w; cnt[ei] += 1
        for t in range(ei + 1, xi + 1):
            r = (close_np[t, c] / close_np[t - 1, c] - 1.0) if (close_np[t - 1, c] > 0 and close_np[t, c] > 0) else 0.0
            if isb and t == xi:
                r = -1.0                                              # confirmed bankruptcy -> -100% on the exit bar
            dsum[t] += w * r; wsum[t] += w; cnt[t] += 1
    port = np.where(wsum > 0, dsum / wsum, 0.0)
    inv = np.where(cnt > 0)[0]
    if len(inv) < 2:
        return None, [], []
    a, z = inv[0], inv[-1]
    pr = pd.Series(port[a:z + 1], index=days[a:z + 1])
    eq = (1 + pr).cumprod()
    total = float(eq.iloc[-1] - 1) * 100
    yrs = max((days[z] - days[a]).days / 365.25, 1e-9)
    cagr = ((1 + total / 100) ** (1 / yrs) - 1) * 100 if total > -100 else -100.0
    sh = float(pr.mean() / pr.std(ddof=1) * np.sqrt(252)) if pr.std(ddof=1) > 0 else float("nan")
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    avg_conc = float(cnt[a:z + 1].mean()); max_conc = int(cnt.max())
    monthly = [[str(d.date()), float(v)] for d, v in ((1 + pr).resample("ME").prod() - 1).items()]
    underwater = ((eq - eq.cummax()) / eq.cummax()) * 100
    eqd = [[str(d.date()), round(float(e), 4), round(float(u), 2)]                 # [date, equity(×), drawdown%]
           for d, e, u in zip(eq.index, eq.values, underwater.values)]
    return dict(total=round(total, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), maxdd=round(dd, 1),
                avg_concurrent=round(avg_conc, 1), max_concurrent=max_conc,
                start=str(days[a].date()), end=str(days[z].date()), n_positions=len(trips)), monthly, eqd


def bench_hold(cd_close, days, a_date, z_date):
    """Buy-hold total return of a benchmark close series over [a_date, z_date]."""
    s = cd_close.reindex(days).ffill()
    s = s[(s.index >= pd.Timestamp(a_date)) & (s.index <= pd.Timestamp(z_date))].dropna()
    if len(s) < 2:
        return None
    eq = s / s.iloc[0]
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    return dict(total=round(float(eq.iloc[-1] - 1) * 100, 1), maxdd=round(dd, 1))


def main():
    b = pickle.load(open(PANELS, "rb"))
    universe = list(b["common"])
    del_sector = b["delisted_sector"]; surv_sector = b["surv_sector"]; bankrupt_tk = set(b["bankrupt_tk"])
    mcap_m = b.get("mktcap_usd")
    sec_of = {t: (surv_sector.get(t) or del_sector.get(t)) for t in universe}
    if SECTOR_FILTER:
        keep_sec = set(SECTOR_FILTER.split(","))
        universe = [t for t in universe if sec_of.get(t) in keep_sec]
        print(f"SECTOR_FILTER={sorted(keep_sec)} -> {len(universe)} names", flush=True)

    cd = load_candles(universe)
    close = pd.DataFrame({t: cd[t]["Close"] for t in universe if t in cd and "Close" in cd[t]}).sort_index()
    close = close[close > 0]
    days = close.index
    op = pd.DataFrame({t: cd[t]["Open"] for t in universe if t in cd}).reindex(index=days, columns=close.columns)
    hp = pd.DataFrame({t: cd[t]["High"] for t in universe if t in cd}).reindex(index=days, columns=close.columns)
    lp = pd.DataFrame({t: cd[t]["Low"] for t in universe if t in cd}).reindex(index=days, columns=close.columns)
    vol = pd.DataFrame({t: cd[t]["Volume"] for t in universe if t in cd}).reindex(index=days, columns=close.columns)
    ohlc4 = (op + hp + lp + close) / 4.0
    dvol = (close * vol).rolling(63, min_periods=20).mean()
    ma50 = close.rolling(50).mean(); ma200 = close.rolling(200).mean()

    ind = compute_indicators(close, ohlc4)
    cols = list(close.columns)
    close_np = close.to_numpy()
    low_np = lp.reindex(index=close.index, columns=close.columns).to_numpy()   # intraday low, for limit-fill detection
    valid_np = close.notna().to_numpy()
    srsi = ind["srsi"].to_numpy(); srsi_ema = ind["srsi_ema"].to_numpy()
    nrsi = ind["nrsi"].to_numpy(); nrsi_ema = ind["nrsi_ema"].to_numpy()
    dd = ind["dd_score"].to_numpy(); ulcer = ind["ulcer"].to_numpy()
    dvol_np = dvol.to_numpy(); ma50_np = ma50.to_numpy(); ma200_np = ma200.to_numpy()
    disthi_np = ind["dist_hi"].to_numpy()
    mom_np = close.pct_change(MOM_WIN).to_numpy()                   # stock trailing momentum (SKIP-SIDEWAYS gate, window=MOM_WIN)
    SWEEP_WINS = [10, 21, 42, 63, 126]                             # candidate lookbacks to sweep (2wk..6mo)
    mom_sweep = {w: close.pct_change(w).to_numpy() for w in SWEEP_WINS}
    _ma50 = close.rolling(50).mean(); _ma20 = close.rolling(20).mean()
    above_rising_ma50 = ((close > _ma50) & (_ma50 > _ma50.shift(20))).to_numpy()   # price above a RISING 50MA
    above_rising_ma20 = ((close > _ma20) & (_ma20 > _ma20.shift(10))).to_numpy()   # price above a RISING 20MA (faster)
    warm = max(SORT_WIN + SORT_SMOOTH + SORT_RSI + SORT_EMA, DD_DD, 200)   # indicator warmup floor

    # PIT market cap: monthly panel ffilled to daily
    capd = None
    if mcap_m is not None:
        capd = mcap_m.sort_index().reindex(columns=cols).reindex(days, method="ffill")
        capd_np = capd.to_numpy()
    # PIT top-N mcap membership mask (optional)
    top_ok = None
    if TOP_MCAP and mcap_m is not None:
        tm = mcap_m.sort_index().reindex(columns=cols)
        top = (tm.rank(axis=1, ascending=False) <= TOP_MCAP)
        top_ok = top.reindex(days, method="ffill").fillna(False).to_numpy()

    start_i = int(np.searchsorted(days.values, np.datetime64(START_DATE)))
    n = len(days)
    # MARKET GATE: only buy when SPY's own RSI is above its moving average (user 2026-09-25)
    mkt_ok = np.ones(n, bool)
    if MKT_GATE:
        _spy = load_candles(["SPY"]).get("SPY", {}).get("Close")
        if _spy is not None:
            _spy = _spy.dropna(); _spy = _spy[_spy > 0]
            _sr = wilder_rsi(_spy, MKT_RSI_LEN)                        # RSI on SPY's NATIVE bars (not reindexed)
            _ma = ema(_sr, MKT_MA_LEN) if MKT_MA_TYPE == "ema" else _sr.rolling(MKT_MA_LEN).mean()
            mkt_ok = (_sr > _ma).reindex(days, method="ffill").fillna(False).to_numpy()  # map to trading days
            _win = days >= pd.Timestamp(START_DATE)                    # report buyable% over the ACTUAL entry window (SPY exists 2015+)
            print(f"MKT_GATE on: SPY RSI{MKT_RSI_LEN} > {MKT_MA_TYPE.upper()}{MKT_MA_LEN} -> buyable on {mkt_ok[_win].mean()*100:.0f}% of days (>= {START_DATE})", flush=True)
    last_valid = np.where(valid_np.any(axis=0), n - 1 - np.argmax(valid_np[::-1], axis=0), -1)  # last real bar per name
    entries = []
    port_trips = []                                                   # (c, ei, xi, is_bank, ulcer) for TUNED entries (momentum-gated) -> portfolio
    tail_trips_all = []                                               # same, but BEFORE the momentum gate -> for the lookback sweep
    live_buys = []; live_armed = []
    raw_buy_count = 0

    for c, t in enumerate(cols):
        trips = run_state_machine(c, close_np[:, c], srsi[:, c], srsi_ema[:, c], nrsi[:, c], nrsi_ema[:, c],
                                  dd[:, c], ulcer[:, c], valid_np[:, c], warm)
        sec = sec_of.get(t); secname = SECTOR_NAME.get(sec, sec or "?")
        for (ei, xi, reason, bac) in trips:
            raw_buy_count += 1
            if ei < start_i:
                continue
            # liquidity filter at entry (scanner realism)
            if not (dvol_np[ei, c] >= MINVOL):
                continue
            if top_ok is not None and not top_ok[ei, c]:
                continue
            mc_ei = capd_np[ei, c] if capd is not None else np.nan
            if MCAP_FLOOR > 0 and not (np.isfinite(mc_ei) and mc_ei >= MCAP_FLOOR):
                continue                                              # HARD $1B+ PIT floor (tradeable universe)
            if MKT_GATE and not mkt_ok[ei]:
                continue                                              # only buy when SPY RSI > its average
            ep = close_np[ei, c]
            if not (ep > 0):
                continue
            rec = dict(ticker=t, sector=secname, sector_etf=sec, date=str(days[ei].date()),
                       dd_score=float(dd[ei, c]), ulcer=float(ulcer[ei, c]),
                       srsi=float(srsi[ei, c]), srsi_spread=float(srsi[ei, c] - srsi_ema[ei, c]),
                       nrsi=float(nrsi[ei, c]), nrsi_spread=float(nrsi[ei, c] - nrsi_ema[ei, c]),
                       bars_after_cross=int(bac), dist_hi=float(disthi_np[ei, c]) * 100.0,
                       above_ma50=bool(np.isfinite(ma50_np[ei, c]) and ep > ma50_np[ei, c]),
                       above_ma200=bool(np.isfinite(ma200_np[ei, c]) and ep > ma200_np[ei, c]),
                       dvol_m=round(float(dvol_np[ei, c]) / 1e6, 1))
            mc = capd_np[ei, c] if capd is not None else np.nan
            rec["cap"] = cap_bucket(mc); rec["mcap_b"] = round(float(mc) / 1e9, 2) if np.isfinite(mc) else None
            rec["mom"] = float(mom_np[ei, c]) * 100.0 if np.isfinite(mom_np[ei, c]) else None
            # forward returns (signal-bar close -> +h close)
            for h in HORIZONS:
                j = ei + h
                rec[f"fwd{h}"] = float(close_np[j, c] / ep - 1.0) * 100.0 if j < n and close_np[j, c] > 0 else np.nan
            # realized round trip (Pine exit)
            if xi > 0 and close_np[xi, c] > 0:
                rr = close_np[xi, c] / ep - 1.0
                if reason != "open" and (t in bankrupt_tk) and xi >= n - 3:
                    rr = -1.0
                rec["rt_ret"] = float(rr) * 100.0; rec["hold_days"] = int(xi - ei); rec["exit_reason"] = reason
            else:
                rec["rt_ret"] = np.nan; rec["hold_days"] = int(n - 1 - ei); rec["exit_reason"] = "open"
            is_tail = tuned_pass(rec["dd_score"], rec["ulcer"], rec["bars_after_cross"], rec["cap"], rec["srsi"], rec["nrsi"], None)  # tail set, momentum-agnostic
            mv = mom_np[ei, c]
            mom_ok = (SCAN_MOM_MIN <= -9) or (np.isfinite(mv) and mv > SCAN_MOM_MIN)
            rec["tuned"] = bool(is_tail and mom_ok)
            entries.append(rec)
            if is_tail:                                              # collect trips (tail_trips_all = pre-momentum, for the sweep)
                xe = xi if xi > 0 else int(last_valid[c])
                isb = (t in bankrupt_tk) and (xe >= last_valid[c]) and (last_valid[c] < n - 3)
                if xe > ei:
                    tail_trips_all.append((c, ei, xe, bool(isb), float(ulcer[ei, c])))
                    if mom_ok:
                        port_trips.append((c, ei, xe, bool(isb), float(ulcer[ei, c])))

        # live scanner: recompute the state at the LAST valid bar for this name
        li = np.where(valid_np[:, c])[0]
        if len(li) == 0:
            continue
        last = li[-1]
        if last < warm:
            continue
        # is a fresh BUY on the last bar? (reuse the trip list -> entry on last bar)
        fresh = any(ei == last for (ei, _, _, _) in trips)
        # armed = a bull cross within CONFIRM_WIN and sortino still bull and gates pass, but not yet fired
        s_ = srsi[:, c]; se_ = srsi_ema[:, c]
        recent_cross = False; bac_live = -1
        for k in range(max(warm, last - CONFIRM_WIN), last + 1):
            if k >= 1 and s_[k - 1] <= se_[k - 1] and s_[k] > se_[k]:
                recent_cross = True; bac_live = last - k
        gates_ok = (dd[last, c] > 0) and (ulcer[last, c] > ULCER_MIN) and (s_[last] > se_[last])
        mc_last = capd_np[last, c] if capd is not None else np.nan
        liq_ok = (dvol_np[last, c] >= MINVOL and (top_ok is None or top_ok[last, c])
                  and (MCAP_FLOOR <= 0 or (np.isfinite(mc_last) and mc_last >= MCAP_FLOOR))
                  and (not MKT_GATE or mkt_ok[last]))
        row = dict(ticker=t, sector=SECTOR_NAME.get(sec, sec or "?"), date=str(days[last].date()),
                   dd_score=round(float(dd[last, c]), 3), ulcer=round(float(ulcer[last, c]), 2),
                   srsi=round(float(s_[last]), 1), srsi_ema=round(float(se_[last]), 1),
                   bars_after_cross=int(bac_live), dvol_m=round(float(dvol_np[last, c]) / 1e6, 1))
        cap_live = cap_bucket(capd_np[last, c] if capd is not None else np.nan)
        row["cap"] = cap_live
        row["mom"] = round(float(mom_np[last, c]) * 100, 1) if np.isfinite(mom_np[last, c]) else None
        row["tuned"] = bool(liq_ok and tuned_pass(float(dd[last, c]), float(ulcer[last, c]), int(bac_live), cap_live, float(s_[last]), float(nrsi[last, c]), mom_np[last, c] if np.isfinite(mom_np[last, c]) else None))
        if fresh and liq_ok:
            live_buys.append(row)
        elif recent_cross and gates_ok and liq_ok and not fresh:
            live_armed.append(row)

    n_ent = len(entries)
    print(f"\nuniverse={len(cols)}  raw Pine BUYs(all history)={raw_buy_count}  "
          f"qualified entries(>= {START_DATE}, liq, mcap>=${MCAP_FLOOR/1e9:g}B)={n_ent}", flush=True)
    if n_ent == 0:
        print("no entries; aborting", flush=True); return

    # ---------- headline stats ----------
    def stats(sub, key="fwd21"):
        r = np.array([e[key] for e in sub if np.isfinite(e.get(key, np.nan))], float)
        if len(r) == 0:
            return None
        return dict(n=len(r), mean=round(float(np.mean(r)), 2), median=round(float(np.median(r)), 2),
                    win=round(float(np.mean(r > 0) * 100), 1))
    rt = np.array([e["rt_ret"] for e in entries if np.isfinite(e.get("rt_ret", np.nan))], float)
    print("\n=== HEADLINE (all qualified entries) ===", flush=True)
    for h in HORIZONS:
        s = stats(entries, f"fwd{h}")
        if s: print(f"  fwd{h:>2}d: n={s['n']:>5}  mean {s['mean']:+.2f}%  median {s['median']:+.2f}%  win {s['win']:.1f}%", flush=True)
    if len(rt):
        print(f"  round-trip (Pine exit): n={len(rt)}  mean {np.mean(rt):+.2f}%  median {np.median(rt):+.2f}%  "
              f"win {np.mean(rt>0)*100:.1f}%  avg_hold {np.mean([e['hold_days'] for e in entries]):.0f}d", flush=True)

    # ---------- baseline vs TUNED = TAIL-TARGETING ----------
    tuned = [e for e in entries if e["tuned"]]
    print(f"\n=== TAIL-TARGETING SCANNER  (ulcer>={SCAN_ULCER_MIN:g}, 0<dd<={SCAN_DD_CEIL:g}, "
          f"ex-mega={bool(SCAN_EX_MEGA)}, ex-micro={bool(SCAN_EX_MICRO)}, 6mo-mom>{SCAN_MOM_MIN:.0%}=skip-sideways) ===", flush=True)
    print(f"  kept {len(tuned)}/{n_ent} entries ({100*len(tuned)/max(1,n_ent):.0f}%)", flush=True)
    for h in HORIZONS:
        s = stats(tuned, f"fwd{h}")
        if s: print(f"  fwd{h:>2}d: n={s['n']:>5}  mean {s['mean']:+.2f}%  median {s['median']:+.2f}%  win {s['win']:.1f}%", flush=True)
    rtt = np.array([e["rt_ret"] for e in tuned if np.isfinite(e.get("rt_ret", np.nan))], float)
    rt_all = np.array([e["rt_ret"] for e in entries if np.isfinite(e.get("rt_ret", np.nan))], float)
    if len(rtt):
        big_all = int((rt_all >= 50).sum())
        print(f"  round-trip: n={len(rtt)}  mean {np.mean(rtt):+.2f}%  median {np.median(rtt):+.2f}%  "
              f"win {np.mean(rtt>0)*100:.1f}%  avg_hold {np.mean([e['hold_days'] for e in tuned]):.0f}d", flush=True)
        print(f"  TAIL: BIG(>=+50%) {np.mean(rtt>=50)*100:.1f}% of tuned trades  HUGE(>=+100%) {np.mean(rtt>=100)*100:.1f}%  "
              f"captures {100*int((rtt>=50).sum())/max(1,big_all):.0f}% of ALL big trades in the universe", flush=True)

    # ---------- PORTFOLIO equity curve (return + drawdown) ----------
    print("\n=== PORTFOLIO (tail basket; entry=signal close, exit=Pine exit; equal-weight) ===", flush=True)
    books = {}
    qd = load_candles(["SPY", "QQQ"])
    spy = qd.get("SPY", {}).get("Close"); qqq = qd.get("QQQ", {}).get("Close")
    for lab, mp, wb in [("ALL-signals EW", 0, None), ("ALL-signals ULCER-wt", 0, "ulcer"),
                        ("max20 EW", 20, None), ("max10 EW", 10, None), ("max5 EW", 5, None)]:
        m, monthly, eqd = run_book(port_trips, close_np, days, maxpos=mp, weight_by=wb)
        if not m:
            continue
        books[lab] = dict(m, monthly=monthly, eq=eqd)
        bs = f"  SPY {bench_hold(spy, days, m['start'], m['end'])['total']:+,.0f}% / QQQ {bench_hold(qqq, days, m['start'], m['end'])['total']:+,.0f}% (same span)" if spy is not None else ""
        print(f"  {lab:<22} total {m['total']:+,.1f}%  CAGR {m['cagr']:+.1f}%  Sharpe {m['sharpe']:.2f}  "
              f"maxDD {m['maxdd']:.1f}%  avg/max concurrent {m['avg_concurrent']:.0f}/{m['max_concurrent']}  "
              f"n={m['n_positions']}", flush=True)
        if lab == "ALL-signals EW" and spy is not None:
            sp = bench_hold(spy, days, m['start'], m['end']); qq = bench_hold(qqq, days, m['start'], m['end'])
            print(f"    vs buy-hold same span: SPY {sp['total']:+,.0f}% (maxDD {sp['maxdd']:.0f}%)  "
                  f"QQQ {qq['total']:+,.0f}% (maxDD {qq['maxdd']:.0f}%)", flush=True)

    # ---------- SKIP-SIDEWAYS LOOKBACK SWEEP (user: 6mo may be too long on daily; try shorter/other) ----------
    def sweep_gate(trips, arr, thr=0.0, boolmode=False):
        if boolmode:
            return [tr for tr in trips if bool(arr[tr[1], tr[0]])]
        return [tr for tr in trips if np.isfinite(arr[tr[1], tr[0]]) and arr[tr[1], tr[0]] > thr]
    base = len(tail_trips_all)
    print(f"\n=== SKIP-SIDEWAYS SWEEP (trend gate on tail set, n={base}) ===", flush=True)
    _midi = int(np.searchsorted(days.values, np.datetime64("2021-01-01")))
    def sline(name, sub):
        if len(sub) < 10:
            print(f"  {name:<24} keeps {100*len(sub)/max(1,base):>3.0f}% (n={len(sub):>3})  (too few)", flush=True); return
        m0, _, _ = run_book(sub, close_np, days, maxpos=0)
        h1, _, _ = run_book([t for t in sub if t[1] < _midi], close_np, days, maxpos=0)
        h2, _, _ = run_book([t for t in sub if t[1] >= _midi], close_np, days, maxpos=0)
        h1s = f"{h1['total']:+,.0f}%" if h1 else "n/a"; h2s = f"{h2['total']:+,.0f}%" if h2 else "n/a"
        print(f"  {name:<24} keeps {100*len(sub)/base:>3.0f}% (n={len(sub):>4})  "
              f"ALL-EW {m0['total']:+8,.0f}%/Sh{m0['sharpe']:.2f}/DD{m0['maxdd']:.0f}  "
              f"[H1 16-20 {h1s}  H2 21-26 {h2s}]", flush=True)
    sline("no gate (all tail)", tail_trips_all)
    for w in SWEEP_WINS:
        sline(f"return[{w}d] > 0", sweep_gate(tail_trips_all, mom_sweep[w], 0.0))
    sline("price>rising 20MA", sweep_gate(tail_trips_all, above_rising_ma20, boolmode=True))
    sline("price>rising 50MA", sweep_gate(tail_trips_all, above_rising_ma50, boolmode=True))
    # combos: fast + medium both up
    both = [tr for tr in tail_trips_all if np.isfinite(mom_sweep[21][tr[1], tr[0]]) and mom_sweep[21][tr[1], tr[0]] > 0
            and np.isfinite(mom_sweep[63][tr[1], tr[0]]) and mom_sweep[63][tr[1], tr[0]] > 0]
    sline("return[21d]>0 & [63d]>0", both)

    # ---------- criteria attribution (which criteria find the best trades) ----------
    print("\n=== CRITERIA ATTRIBUTION (fwd21d mean / median / win% by bin) ===", flush=True)
    feat_labels = ["dd_score", "ulcer", "srsi", "srsi_spread", "nrsi", "bars_after_cross", "dist_hi"]
    for feat in feat_labels:
        vals = [e[feat] for e in entries]
        edges = qbins(vals, ["q1", "q2", "q3", "q4"])
        if edges is None:
            continue
        rows = bin_report(entries, feat, edges, "fwd21")
        if not rows:
            continue
        print(f"\n  [{feat}]", flush=True)
        for (lab, cnt, mn, md, win) in rows:
            print(f"     {lab:>16}  n={cnt:>5}  mean {mn:+.2f}%  median {md:+.2f}%  win {win:.0f}%", flush=True)

    # boolean criteria
    print("\n  [above_ma50 / above_ma200]", flush=True)
    for feat in ("above_ma50", "above_ma200"):
        for val in (True, False):
            sub = [e for e in entries if e[feat] == val]
            s = stats(sub)
            if s: print(f"     {feat}={val!s:>5}  n={s['n']:>5}  mean {s['mean']:+.2f}%  median {s['median']:+.2f}%  win {s['win']:.0f}%", flush=True)

    # ---------- segmentation: cap x sector (hard rule: never average across the universe) ----------
    print("\n=== SEGMENT: cap x sector (fwd21d) ===", flush=True)
    from collections import defaultdict
    seg = defaultdict(list)
    for e in entries:
        seg[(e["cap"], e["sector"])].append(e)
    order = sorted(seg.keys(), key=lambda k: -len(seg[k]))
    print(f"  {'cap':>6} {'sector':>14} {'n':>5} {'mean':>8} {'median':>8} {'win%':>6}", flush=True)
    for k in order:
        if len(seg[k]) < 15:
            continue
        s = stats(seg[k])
        if s: print(f"  {k[0]:>6} {k[1]:>14} {s['n']:>5} {s['mean']:>+7.2f}% {s['median']:>+7.2f}% {s['win']:>5.0f}%", flush=True)

    # ---------- live scanner ----------
    tuned_buys = [r for r in live_buys if r.get("tuned")]
    print(f"\n=== LIVE SCANNER (as of {str(days[-1].date())}) ===", flush=True)
    print(f"  FRESH BUYs: {len(live_buys)}  (of which TUNED: {len(tuned_buys)})   ARMED: {len(live_armed)}", flush=True)
    print("  -- TAIL-TARGET buys (ulcer>=%g, 0<dd<=%g, non-mega) --" % (SCAN_ULCER_MIN, SCAN_DD_CEIL), flush=True)
    for r in sorted(tuned_buys, key=lambda x: x["srsi"])[:40] or [None]:
        if r is None:
            print("     (none today)", flush=True); break
        print(f"   BUY* {r['ticker']:>6} {r['sector']:>12} {r['cap']:>5}  dd={r['dd_score']:+.2f} "
              f"srsi={r['srsi']:.0f}>{r['srsi_ema']:.0f} bac={r['bars_after_cross']} ${r['dvol_m']:.0f}M", flush=True)
    print("  -- all other fresh Pine buys (not tuned) --", flush=True)
    for r in sorted([x for x in live_buys if not x.get("tuned")], key=lambda x: -x["dd_score"])[:20]:
        print(f"   buy  {r['ticker']:>6} {r['sector']:>12} {r['cap']:>5}  dd={r['dd_score']:+.2f} "
              f"srsi={r['srsi']:.0f}>{r['srsi_ema']:.0f} bac={r['bars_after_cross']} ${r['dvol_m']:.0f}M", flush=True)

    # ---------- persist ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = dict(
        strategy="sortino_dd_scanner (Pine v6 port: own-Sortino-RSI cross + DD>0; OHLC4-RSI exit; ULCER OFF per user)",
        params=dict(SORT_WIN=SORT_WIN, SORT_SMOOTH=SORT_SMOOTH, SORT_RSI=SORT_RSI, SORT_EMA=SORT_EMA,
                    CONFIRM_WIN=CONFIRM_WIN, DD_HARD=DD_HARD, ULCER_MIN=ULCER_MIN, MINVOL=MINVOL,
                    MCAP_FLOOR=MCAP_FLOOR, TOP_MCAP=TOP_MCAP, START_DATE=START_DATE, HORIZONS=HORIZONS),
        tuned_config=dict(mode="tail-targeting+skip-sideways", ulcer_min=SCAN_ULCER_MIN, dd_ceil=SCAN_DD_CEIL,
                          exclude_mega=bool(SCAN_EX_MEGA), exclude_micro=bool(SCAN_EX_MICRO),
                          mcap_floor=MCAP_FLOOR, mom6_min=SCAN_MOM_MIN),
        universe=len(cols), raw_buys=raw_buy_count, n_entries=n_ent,
        headline={f"fwd{h}": stats(entries, f"fwd{h}") for h in HORIZONS},
        headline_tuned={f"fwd{h}": stats(tuned, f"fwd{h}") for h in HORIZONS},
        round_trip=(dict(n=len(rt), mean=round(float(np.mean(rt)), 2), median=round(float(np.median(rt)), 2),
                         win=round(float(np.mean(rt > 0) * 100), 1)) if len(rt) else None),
        round_trip_tuned=(dict(n=len(rtt), mean=round(float(np.mean(rtt)), 2), median=round(float(np.median(rtt)), 2),
                               win=round(float(np.mean(rtt > 0) * 100), 1)) if len(rtt) else None),
        portfolio={k: {kk: vv for kk, vv in v.items() if kk not in ("monthly", "eq")} for k, v in books.items()},
        portfolio_monthly={k: v["monthly"] for k, v in books.items()},
        portfolio_equity=_equity_export(books, spy, qqq, days),
        as_of=str(days[-1].date()), live_buys=live_buys, live_buys_tuned=[r for r in live_buys if r.get("tuned")],
        live_armed=live_armed)
    json.dump(payload, open(f"{OUT_DIR}/sortino_dd_scanner.json", "w"), indent=0)
    # full entries table (for later re-analysis / re-tuning)
    pd.DataFrame(entries).to_csv(f"{OUT_DIR}/sortino_dd_entries.csv", index=False)
    print(f"\nwrote {OUT_DIR}/sortino_dd_scanner.json  (+ sortino_dd_entries.csv, {n_ent} rows)", flush=True)
    try:
        from core.models import BacktestResult as BR
        from django.utils import timezone as tz
        BR.objects.update_or_create(kind="sortino_dd_scanner", defaults=dict(computed_at=tz.now(), payload=payload))
        print("saved BacktestResult[sortino_dd_scanner]", flush=True)
    except Exception as e:
        print("BR save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
