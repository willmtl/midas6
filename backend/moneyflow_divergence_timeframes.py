#!/usr/bin/env python3
"""MONEY-FLOW indicator bullish-DIVERGENCE bounce books, all 4 (MFI / CMF / A-D line / OBV) x daily/weekly/monthly
candle (user: "test all 4"). Same divergence rule as the RSI sweep, for direct comparability: price makes a 20-bar
lower-low while the INDICATOR is above its value 10 bars ago (for the cumulative lines ADL/OBV that reads as
net-accumulation-despite-the-price-drop = the classic money-flow divergence; for bounded MFI/CMF it's the
oscillator higher-low). Standalone EW long book, hold 1 & 2 bars, survivorship-aware (covered u delisted, through
delisting), split-adjusted daily OHLCV, $5M/day + >$5.

Honors hard rules: standalone $5M selector, GROSS + NET (10/25/50 bps/side), by regime (all/bear/bull) vs SPY,
SEGMENTED by cap bucket per (indicator x timeframe), + cap x sector grid for the best (indicator,timeframe).
Saves BacktestResult[moneyflow_divergence_timeframes] + JSON.

A/D-line divergence is the closest cousin to the flagship's load-bearing conviction gate (ta.accdist), so ADL is
the one to watch. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/moneyflow_divergence_timeframes.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from core.models import Candle, Sector, DelistedCompany, Fundamental

DVOL_FLOOR = 5e6; PRICE_FLOOR = 5.0
LOW_WIN = 20; DIV_LAG = 10; START = "2014-06-01"
OUT = "/app/.data/studies/moneyflow_divergence_timeframes.json"
TF = {"daily": ("D", 252), "weekly": ("W-FRI", 52), "monthly": ("ME", 12)}
INDS = ["mfi", "cmf", "adl", "obv"]

etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
covered = set()
for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        try:
            tk = json.loads(line).get("ticker")
            if tk and "." not in tk and tk not in etfs:
                covered.add(tk)
        except Exception:
            pass
delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
universe = sorted((covered | delisted) - etfs)
print(f"universe {len(universe)} ({len(delisted)} delisted)", flush=True)

mcap = dict(Fundamental.objects.filter(ticker__in=universe).values_list("ticker", "market_cap"))
fsec = dict(Fundamental.objects.filter(ticker__in=universe).values_list("ticker", "sector"))


def cap_bucket(tk):
    mc = mcap.get(tk)
    if not mc or mc <= 0:
        return "unknown"
    return "micro(<0.5B)" if mc < 5e8 else "small(<2B)" if mc < 2e9 else "large(>=2B)"


spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte=START).values_list("date", "close")),
                   columns=["date", "close"]); spy["date"] = pd.to_datetime(spy["date"])
spy_c = spy.set_index("date")["close"].astype(float).sort_index()
spy_bull_d = spy_c > spy_c.rolling(200).mean()


def tf_ohlcv(h, l, c, v, tfname, rule, idx):
    """Aggregate daily OHLCV to the timeframe. Daily = native trading-day reindex (calendar 'D' resample injects
    weekend NaNs that break the indicators); weekly/monthly = H max, L min, C last, V sum."""
    if tfname == "daily":
        return h.reindex(idx), l.reindex(idx), c.reindex(idx), v.reindex(idx)
    return (h.resample(rule).max().reindex(idx), l.resample(rule).min().reindex(idx),
            c.resample(rule).last().reindex(idx), v.resample(rule).sum().reindex(idx))


def calc_inds(h, l, c, v):
    out = {}
    tp = (h + l + c) / 3.0; rmf = tp * v; dtp = tp.diff()
    pos = rmf.where(dtp > 0, 0.0); neg = rmf.where(dtp < 0, 0.0)
    mr = pos.rolling(14).sum() / neg.rolling(14).sum().replace(0, np.nan)
    out["mfi"] = 100 - 100 / (1 + mr)                                   # Money Flow Index (volume-weighted RSI)
    rng = (h - l).replace(0, np.nan); mfm = ((c - l) - (h - c)) / rng; mfv = (mfm * v).fillna(0.0)
    out["cmf"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)   # Chaikin Money Flow
    out["adl"] = mfv.cumsum()                                           # Accumulation/Distribution line (ta.accdist)
    out["obv"] = (np.sign(c.diff()).fillna(0.0) * v).cumsum()           # On-Balance Volume
    return out


# ---- per-timeframe panels: for each TF, FIRED[ind] + CL + fwd + regime ----
# Build index/spy per tf first
tf_idx, tf_spy_bull, tf_spy_fwd = {}, {}, {}
for tfname, (rule, _p) in TF.items():
    idx = spy_c.index if tfname == "daily" else spy_c.resample(rule).last().index
    stf = spy_c.reindex(idx) if tfname == "daily" else spy_c.resample(rule).last().reindex(idx)
    tf_idx[tfname] = idx
    tf_spy_bull[tfname] = (spy_bull_d.reindex(idx) if tfname == "daily" else spy_bull_d.resample(rule).last().reindex(idx)).fillna(False)
    tf_spy_fwd[tfname] = stf.shift(-1) / stf - 1.0

FIRED = {tf: {ind: {} for ind in INDS} for tf in TF}
CL = {tf: {} for tf in TF}
for i in range(0, len(universe), 300):
    rows = Candle.objects.filter(ticker__in=universe[i:i + 300], date__gte=START).values_list("ticker", "date", "high", "low", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["ticker", "date", "high", "low", "close", "volume"])
    if df.empty:
        continue
    df["date"] = pd.to_datetime(df["date"])
    for col in ("high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    for tk, g in df.groupby("ticker", sort=False):
        s = g.sort_values("date").set_index("date")
        cd = s["close"].ffill(limit=252)
        if cd.notna().sum() < 300:
            continue
        hd, ld, vd = s["high"].ffill(limit=252), s["low"].ffill(limit=252), s["volume"].fillna(0.0)
        dvol_d = (cd * s["volume"]).rolling(20, min_periods=10).mean()
        for tfname, (rule, _p) in TF.items():
            idx = tf_idx[tfname]
            h, l, c, v = tf_ohlcv(hd, ld, cd, vd, tfname, rule, idx)
            if c.notna().sum() < max(30, LOW_WIN + DIV_LAG + 5):
                continue
            inds = calc_inds(h, l, c, v)
            price_low = c <= c.rolling(LOW_WIN, min_periods=max(10, LOW_WIN - 5)).min()
            dvol_tf = dvol_d.reindex(idx) if tfname == "daily" else dvol_d.resample(rule).last().reindex(idx)
            liq = (dvol_tf >= DVOL_FLOOR) & (c > PRICE_FLOOR)
            CL[tfname][tk] = c
            for ind in INDS:
                x = inds[ind]
                FIRED[tfname][ind][tk] = (price_low & (x > x.shift(DIV_LAG)) & liq).fillna(False)
    print(f"  loaded {min(i+300,len(universe))}/{len(universe)}", flush=True)

# frame up
panels = {}
for tfname in TF:
    idx = tf_idx[tfname]
    cl = pd.DataFrame(CL[tfname])
    fired = {ind: pd.DataFrame(FIRED[tfname][ind]).reindex(columns=cl.columns).fillna(False) for ind in INDS}
    fwd1 = cl.shift(-1) / cl - 1.0; fwd2 = cl.shift(-2) / cl - 1.0
    panels[tfname] = dict(idx=idx, CL=cl, fired=fired, fwd1=fwd1, fwd2=fwd2,
                          spy_bull=tf_spy_bull[tfname], spy_fwd1=tf_spy_fwd[tfname])
    fc = {ind: int(fired[ind].values.sum()) for ind in INDS}
    print(f"[{tfname}] bars {idx[0].date()}..{idx[-1].date()} names {cl.shape[1]} fires {fc}", flush=True)


def run_book(P, ind, regime, hold, cost_bps, cols=None, min_n=5, member=None):
    FIREDdf, spy_bull, idx = P["fired"][ind], P["spy_bull"], P["idx"]
    fwd = P["fwd1"] if hold == 1 else P["fwd2"]
    if cols is not None:
        keep = [c for c in cols if c in FIREDdf.columns]
        FIREDdf = FIREDdf[keep]; fwd = fwd[keep]
    pr, turns, held, prev = [], [], [], set()
    for d in idx[:-hold]:
        if regime == "bear" and bool(spy_bull.get(d, False)):
            pr.append(0.0); held.append(0); continue
        if regime == "bull" and not bool(spy_bull.get(d, False)):
            pr.append(0.0); held.append(0); continue
        row = FIREDdf.loc[d].astype(bool); frow = fwd.loc[d]
        nm = [t for t in row.index if bool(row[t]) and pd.notna(frow[t])]
        if member is not None and d in member.index:      # PIT top-N membership at THIS period (no look-ahead)
            md = member.loc[d]
            nm = [t for t in nm if t in md.index and bool(md[t])]
        if len(nm) < min_n:
            pr.append(0.0); held.append(0); prev = set(); continue
        r = fwd.loc[d, nm].clip(-1.0, 1.5).mean()
        cur = set(nm); turn = len(cur ^ prev) / (2 * max(len(cur), 1))
        pr.append(float(r) - turn * 2 * cost_bps / 1e4 / hold); turns.append(turn); prev = cur
        held.append(len(nm))
    return pd.Series(pr, index=idx[:-hold]), (np.mean(turns) if turns else 0), (np.mean([h for h in held if h]) if any(held) else 0)


def stats(pr, ppy, bench=None):
    pr = pr.dropna()
    if len(pr) < max(12, ppy // 4):
        return {"n": int(len(pr))}
    eq = np.cumprod(1 + pr.values); yrs = len(pr) / ppy
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = pr.mean() / pr.std() * math.sqrt(ppy) if pr.std() > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100); hit = float((pr.values > 0).mean() * 100)
    o = {"n": int(len(pr)), "total_pct": float((eq[-1] - 1) * 100), "cagr_pct": float(cagr),
         "sharpe": float(sh), "maxdd_pct": dd, "hit_pct": hit}
    if bench is not None:
        b = bench.reindex(pr.index).fillna(0.0)
        o["vs_spy_pp"] = float(((eq[-1] - 1) - (np.cumprod(1 + b.values)[-1] - 1)) * 100)
    return o


res = {"params": {"low_win": LOW_WIN, "div_lag": DIV_LAG, "dvol_floor": DVOL_FLOOR, "price_floor": PRICE_FLOOR,
                  "start": START, "indicators": INDS,
                  "note": "same divergence rule (price N-bar low & indicator>lag) per money-flow indicator x timeframe; static cap/sector snapshot (caveat)"},
       "by_indicator": {}, "by_cap": {}, "cap_x_sector": {}, "spy_ref": {}}
for tfname, (rule, ppy) in TF.items():
    res["spy_ref"][tfname] = stats(panels[tfname]["spy_fwd1"], ppy)

print("\n=== money-flow bullish-divergence: hold1 ALL (GROSS -> NET25) + best BEAR pocket, per indicator x timeframe ===", flush=True)
print(f"SPY: daily CAGR {res['spy_ref']['daily'].get('cagr_pct',0):.1f}%/Sh{res['spy_ref']['daily'].get('sharpe',0):.2f} | monthly {res['spy_ref']['monthly'].get('cagr_pct',0):.1f}%/Sh{res['spy_ref']['monthly'].get('sharpe',0):.2f}", flush=True)
for ind in INDS:
    res["by_indicator"][ind] = {}
    print(f"\n## {ind.upper()}", flush=True)
    for tfname, (rule, ppy) in TF.items():
        P = panels[tfname]; res["by_indicator"][ind][tfname] = {"total_fires": int(P["fired"][ind].values.sum()), "holds": {}}
        for hold in (1, 2):
            res["by_indicator"][ind][tfname]["holds"][hold] = {}
            for regime in ("all", "bear", "bull"):
                cells = {}
                for cb in (0, 10, 25, 50):
                    s, tn, nm = run_book(P, ind, regime, hold, cb)
                    key = "gross" if cb == 0 else f"net_{cb}"
                    st = stats(s, ppy, bench=P["spy_fwd1"] if regime == "all" else None)
                    if cb == 0:
                        st["turn_pct"] = float(tn * 100); st["avg_names"] = float(nm)
                    cells[key] = st
                res["by_indicator"][ind][tfname]["holds"][hold][regime] = cells
        h1 = res["by_indicator"][ind][tfname]["holds"][1]["all"]
        bear2 = res["by_indicator"][ind][tfname]["holds"][2]["bear"]["net_25"]
        print(f"  {tfname:7} fires {res['by_indicator'][ind][tfname]['total_fires']:6}  "
              f"GROSS {h1['gross'].get('cagr_pct',0):+6.1f}%/Sh{h1['gross'].get('sharpe',0):4.2f} turn{h1['gross'].get('turn_pct',0):3.0f}%  "
              f"NET25 {h1['net_25'].get('cagr_pct',0):+6.1f}%/Sh{h1['net_25'].get('sharpe',0):4.2f} (vsSPY {h1['net_25'].get('vs_spy_pp',0):+.0f})  "
              f"| bear-h2 NET25 {bear2.get('cagr_pct',0):+5.1f}%/Sh{bear2.get('sharpe',0):4.2f}", flush=True)

# ---- cap bucket per indicator x timeframe (all regime, hold1, NET25) ----
print("\n=== by CAP bucket (all regime, hold1, NET25) ===", flush=True)
buckets = ["micro(<0.5B)", "small(<2B)", "large(>=2B)"]
by_cap_cols = {b: [t for t in panels['monthly']['CL'].columns if cap_bucket(t) == b] for b in buckets}
for ind in INDS:
    res["by_cap"][ind] = {}
    for tfname, (rule, ppy) in TF.items():
        P = panels[tfname]; res["by_cap"][ind][tfname] = {}
        for b in buckets:
            cols = by_cap_cols[b]
            if len(cols) < 20:
                continue
            s25, _, nm = run_book(P, ind, "all", 1, 25, cols=cols)
            st = stats(s25, ppy, bench=P["spy_fwd1"])
            res["by_cap"][ind][tfname][b] = {"n_names": len(cols), "net_25": st, "avg_fires": float(nm)}
    # print compact: best cap cell per indicator across timeframes
    best = None
    for tfname in TF:
        for b, cell in res["by_cap"][ind][tfname].items():
            sh = cell["net_25"].get("sharpe", -9)
            if best is None or sh > best[3]:
                best = (tfname, b, cell["net_25"].get("cagr_pct", 0), sh, cell["net_25"].get("vs_spy_pp", 0))
    if best:
        print(f"  {ind.upper():4} best cap cell: {best[0]}/{best[1]} NET25 {best[2]:+.1f}%/Sh{best[3]:.2f} vsSPY {best[4]:+.0f}pp", flush=True)

# ---- cap x sector grid for the best (indicator,timeframe) by all/hold1/net25 sharpe ----
best_it = max([(ind, tf) for ind in INDS for tf in TF],
              key=lambda it: res["by_indicator"][it[0]][it[1]]["holds"][1]["all"]["net_25"].get("sharpe", -9))
bi, btf = best_it
print(f"\n=== cap x SECTOR grid for BEST = {bi.upper()} / {btf} (all regime, hold1, NET25) ===", flush=True)
P = panels[btf]; ppy = TF[btf][1]
secs = sorted({fsec.get(t) for t in P['CL'].columns if fsec.get(t)})
for b in buckets:
    res["cap_x_sector"].setdefault(b, {})
    for sec in secs:
        cols = [t for t in P['CL'].columns if cap_bucket(t) == b and fsec.get(t) == sec]
        if len(cols) < 15:
            continue
        s25, _, nm = run_book(P, bi, "all", 1, 25, cols=cols)
        st = stats(s25, ppy, bench=P["spy_fwd1"])
        if st.get("n", 0) < ppy // 2 or nm < 2:
            continue
        res["cap_x_sector"][b][sec] = {"n_names": len(cols), "net_25": st, "avg_fires": float(nm)}
        print(f"  {b:12} {str(sec)[:22]:22} n={len(cols):4} NET25 {st.get('cagr_pct',0):+6.1f}%/Sh{st.get('sharpe',0):4.2f} vsSPY {st.get('vs_spy_pp',0):+.0f}pp", flush=True)

# ---- TOP-N by MARKET CAP sub-segment (user: "top 100 stock per marketcap"). Static current mktcap => LOOK-AHEAD
# caveat (today's biggest incl. names that grew in). Small pools => relax min-names to 3; report avg_fires so
# sparsity is visible (mega-cap divergences are rare). weekly+monthly only (daily is turnover-dead regardless). ----
ranked = sorted([t for t in panels['monthly']['CL'].columns if (mcap.get(t) or 0) > 0],
                key=lambda t: mcap.get(t) or 0, reverse=True)
print("\n=== TOP-N by market cap (all regime, hold1, min-names=3; static-cap LOOK-AHEAD caveat) ===", flush=True)
print(f"{'topN':>5} {'ind':>4} {'tf':>7} {'GROSS':>8} {'NET25':>8} {'Sh':>5} {'vsSPY':>7} {'avgFires':>8}", flush=True)
res["by_topn"] = {}
for N in (50, 100, 200):
    cols = ranked[:N]; res["by_topn"][N] = {}
    for ind in INDS:
        res["by_topn"][N][ind] = {}
        for tfname in ("weekly", "monthly"):
            P = panels[tfname]; ppy = TF[tfname][1]
            g, _, nm = run_book(P, ind, "all", 1, 0, cols=cols, min_n=3)
            n25, _, _ = run_book(P, ind, "all", 1, 25, cols=cols, min_n=3)
            sg = stats(g, ppy, bench=P['spy_fwd1']); s25 = stats(n25, ppy, bench=P['spy_fwd1'])
            res["by_topn"][N][ind][tfname] = {"gross": sg, "net_25": s25, "avg_fires": float(nm)}
            print(f"{N:>5} {ind:>4} {tfname:>7} {sg.get('cagr_pct',0):>+7.1f}% {s25.get('cagr_pct',0):>+7.1f}% "
                  f"{s25.get('sharpe',0):>5.2f} {s25.get('vs_spy_pp',0):>+6.0f} {nm:>8.1f}", flush=True)

# ---- PIT top-N by market cap (NO look-ahead): rank by PAST mktcap = as-traded close x PIT shares
# (FinancialReport.shares_outstanding ffilled by avail_date) at EACH month-end; a name counts as top-N only
# in the months it actually was. weekly/daily inherit the month's membership (ffill). ----
import price_basis
from collections import defaultdict
from core.models import FinancialReport
mtf = panels['monthly']; midx_m = mtf['idx']; mcols = list(mtf['CL'].columns)
frs = defaultdict(list)
for tk, ad, sh in FinancialReport.objects.filter(ticker__in=mcols, shares_outstanding__gt=0).values_list("ticker", "avail_date", "shares_outstanding"):
    if ad and sh:
        frs[tk].append((pd.Timestamp(ad).value, float(sh)))
mts = np.array([d.value for d in midx_m], dtype="int64")
shares_m = {}
for tk in mcols:
    pts = sorted(frs.get(tk, []))
    if not pts:
        continue
    di = np.array([t for t, _ in pts]); sv = np.array([v for _, v in pts]); o = np.full(len(midx_m), np.nan)
    for j, dd in enumerate(mts):
        b = np.searchsorted(di, dd, side="right")
        if b > 0:
            o[j] = sv[b - 1]
    shares_m[tk] = pd.Series(o, index=midx_m)
shares_m = pd.DataFrame(shares_m)
mcap_pit_m = price_basis.as_traded_close(mtf['CL'][shares_m.columns]) * shares_m
cov = float(mcap_pit_m.notna().any().mean())
print(f"\n=== PIT top-N by market cap (rank by PAST mktcap each month; {mcap_pit_m.shape[1]} names w/ shares, {cov*100:.0f}% ever-covered) ===", flush=True)
print(f"{'topN':>5} {'ind':>4} {'tf':>7} {'GROSS':>8} {'NET25':>8} {'Sh':>5} {'vsSPY':>7} {'avgFires':>8}", flush=True)


def pit_member(N, idx):
    mask_m = pd.DataFrame(False, index=midx_m, columns=mcap_pit_m.columns)
    for d in midx_m:
        row = mcap_pit_m.loc[d].dropna()
        if len(row):
            mask_m.loc[d, row.nlargest(N).index] = True
    return mask_m if idx.equals(midx_m) else mask_m.reindex(idx, method="ffill")


res["by_topn_pit"] = {}
for N in (100, 200):
    res["by_topn_pit"][N] = {}
    for ind in INDS:
        res["by_topn_pit"][N][ind] = {}
        for tfname in ("weekly", "monthly"):
            P = panels[tfname]; ppy = TF[tfname][1]; mem = pit_member(N, P["idx"])
            g, _, nm = run_book(P, ind, "all", 1, 0, min_n=3, member=mem)
            n25, _, _ = run_book(P, ind, "all", 1, 25, min_n=3, member=mem)
            sg = stats(g, ppy, bench=P['spy_fwd1']); s25 = stats(n25, ppy, bench=P['spy_fwd1'])
            res["by_topn_pit"][N][ind][tfname] = {"gross": sg, "net_25": s25, "avg_fires": float(nm)}
            print(f"{N:>5} {ind:>4} {tfname:>7} {sg.get('cagr_pct',0):>+7.1f}% {s25.get('cagr_pct',0):>+7.1f}% "
                  f"{s25.get('sharpe',0):>5.2f} {s25.get('vs_spy_pp',0):>+6.0f} {nm:>8.1f}", flush=True)

res["best_indicator_timeframe"] = {"indicator": bi, "timeframe": btf}
Path(OUT).write_text(json.dumps(res, indent=2, default=float))
print(f"\nwrote {OUT}", flush=True)
try:
    from core.models import BacktestResult
    from django.utils import timezone
    BacktestResult.objects.update_or_create(kind="moneyflow_divergence_timeframes",
        defaults={"payload": res, "computed_at": timezone.now()})
    print("Saved BacktestResult[moneyflow_divergence_timeframes]", flush=True)
except Exception as e:
    print("save skipped:", e, flush=True)
