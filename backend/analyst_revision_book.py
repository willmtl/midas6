#!/usr/bin/env python3
"""ANALYST-REVISION MOMENTUM BOOK (standalone, NOT the flagship). Hypothesis: away from the flagship's contrarian
deep-value edge, analyst REVISION momentum (targets being RAISED / net UPGRADES) predicts continued outperformance
(post-revision drift). Long-only momentum book: monthly, among liquid analyst-covered names, LONG the top quintile
by each signal, EW, hold 1 month. Now testable because the backfill made coverage ~complete (5,099 tk, 2012+).

GATE vs the 'sentiment = beta not alpha' prior: report (1) long book vs SPY on RETURN, (2) LONG-SHORT spread
(top−bottom quintile, market-neutral → isolates alpha from beta), (3) BULL vs BEAR month split (beta shows up as
all-return-in-bull). Segmented by cap bucket. PIT: signals use only ratings dated <= month-end; fwd ret = next
month close/close. Saves BacktestResult[analyst_revision_book] + JSON.
"""
import os, json, math, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles

ANALYST = "/app/.data/analyst_ratings.jsonl"
MIN_DVOL = 5e6
OUT = "/app/.data/studies/analyst_revision_book.json"


def main():
    # ---- analyst events per ticker: (ts, target, action) ----
    tgt = defaultdict(list)     # ticker -> [(Timestamp, target)]
    act = defaultdict(list)     # ticker -> [(Timestamp, +1 upgrade / -1 downgrade)]
    for line in Path(ANALYST).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        tk, d = r.get("ticker"), r.get("date")
        if not tk or not d:
            continue
        ts = pd.Timestamp(d)
        if r.get("price_target"):
            try:
                tgt[tk].append((ts, float(r["price_target"])))
            except (TypeError, ValueError):
                pass
        a = (r.get("rating_action") or "").lower()
        if "upgrade" in a:
            act[tk].append((ts, 1))
        elif "downgrade" in a:
            act[tk].append((ts, -1))
    for tk in tgt:
        tgt[tk].sort()
    for tk in act:
        act[tk].sort()
    covered = sorted(set(tgt) | set(act))
    print(f"analyst events: {len(covered)} tickers", flush=True)

    # ---- candles (batched, parallel-off) restricted to covered names that have price history ----
    from django.db import connection
    from core.models import Candle, Fundamental
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    havec = set(Candle.objects.filter(ticker__in=covered, interval="1d")
                .values_list("ticker", flat=True).distinct())
    uni = sorted(set(covered) & havec)
    print(f"covered with candles: {len(uni)}", flush=True)
    cand = {}
    for i in range(0, len(uni), 40):
        cand.update(load_candles(uni[i:i + 40]))
    # market cap (static snapshot) for cap-bucket segmentation (caveat: not PIT)
    mcap = dict(Fundamental.objects.filter(ticker__in=uni).values_list("ticker", "market_cap"))

    # SPY
    spy = load_candles(["SPY"]).get("SPY")
    spy_m = spy["Close"].resample("ME").last()
    midx = spy_m.index[(spy_m.index >= "2016-01-01")]
    spy_ret = spy_m.reindex(midx).pct_change().shift(-1)          # fwd 1-mo SPY return, aligned to selection month
    spy200 = spy["Close"].rolling(200).mean().resample("ME").last().reindex(midx)
    bull = (spy_m.reindex(midx) >= spy200)                        # PIT bull/bear at each month-end

    # ---- per-name monthly panels: fwd return, dollar-vol, and the signals ----
    close_m, fret, dvol = {}, {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        c = df["Close"].resample("ME").last().reindex(midx)
        v = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)   # ~avg daily $vol in the month
        close_m[tk] = c
        fret[tk] = c.pct_change().shift(-1)                        # next-month close/close
        dvol[tk] = v
    close_m = pd.DataFrame(close_m); fret = pd.DataFrame(fret); dvol = pd.DataFrame(dvol)
    tickers = list(close_m.columns)

    midx_ts = np.array([d.value for d in midx], dtype="int64")

    def consensus_target(tk, win_days=180):
        pts = tgt.get(tk)
        if not pts:
            return pd.Series(np.nan, index=midx)
        di = np.array([t.value for t, _ in pts]); tv = np.array([v for _, v in pts])
        win = win_days * 86400 * 10**9
        out = np.full(len(midx), np.nan)
        for j, d in enumerate(midx_ts):
            a = np.searchsorted(di, d - win, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                out[j] = np.median(tv[a:b])
        return pd.Series(out, index=midx)

    def net_upg(tk, win_days=90):
        pts = act.get(tk)
        if not pts:
            return pd.Series(np.nan, index=midx)
        di = np.array([t.value for t, _ in pts]); av = np.array([v for _, v in pts])
        win = win_days * 86400 * 10**9
        out = np.full(len(midx), np.nan)
        for j, d in enumerate(midx_ts):
            a = np.searchsorted(di, d - win, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                out[j] = av[a:b].sum()
        return pd.Series(out, index=midx)

    ctar = pd.DataFrame({tk: consensus_target(tk) for tk in tickers})
    nupg = pd.DataFrame({tk: net_upg(tk) for tk in tickers})
    # signals
    sig = {}
    sig["tgt_rev_3m"] = (ctar - ctar.shift(3)) / close_m.where(close_m > 0)     # target revision momentum
    sig["net_upg_90d"] = nupg                                                    # net upgrades − downgrades
    sig["upside_level"] = ctar / close_m.where(close_m > 0) - 1                  # LEVEL control
    sig["price_mom_6m"] = close_m / close_m.shift(6) - 1                          # price-momentum benchmark

    liquid = dvol >= MIN_DVOL

    def book(signal, q=0.2, leg="long"):
        rets = []
        for i, d in enumerate(midx[:-1]):
            s = signal.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]
            m = s.notna() & fr.notna() & lq
            names = s[m]
            if len(names) < 20:
                rets.append(np.nan); continue
            k = max(1, int(len(names) * q))
            top = names.nlargest(k).index; bot = names.nsmallest(k).index
            rt = fr[top].mean(); rb = fr[bot].mean()
            rets.append(rt if leg == "long" else (rt - rb) if leg == "ls" else rb)
        return pd.Series(rets, index=midx[:-1])

    def stats(rets, bench=None):
        r = rets.dropna()
        if len(r) < 12:
            return {"n": len(r)}
        eq = (1 + r).prod(); yrs = len(r) / 12.0
        cagr = eq ** (1 / yrs) - 1 if eq > 0 else -1
        sh = r.mean() / r.std() * math.sqrt(12) if r.std() > 0 else 0
        dd = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        out = {"n": int(len(r)), "total_pct": float((eq - 1) * 100), "cagr_pct": float(cagr * 100),
               "sharpe": float(sh), "maxdd_pct": float(dd * 100)}
        if bench is not None:
            b = bench.reindex(r.index)
            out["vs_spy_pp"] = float(((eq - 1) - ((1 + b).prod() - 1)) * 100)
            # bull/bear split (beta check)
            bl = bull.reindex(r.index).fillna(True)
            out["bull_mean_pct"] = float(r[bl].mean() * 100); out["bear_mean_pct"] = float(r[~bl].mean() * 100)
            out["bull_spy_pct"] = float(b[bl].mean() * 100); out["bear_spy_pct"] = float(b[~bl].mean() * 100)
        return out

    spyv = spy_ret.reindex(midx[:-1])
    print(f"\n=== ANALYST-REVISION BOOK (top-quintile long, monthly, {midx[0].date()}..{midx[-1].date()}) ===", flush=True)
    print(f"SPY same window: total {((1+spyv.dropna()).prod()-1)*100:.0f}%", flush=True)
    print(f"{'signal':>14s} {'LONG tot':>9s} {'CAGR':>7s} {'Sh':>5s} {'vsSPY':>8s} {'bull/bear ret':>16s} {'L-S tot':>8s} {'L-S Sh':>6s}", flush=True)
    res = {"spy_total_pct": float(((1 + spyv.dropna()).prod() - 1) * 100), "window": [str(midx[0].date()), str(midx[-1].date())], "signals": {}}
    for name, s in sig.items():
        lo = stats(book(s, leg="long"), bench=spyv)
        ls = stats(book(s, leg="ls"))
        res["signals"][name] = {"long": lo, "long_short": ls}
        print(f"{name:>14s} {lo.get('total_pct',0):>8.0f}% {lo.get('cagr_pct',0):>6.1f}% {lo.get('sharpe',0):>5.2f} "
              f"{lo.get('vs_spy_pp',0):>+7.0f} {lo.get('bull_mean_pct',0):>+6.1f}/{lo.get('bear_mean_pct',0):>+5.1f}%  "
              f"{ls.get('total_pct',0):>7.0f}% {ls.get('sharpe',0):>6.2f}", flush=True)

    # cap-bucket segmentation for the best revision signal
    def cap_bucket(tk):
        mc = mcap.get(tk)
        if mc is None or mc <= 0: return "unknown"
        return "micro(<0.5B)" if mc < 5e8 else "small(<2B)" if mc < 2e9 else "large(>=2B)"
    print("\n=== long book by CAP bucket (tgt_rev_3m & upside_level) ===", flush=True)
    res["by_cap"] = {}
    for sg in ["tgt_rev_3m", "upside_level"]:
        res["by_cap"][sg] = {}
        for bkt in ["micro(<0.5B)", "small(<2B)", "large(>=2B)"]:
            cols = [t for t in tickers if cap_bucket(t) == bkt]
            if len(cols) < 30:
                continue
            st = stats(book(sig[sg][cols], leg="long"), bench=spyv)
            res["by_cap"][sg][bkt] = st
            print(f"  {sg:>12s} {bkt:>14s} n={len(cols):>4d}  total {st.get('total_pct',0):>7.0f}%  CAGR {st.get('cagr_pct',0):>5.1f}%  Sh {st.get('sharpe',0):>5.2f}  vsSPY {st.get('vs_spy_pp',0):>+6.0f}", flush=True)

    # ---- ROBUSTNESS on the LARGE-CAP book: both-halves + transaction-cost haircut ----
    large = [t for t in tickers if cap_bucket(t) == "large(>=2B)"]
    print(f"\n=== LARGE-CAP robustness (n={len(large)}) — both halves + costs ===", flush=True)
    res["largecap_robust"] = {}
    def costed(signal, q=0.2, bps=20.0):
        rets, prev = [], set()
        for i, d in enumerate(midx[:-1]):
            s = signal.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]
            m = s.notna() & fr.notna() & lq; names = s[m]
            if len(names) < 20:
                rets.append(np.nan); prev = set(); continue
            k = max(1, int(len(names) * q)); top = set(names.nlargest(k).index)
            turn = len(top ^ prev) / max(1, len(top))
            rets.append(fr[list(top)].mean() - (bps / 1e4) * turn); prev = top
        return pd.Series(rets, index=midx[:-1])
    for sg in ["tgt_rev_3m", "upside_level"]:
        s = sig[sg][large]
        full = book(s, leg="long"); h1 = full[full.index < "2021-01-01"]; h2 = full[full.index >= "2021-01-01"]
        cst = costed(s, bps=20.0)
        st_full = stats(full, bench=spyv); st1 = stats(h1); st2 = stats(h2); st_c = stats(cst, bench=spyv)
        res["largecap_robust"][sg] = {"full": st_full, "h1_2016_2020": st1, "h2_2021_2026": st2, "costed_20bps": st_c}
        print(f"  {sg:>12s} FULL Sh {st_full.get('sharpe',0):.2f} ({st_full.get('cagr_pct',0):.0f}%CAGR) | "
              f"H1 Sh {st1.get('sharpe',0):.2f} | H2 Sh {st2.get('sharpe',0):.2f} | "
              f"COSTED(20bps) Sh {st_c.get('sharpe',0):.2f} ({st_c.get('cagr_pct',0):.0f}%CAGR, vsSPY {st_c.get('vs_spy_pp',0):+.0f})", flush=True)

    # ---- RSI(14)-CROSS-AVERAGE overlay on the large-cap revision book (user): add technical WHEN to the
    # fundamental WHAT. RSI(14) vs its SMA(14); state = RSI>SMA at month-end; cross = crossed ABOVE in the
    # trailing window (daily last-10d / weekly last-4wk). Gate the top-revision pool by the RSI condition. ----
    def _rsi(c, n=14):
        d = c.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
        ru = up.ewm(alpha=1.0 / n, adjust=False).mean(); rd = dn.ewm(alpha=1.0 / n, adjust=False).mean()
        return 100 - 100 / (1 + ru / rd.replace(0, np.nan))
    st_d, cr_d, st_w, cr_w = {}, {}, {}, {}
    for tk in large:
        df = cand.get(tk)
        if df is None or df.empty:
            continue
        c = df["Close"]; r = _rsi(c); sm = r.rolling(14).mean(); ab = r > sm
        cru = ab & (~ab.shift(1).fillna(False))
        st_d[tk] = ab.resample("ME").last().reindex(midx)
        cr_d[tk] = (cru.rolling(10).max() > 0).resample("ME").last().reindex(midx)
        cw = c.resample("W").last(); rw = _rsi(cw); smw = rw.rolling(14).mean(); abw = rw > smw
        crw = abw & (~abw.shift(1).fillna(False))
        st_w[tk] = abw.resample("ME").last().reindex(midx)
        cr_w[tk] = (crw.rolling(4).max() > 0).resample("ME").last().reindex(midx)
    st_d = pd.DataFrame(st_d).reindex(columns=large); cr_d = pd.DataFrame(cr_d).reindex(columns=large)
    st_w = pd.DataFrame(st_w).reindex(columns=large); cr_w = pd.DataFrame(cr_w).reindex(columns=large)

    def gated(sigdf, gatedf, q=0.2, bps=20.0):
        rets, prev = [], set()
        for i, d in enumerate(midx[:-1]):
            s = sigdf.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]
            m = s.notna() & fr.notna() & lq
            if gatedf is not None:
                m = m & gatedf.loc[d].fillna(False)
            names = s[m]
            if len(names) < 10:
                rets.append(np.nan); prev = set(); continue
            k = max(1, int(len(names) * q)); top = set(names.nlargest(k).index)
            turn = len(top ^ prev) / max(1, len(top)); rets.append(fr[list(top)].mean() - (bps / 1e4) * turn); prev = top
        return pd.Series(rets, index=midx[:-1])

    rev = sig["tgt_rev_3m"][large]
    print("\n=== LARGE-CAP revision × RSI(14)-cross-average (costed 20bps) ===", flush=True)
    print(f"{'arm':>26s} {'total':>8s} {'CAGR':>6s} {'Sh':>5s} {'vsSPY':>7s} {'avgN':>5s}", flush=True)
    res["rsi_mix"] = {}
    arms = [("revision only", None), ("rev × RSI14>SMA daily", st_d), ("rev × RSI14 cross daily", cr_d),
            ("rev × RSI14>SMA weekly", st_w), ("rev × RSI14 cross weekly", cr_w)]
    for lab, g in arms:
        st = stats(gated(rev, g), bench=spyv)
        # avg names selected/mo (diagnostic on how much the gate thins the pool)
        res["rsi_mix"][lab] = st
        print(f"{lab:>26s} {st.get('total_pct',0):>7.0f}% {st.get('cagr_pct',0):>5.1f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f}", flush=True)
    # RSI state STANDALONE (no revision): long ALL large-caps in RSI-bullish state, EW — control
    def state_book(gatedf, bps=20.0):
        rets, prev = [], set()
        for i, d in enumerate(midx[:-1]):
            fr = fret.loc[d]; lq = liquid.loc[d]; g = gatedf.loc[d].fillna(False)
            names = fr[g & fr.notna() & lq].index
            if len(names) < 10:
                rets.append(np.nan); prev = set(); continue
            cur = set(names); turn = len(cur ^ prev) / max(1, len(cur)); rets.append(fr[list(cur)].mean() - (bps / 1e4) * turn); prev = cur
        return pd.Series(rets, index=midx[:-1])
    for lab, g in [("RSI14>SMA daily STANDALONE", st_d), ("RSI14>SMA weekly STANDALONE", st_w)]:
        st = stats(state_book(g), bench=spyv); res["rsi_mix"][lab] = st
        print(f"{lab:>26s} {st.get('total_pct',0):>7.0f}% {st.get('cagr_pct',0):>5.1f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f}", flush=True)

    # ---- CHOPPY-vs-TRENDING test (user hypothesis: RSI(14)-cross whipsaws in sideways tape, only pays in trends).
    # ADX(14) trend strength per name; split large-caps into TRENDING (ADX>25) / CHOPPY (ADX<20) at each month-end
    # (PIT). Within each regime compare revision-only vs revision×RSI-cross — if the cross HELPS in trend and HURTS
    # in chop, the hypothesis holds and a trend-gated RSI-cross could add. ----
    def _adx(df, n=14):
        h, l, c = df["High"], df["Low"], df["Close"]
        up = h.diff(); dn = -l.diff()
        pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
        pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean() / atr
        mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean() / atr
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        return dx.ewm(alpha=1.0 / n, adjust=False).mean()
    adxm = {}
    for tk in large:
        df = cand.get(tk)
        if df is None or df.empty:
            continue
        adxm[tk] = _adx(df).resample("ME").last().reindex(midx)
    adxm = pd.DataFrame(adxm).reindex(columns=large)
    trending = adxm > 25; choppy = adxm < 20
    print("\n=== CHOPPY vs TRENDING: does RSI(14)-cross add WITHIN a trend regime? (ADX14, costed 20bps) ===", flush=True)
    print(f"{'regime':>10s} {'book':>16s} {'total':>8s} {'CAGR':>6s} {'Sh':>5s} {'vsSPY':>7s}", flush=True)
    res["chop_trend"] = {}
    for rlab, reg in [("TREND(>25)", trending), ("CHOP(<20)", choppy)]:
        base_r = stats(gated(rev, reg), bench=spyv)                       # revision-only within regime
        rsi_r = stats(gated(rev, reg & cr_d), bench=spyv)                 # + RSI-cross within regime
        res["chop_trend"][rlab] = {"rev_only": base_r, "rev_x_rsicross": rsi_r}
        for blab, st in [("rev only", base_r), ("rev × RSIcross", rsi_r)]:
            print(f"{rlab:>10s} {blab:>16s} {st.get('total_pct',0):>7.0f}% {st.get('cagr_pct',0):>5.1f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f}", flush=True)
        d_sh = rsi_r.get('sharpe', 0) - base_r.get('sharpe', 0)
        print(f"           -> RSI-cross ΔSharpe within {rlab}: {d_sh:+.2f}", flush=True)

    # ---- FAST trend classifier (user: ADX too laggy). Kaufman Efficiency Ratio ER = |net move| / total path over
    # 10d (no smoothing lag). Re-test RSI-cross within a RESPONSIVE trend/chop split; also a faster 5-day cross. ----
    def _er(c, n=10):
        return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum().replace(0, np.nan)
    erm, cr5 = {}, {}
    for tk in large:
        df = cand.get(tk)
        if df is None or df.empty:
            continue
        c = df["Close"]
        erm[tk] = _er(c).resample("ME").last().reindex(midx)
        r = _rsi(c); sm = r.rolling(14).mean(); ab = r > sm
        cru = ab & (~ab.shift(1).fillna(False))
        cr5[tk] = (cru.rolling(5).max() > 0).resample("ME").last().reindex(midx)      # cross in last 5 trading days
    erm = pd.DataFrame(erm).reindex(columns=large); cr5 = pd.DataFrame(cr5).reindex(columns=large)
    er_trend = erm > 0.5; er_chop = erm < 0.3
    print("\n=== FAST trend split (Kaufman ER10) — does RSI-cross add within a RESPONSIVE regime? (costed 20bps) ===", flush=True)
    print(f"{'regime':>12s} {'book':>18s} {'total':>8s} {'Sh':>5s} {'vsSPY':>7s}", flush=True)
    res["er_trend"] = {}
    for rlab, reg in [("ER-TREND>0.5", er_trend), ("ER-CHOP<0.3", er_chop)]:
        b0 = stats(gated(rev, reg), bench=spyv)
        b1 = stats(gated(rev, reg & cr_d), bench=spyv)          # + RSI cross (10d)
        b2 = stats(gated(rev, reg & cr5), bench=spyv)           # + RSI cross (fast 5d)
        res["er_trend"][rlab] = {"rev_only": b0, "rev_x_cross10": b1, "rev_x_cross5": b2}
        for blab, st in [("rev only", b0), ("× RSIcross10", b1), ("× RSIcross5", b2)]:
            print(f"{rlab:>12s} {blab:>18s} {st.get('total_pct',0):>7.0f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f}", flush=True)
        print(f"             -> ΔSharpe cross10 {b1.get('sharpe',0)-b0.get('sharpe',0):+.2f} | cross5 {b2.get('sharpe',0)-b0.get('sharpe',0):+.2f}", flush=True)

    # ---- REVERSAL test (user: 'it's trend reversal'). If the revision edge is analysts turning positive on FALLING
    # names (catching the turn), it should concentrate where price was DOWN / below trend at purchase, NOT where it
    # was already rising. Split the large-cap revision book by prior 3mo momentum sign and by below/above 200d-MA. ----
    clg = close_m[large]
    mom3 = (clg / clg.shift(3) - 1)
    below200 = {}
    for tk in large:
        df = cand.get(tk)
        if df is None or df.empty:
            continue
        ma = df["Close"].rolling(200).mean().resample("ME").last().reindex(midx)
        px = df["Close"].resample("ME").last().reindex(midx)
        below200[tk] = px < ma
    below200 = pd.DataFrame(below200).reindex(columns=large)
    print("\n=== REVERSAL vs CONTINUATION: where does the revision edge live? (large-cap, costed 20bps) ===", flush=True)
    print(f"{'split':>26s} {'total':>8s} {'CAGR':>6s} {'Sh':>5s} {'vsSPY':>7s}", flush=True)
    res["reversal"] = {}
    splits = [("prior 3mo DOWN (reversal)", mom3 < 0), ("prior 3mo UP (continuation)", mom3 >= 0),
              ("below 200d-MA (fallen)", below200), ("above 200d-MA (rising)", ~below200.fillna(False))]
    for lab, sp in splits:
        st = stats(gated(rev, sp), bench=spyv)
        res["reversal"][lab] = st
        print(f"{lab:>26s} {st.get('total_pct',0):>7.0f}% {st.get('cagr_pct',0):>5.1f}% {st.get('sharpe',0):>5.2f} {st.get('vs_spy_pp',0):>+6.0f}", flush=True)
    # explicit REVERSAL book: rising targets AND was falling (below-MA), top-quintile by revision
    rb = stats(gated(rev, below200 & (mom3 < 0)), bench=spyv)
    res["reversal"]["REVERSAL book (below-MA & 3mo-down & top-rev)"] = rb
    print(f"{'REVERSAL book (fallen & rising-tgt)':>26s} {rb.get('total_pct',0):>7.0f}% {rb.get('cagr_pct',0):>5.1f}% {rb.get('sharpe',0):>5.2f} {rb.get('vs_spy_pp',0):>+6.0f}", flush=True)

    # ---- SUCCESS / WIN RATE (user) — position hit-rate, beat-SPY rate, monthly beat-rate, win/loss payoff ----
    def winrate(sigdf, gatedf, q=0.2):
        pos, exc, mo, mospy = [], [], [], []
        for i, d in enumerate(midx[:-1]):
            s = sigdf.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]
            m = s.notna() & fr.notna() & lq
            if gatedf is not None:
                m = m & gatedf.loc[d].fillna(False)
            names = s[m]
            if len(names) < 10:
                continue
            k = max(1, int(len(names) * q)); top = names.nlargest(k).index
            rr = fr[top]; sp = float(spyv.loc[d]) if d in spyv.index and pd.notna(spyv.loc[d]) else np.nan
            pos.extend(rr.values.tolist()); exc.extend((rr - sp).values.tolist())
            mo.append(float(rr.mean())); mospy.append(sp)
        pos = np.array(pos); exc = np.array(exc); mo = np.array(mo); mospy = np.array(mospy)
        w = pos[pos > 0]; l = pos[pos < 0]
        return {"n_positions": int(len(pos)), "avg_picks_per_mo": round(len(pos) / max(1, len(mo)), 1),
                "pos_winrate": round(float((pos > 0).mean()) * 100, 1),
                "pos_beat_spy": round(float(np.nanmean(exc > 0)) * 100, 1),
                "avg_win_pct": round(float(w.mean()) * 100, 2), "avg_loss_pct": round(float(l.mean()) * 100, 2),
                "win_loss_ratio": round(float(w.mean() / abs(l.mean())), 2),
                "avg_pos_ret_pct": round(float(pos.mean()) * 100, 2),
                "n_months": int(len(mo)), "month_winrate": round(float((mo > 0).mean()) * 100, 1),
                "month_beat_spy": round(float(np.nanmean(mo > mospy)) * 100, 1)}
    print("\n=== SUCCESS / WIN RATE ===", flush=True)
    res["winrate"] = {}
    for lab, g in [("revision (all large-cap)", None), ("revision × above-200d-MA", ~below200.fillna(False))]:
        wr = winrate(rev, g); res["winrate"][lab] = wr
        print(f"  {lab}:", flush=True)
        print(f"    positions: {wr['n_positions']} (~{wr['avg_picks_per_mo']}/mo) | POSITION WIN RATE {wr['pos_winrate']}% | beat-SPY {wr['pos_beat_spy']}%", flush=True)
        print(f"    payoff: avg win {wr['avg_win_pct']}% / avg loss {wr['avg_loss_pct']}% = {wr['win_loss_ratio']}x | avg pick {wr['avg_pos_ret_pct']}%/mo", flush=True)
        print(f"    monthly ({wr['n_months']}mo): book positive {wr['month_winrate']}% | book beat SPY {wr['month_beat_spy']}%", flush=True)

    # ---- TAKE-PROFIT / STOP-LOSS experiment (user) on the base large-cap revision book. For each monthly pick,
    # walk its INTRA-MONTH daily closes: exit at the first daily close that hits +TP or −SL (relative to buy),
    # else hold to month-end. SL-priority if same day. Tests whether capping winners / cutting losers helps a
    # payoff-skew momentum book. ----
    month_data = []
    for i, d in enumerate(midx[:-1]):
        d1 = midx[i + 1]; s = rev.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]
        m = s.notna() & fr.notna() & lq; names = s[m]
        if len(names) < 10:
            month_data.append(None); continue
        k = max(1, int(len(names) * 0.2)); top = list(names.nlargest(k).index); paths = []
        for tk in top:
            df = cand.get(tk); p0 = close_m[tk].loc[d] if tk in close_m.columns else np.nan
            if df is None or not (p0 > 0):
                paths.append((np.array([float(fr[tk])]), float(fr[tk]))); continue
            w = df["Close"][(df["Close"].index > d) & (df["Close"].index <= d1)]
            pr = (w.values / p0 - 1) if len(w) else np.array([float(fr[tk])])
            paths.append((pr, float(pr[-1])))
        month_data.append((d, paths, top))

    def tpsl_book(tp, sl, bps=20.0):
        rets, allpos, prev = [], [], set()
        for md in month_data:
            if md is None:
                rets.append(np.nan); prev = set(); continue
            d, paths, top = md; rr = []
            for pr, base in paths:
                ex = base
                for r in pr:
                    if sl is not None and r <= sl:
                        ex = r; break
                    if tp is not None and r >= tp:
                        ex = r; break
                rr.append(ex)
            allpos.extend(rr); turn = len(set(top) ^ prev) / max(1, len(top)); prev = set(top)
            rets.append(float(np.mean(rr)) - (bps / 1e4) * turn)
        ser = pd.Series(rets, index=midx[:-1]); ap = np.array(allpos)
        st = stats(ser, bench=spyv); st["pos_winrate"] = round(float((ap > 0).mean()) * 100, 1)
        st["avg_pos_pct"] = round(float(ap.mean()) * 100, 2)
        return st

    print("\n=== TAKE-PROFIT / STOP-LOSS on the large-cap revision book (intra-month daily, costed 20bps) ===", flush=True)
    print(f"{'arm':>14s} {'total':>8s} {'CAGR':>6s} {'Sh':>5s} {'vsSPY':>7s} {'posWin':>7s} {'avgPos':>7s}", flush=True)
    res["tpsl"] = {}
    arms = [("baseline", None, None), ("SL -8%", None, -0.08), ("SL -10%", None, -0.10),
            ("SL -15%", None, -0.15), ("SL -20%", None, -0.20), ("TP +15%", 0.15, None),
            ("TP +20%", 0.20, None), ("TP +30%", 0.30, None), ("TP20/SL10", 0.20, -0.10),
            ("TP30/SL15", 0.30, -0.15)]
    for lab, tp, sl in arms:
        st = tpsl_book(tp, sl); res["tpsl"][lab] = st
        print(f"{lab:>14s} {st.get('total_pct',0):>7.0f}% {st.get('cagr_pct',0):>5.1f}% {st.get('sharpe',0):>5.2f} "
              f"{st.get('vs_spy_pp',0):>+6.0f} {st.get('pos_winrate',0):>6.1f}% {st.get('avg_pos_pct',0):>+6.2f}%", flush=True)

    Path(OUT).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT}", flush=True)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="analyst_revision_book",
            defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[analyst_revision_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
