#!/usr/bin/env python3
"""ANALYST TARGET-RAISE BREADTH — the historically-available cousin of book #2 (tgt_rev_3m = target MAGNITUDE).
analyst_ratings.jsonl carries a per-record `price_target_action` in {raises,lowers,maintains,...}; analysts nudge
targets far more often than they formally upgrade, so target-raise BREADTH is a higher-frequency, magnitude-
independent signal (fires on broad small raises, and sidesteps the split-adjustment landmine entirely — direction,
not level). Q: does breadth beat / add to the magnitude signal it already ships?

Signals (PIT, trailing 90d, records dated <= month-end):
  mag_tgt_rev_3m : (median consensus target now − 3mo ago)/price  <- book #2's shipped signal (baseline)
  raise_ratio    : raises / (raises + lowers)                       <- pure breadth (direction)
  net_raise_cnt  : (raises − lowers) count                          <- breadth weighted by coverage
  net_analysts   : (#distinct analysts raising − #lowering)         <- de-duplicated breadth
  composite      : z(mag) + z(raise_ratio) cross-sectional          <- additivity test

Long top-quintile, monthly, EW, hold 1mo, $5M dvol floor, costed 20bps/side. PIT cap buckets (as-traded close ×
PIT shares). Reports long vs SPY, LONG-SHORT spread (beta-neutral alpha), BULL/BEAR split, both halves, per bucket.
Saves BacktestResult[analyst_breadth_book]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/analyst_breadth_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
import price_basis
from seq_fundamental_study import load_candles

ANALYST = "/app/.data/analyst_ratings.jsonl"
MIN_DVOL = 5e6; COST_BPS = 20.0; WIN_D = 90
BUCKETS = [("large", 10e9, np.inf), ("mid", 2e9, 10e9), ("small", 0.0, 2e9)]


def main():
    tgt = defaultdict(list)      # ticker -> [(ts, median-target-source: (ts, target))]
    acts = defaultdict(list)     # ticker -> [(ts, +1 raise / -1 lower, analyst)]
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
        pt = r.get("price_target")
        if pt:
            try:
                tgt[tk].append((ts.value, float(pt)))
            except (TypeError, ValueError):
                pass
        a = (r.get("price_target_action") or "").lower()
        if a in ("raises", "lowers"):
            acts[tk].append((ts.value, 1 if a == "raises" else -1, r.get("analyst") or ""))
    for tk in tgt:
        tgt[tk].sort()
    for tk in acts:
        acts[tk].sort(key=lambda x: x[0])
    covered = sorted(set(tgt) | set(acts))
    print(f"analyst target events: {len(covered)} tickers", flush=True)

    from django.db import connection
    from core.models import Candle, FinancialReport
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    havec = set(Candle.objects.filter(ticker__in=covered, interval="1d").values_list("ticker", flat=True).distinct())
    uni = sorted(set(covered) & havec)
    print(f"covered with candles: {len(uni)}", flush=True)
    cand = {}
    for i in range(0, len(uni), 40):
        cand.update(load_candles(uni[i:i + 40]))

    spy = load_candles(["SPY"]).get("SPY")
    spy_m = spy["Close"].resample("ME").last()
    midx = spy_m.index[spy_m.index >= "2016-01-01"]
    midx_ts = np.array([d.value for d in midx], dtype="int64")
    spy_fwd = spy_m.reindex(midx).pct_change().shift(-1)
    spy200 = spy["Close"].rolling(200).mean().resample("ME").last().reindex(midx)
    bull = (spy_m.reindex(midx) >= spy200)

    close_m, dvol = {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        close_m[tk] = df["Close"].resample("ME").last().reindex(midx)
        dvol[tk] = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol = pd.DataFrame(dvol)
    tickers = list(close_m.columns)
    fret = close_m.pct_change().shift(-1)
    win = WIN_D * 86400 * 10**9

    def consensus_target(tk, win_days=180):
        pts = tgt.get(tk)
        if not pts:
            return pd.Series(np.nan, index=midx)
        di = np.array([t for t, _ in pts]); tv = np.array([v for _, v in pts]); w = win_days * 86400 * 10**9
        out = np.full(len(midx), np.nan)
        for j, d in enumerate(midx_ts):
            a = np.searchsorted(di, d - w, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                out[j] = np.median(tv[a:b])
        return pd.Series(out, index=midx)

    def breadth(tk):
        """trailing-90d raises/lowers -> (raise_ratio, net_raise_cnt, net_distinct_analysts)."""
        pts = acts.get(tk)
        rr = np.full(len(midx), np.nan); nc = np.full(len(midx), np.nan); na = np.full(len(midx), np.nan)
        if not pts:
            return rr, nc, na
        di = np.array([t for t, _, _ in pts]); sv = np.array([s for _, s, _ in pts]); an = [a for _, _, a in pts]
        for j, d in enumerate(midx_ts):
            a = np.searchsorted(di, d - win, side="right"); b = np.searchsorted(di, d, side="right")
            if b > a:
                seg = sv[a:b]; nr = int((seg > 0).sum()); nl = int((seg < 0).sum())
                if nr + nl > 0:
                    rr[j] = nr / (nr + nl)
                nc[j] = nr - nl
                ras = {an[k] for k in range(a, b) if sv[k] > 0}; las = {an[k] for k in range(a, b) if sv[k] < 0}
                na[j] = len(ras) - len(las)
        return rr, nc, na

    ctar = pd.DataFrame({tk: consensus_target(tk) for tk in tickers})
    mag = (ctar - ctar.shift(3)) / close_m.where(close_m > 0)
    RR, NC, NA = {}, {}, {}
    for tk in tickers:
        rr, nc, na = breadth(tk); RR[tk] = rr; NC[tk] = nc; NA[tk] = na
    RR = pd.DataFrame(RR, index=midx); NC = pd.DataFrame(NC, index=midx); NA = pd.DataFrame(NA, index=midx)

    # composite = cross-sectional z(mag) + z(raise_ratio)
    def zrow(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)
    COMP = zrow(mag) + zrow(RR)

    signals = {"mag_tgt_rev_3m": mag, "raise_ratio": RR, "net_raise_cnt": NC, "net_analysts": NA, "composite": COMP}

    # PIT mktcap for cap buckets
    frs = defaultdict(list)
    for tk, ad, sh in FinancialReport.objects.filter(ticker__in=tickers, shares_outstanding__gt=0).values_list("ticker", "avail_date", "shares_outstanding"):
        if ad and sh:
            frs[tk].append((pd.Timestamp(ad).value, float(sh)))
    shares_m = {}
    for tk in tickers:
        pts = sorted(frs.get(tk, []))
        if not pts:
            continue
        di = np.array([t for t, _ in pts]); sv = np.array([v for _, v in pts]); o = np.full(len(midx), np.nan)
        for j, dd in enumerate(midx_ts):
            b = np.searchsorted(di, dd, side="right")
            if b > 0:
                o[j] = sv[b - 1]
        shares_m[tk] = pd.Series(o, index=midx)
    shares_m = pd.DataFrame(shares_m).reindex(columns=tickers)
    mcap = (price_basis.as_traded_close(close_m) * shares_m).reindex(columns=tickers)
    liquid = dvol >= MIN_DVOL

    def stats(r, bench=None):
        r = r.dropna()
        if len(r) < 12:
            return {"n": int(len(r))}
        eq = (1 + r).prod(); yrs = len(r) / 12.0
        o = {"n": int(len(r)), "total_pct": float((eq - 1) * 100), "cagr_pct": float((eq ** (1 / yrs) - 1) * 100) if eq > 0 else -100.0,
             "sharpe": float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0,
             "maxdd_pct": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)}
        if bench is not None:
            b = bench.reindex(r.index).fillna(0.0)
            o["vs_spy_pp"] = float(((eq - 1) - ((1 + b).prod() - 1)) * 100)
            bl = bull.reindex(r.index).fillna(True)
            o["bull_mean_pct"] = float(r[bl].mean() * 100); o["bear_mean_pct"] = float(r[~bl].mean() * 100)
        return o

    def book(sig, lo, hi, leg="long", q=0.2):
        rets, prev = [], set()
        for d in midx[:-1]:
            s = sig.loc[d]; fr = fret.loc[d]; lq = liquid.loc[d]; mc = mcap.loc[d]
            m = s.notna() & fr.notna() & lq & (mc >= lo) & (mc < hi)
            names = s[m]
            if len(names) < 20:
                rets.append(np.nan); prev = set(); continue
            k = max(1, int(len(names) * q)); top = set(names.nlargest(k).index); bot = set(names.nsmallest(k).index)
            if leg == "ls":
                rets.append(float(fr[list(top)].mean() - fr[list(bot)].mean())); prev = top
            else:
                turn = len(top ^ prev) / max(1, len(top))
                rets.append(float(fr[list(top)].mean()) - (COST_BPS / 1e4) * turn); prev = top
        return pd.Series(rets, index=midx[:-1])

    def halves(r, bench):
        r = r.dropna(); mid = len(r) // 2
        if len(r) < 24:
            return {}
        return {"H1_sh": round(stats(r.iloc[:mid]).get("sharpe", 0), 2), "H2_sh": round(stats(r.iloc[mid:]).get("sharpe", 0), 2)}

    res = {"params": {"win_days": WIN_D, "cost_bps": COST_BPS, "window": [str(midx[0].date()), str(midx[-1].date())]}, "buckets": {}}
    spyv = spy_fwd.reindex(midx[:-1])
    print(f"\n=== ANALYST TARGET-RAISE BREADTH vs MAGNITUDE (monthly, PIT cap-bucketed, costed {COST_BPS:.0f}bps, {midx[0].date()}..{midx[-1].date()}) ===", flush=True)
    print(f"  SPY: total {stats(spyv).get('total_pct',0):+.0f}% / Sh{stats(spyv).get('sharpe',0):.2f}", flush=True)
    for bname, lo, hi in BUCKETS:
        res["buckets"][bname] = {}
        print(f"\n--- {bname.upper()} cap (${lo/1e9:.0f}-{hi/1e9 if np.isfinite(hi) else '∞'}B) ---", flush=True)
        print(f"{'signal':>16} {'Ltot':>7} {'CAGR':>6} {'Sh':>5} {'DD':>7} {'vsSPY':>7} {'bull/bear':>12} {'LS_Sh':>6} {'H1/H2sh':>9}", flush=True)
        for name, sig in signals.items():
            lg = book(sig, lo, hi, "long"); ls = book(sig, lo, hi, "ls")
            st = stats(lg, bench=spyv); ls_st = stats(ls); hv = halves(lg, spyv)
            st["long_short"] = ls_st; st["halves"] = hv
            res["buckets"][bname][name] = st
            print(f"{name:>16} {st.get('total_pct',0):>+6.0f}% {st.get('cagr_pct',0):>+5.1f}% {st.get('sharpe',0):>5.2f} "
                  f"{st.get('maxdd_pct',0):>6.1f}% {st.get('vs_spy_pp',0):>+6.0f} {st.get('bull_mean_pct',0):>+5.1f}/{st.get('bear_mean_pct',0):>+4.1f} "
                  f"{ls_st.get('sharpe',0):>6.2f} {hv.get('H1_sh',0):>4.2f}/{hv.get('H2_sh',0):>4.2f}", flush=True)

    Path("/app/.data/studies/analyst_breadth_book.json").write_text(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="analyst_breadth_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[analyst_breadth_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
