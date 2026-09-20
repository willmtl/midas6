#!/usr/bin/env python3
"""FDA CATALYST — Phase 1 event study. Does a pharma/biotech stock DRIFT UP into its FDA action date (approval), so we
can buy ahead and SELL BEFORE the binary print? For each mapped approval event (openFDA action_date, ticker via
fda_ticker_map), measure the run-up return over [action_date - T_entry, action_date - T_exit] trading days (exit BEFORE
the print). Sweep T_entry x T_exit. Report mean/median, win rate, t-stat, both halves (pre/post 2021), and EXCESS vs
XBI (biotech beta must be beaten, not just cash). Segment the headline window by cap bucket x kind (ORIG vs sNDA) —
never averaged. Saves BacktestResult[fda_runup_study].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fda_runup_study.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, FinancialReport

APPROVALS = "/app/.data/fda_approvals.json"
TMAP = "/app/.data/fda_ticker_map.json"
SWEEP = [(5, 0), (5, 3), (10, 0), (10, 3), (10, 5), (21, 0), (21, 3), (21, 5), (42, 0), (42, 5)]
HEADLINE = (21, 3)
SPLIT = pd.Timestamp("2021-01-01")


def cap_bucket(mc):
    if mc is None or mc <= 0:
        return "unknown"
    if mc < 3e8:
        return "micro(<300M)"
    if mc < 2e9:
        return "small(300M-2B)"
    if mc < 1e10:
        return "mid(2-10B)"
    return "large(>10B)"


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def load_close(tickers):
    out = {}
    tickers = list(tickers)
    for i in range(0, len(tickers), 200):
        rows = Candle.objects.filter(ticker__in=tickers[i:i + 200], interval="1d", date__gte="2014-06-01"
                                     ).values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            out[tk] = g.sort_values("date").set_index("date")["close"]
    return out


def shares_index(tickers):
    m = {}
    for tk, av, sh in FinancialReport.objects.filter(ticker__in=list(tickers)).exclude(shares_outstanding=None
            ).values_list("ticker", "avail_date", "shares_outstanding"):
        m.setdefault(tk, []).append((pd.Timestamp(av), float(sh)))
    for tk in m:
        m[tk].sort()
    return m


def main():
    events = json.load(open(APPROVALS))
    tmap = json.load(open(TMAP))
    ev = [e for e in events if e["action_date"] >= "2015-01-01" and tmap.get(e["sponsor_name"])]
    for e in ev:
        e["ticker"] = tmap[e["sponsor_name"]]
    tickers = {e["ticker"] for e in ev}
    print(f"events >=2015 with tradeable ticker: {len(ev)} across {len(tickers)} tickers", flush=True)

    close = load_close(tickers | {"XBI"})
    shr = shares_index(tickers)
    xbi = close.get("XBI")

    def px_at(s, ref_pos_date, back):
        """close `back` trading days before the trading-day at/just-before ref date; None if unavailable."""
        if s is None or len(s) == 0:
            return None, None
        pos = s.index.searchsorted(ref_pos_date, side="right") - 1     # last trading day <= action_date
        j = pos - back
        if pos < 0 or j < 0 or pos >= len(s):
            return None, None
        return float(s.iloc[j]), s.index[j]

    def mktcap(tk, when):
        lst = shr.get(tk) or []
        val = None
        for av, sh in lst:
            if av <= when:
                val = sh
            else:
                break
        if val is None:
            return None
        p, _ = px_at(close.get(tk), when, 0)
        return val * p if p else None

    # build per-event run-up records for each sweep combo
    recs = {c: [] for c in SWEEP}
    for e in ev:
        tk = e["ticker"]; s = close.get(tk)
        if s is None:
            continue
        d = pd.Timestamp(e["action_date"])
        for (te, tx) in SWEEP:
            p_in, din = px_at(s, d, te)
            p_out, dout = px_at(s, d, tx)
            if not p_in or not p_out or p_in <= 0:
                continue
            r = p_out / p_in - 1.0
            # XBI over the same calendar window
            xr = np.nan
            if xbi is not None and din is not None and dout is not None:
                xi = xbi.reindex([din], method="ffill"); xo = xbi.reindex([dout], method="ffill")
                if len(xi) and len(xo) and float(xi.iloc[0]) > 0:
                    xr = float(xo.iloc[0]) / float(xi.iloc[0]) - 1.0
            recs[(te, tx)].append({"tk": tk, "date": d, "kind": e["kind"], "r": r,
                                   "xr": xr, "excess": r - (xr if not np.isnan(xr) else 0.0),
                                   "cap": cap_bucket(mktcap(tk, din)) if din is not None else "unknown"})

    def summ(rows, key="r"):
        v = pd.Series([x[key] for x in rows]).dropna()
        if len(v) < 5:
            return {"n": len(v)}
        return {"n": int(len(v)), "mean_pct": round(float(v.mean() * 100), 2), "median_pct": round(float(v.median() * 100), 2),
                "win_pct": round(float((v > 0).mean() * 100), 1), "t": round(tstat(v), 2)}

    res = {"n_events": len(ev), "n_tickers": len(tickers), "sweep": {}, "headline": f"{HEADLINE[0]}->{HEADLINE[1]}"}
    print(f"\n{'entry->exit':>12} {'n':>5} {'mean%':>7} {'med%':>6} {'win%':>6} {'t':>5} | {'exMean%':>7} {'exWin%':>6} {'exT':>5} | {'H1t':>5} {'H2t':>5}", flush=True)
    for c in SWEEP:
        rows = recs[c]
        base = summ(rows, "r"); ex = summ(rows, "excess")
        h1 = summ([x for x in rows if x["date"] < SPLIT], "r"); h2 = summ([x for x in rows if x["date"] >= SPLIT], "r")
        res["sweep"][f"{c[0]}->{c[1]}"] = {"raw": base, "excess_vs_xbi": ex,
                                           "h1_t": h1.get("t"), "h2_t": h2.get("t"),
                                           "h1_mean": h1.get("mean_pct"), "h2_mean": h2.get("mean_pct")}
        print(f"{str(c[0])+'->'+str(c[1]):>12} {base.get('n',0):>5} {base.get('mean_pct',0):>7.2f} {base.get('median_pct',0):>6.2f} "
              f"{base.get('win_pct',0):>6.1f} {base.get('t',0):>5.2f} | {ex.get('mean_pct',0):>7.2f} {ex.get('win_pct',0):>6.1f} "
              f"{ex.get('t',0):>5.2f} | {str(h1.get('t','')):>5} {str(h2.get('t','')):>5}", flush=True)

    # headline window segmented by cap x kind
    hrows = recs[HEADLINE]
    print(f"\n=== headline {HEADLINE[0]}->{HEADLINE[1]}d segmented (raw run-up) ===", flush=True)
    seg = {}
    for kind in ("ORIG", "sNDA"):
        for cap in ("micro(<300M)", "small(300M-2B)", "mid(2-10B)", "large(>10B)", "unknown"):
            rows = [x for x in hrows if x["kind"] == kind and x["cap"] == cap]
            st = summ(rows, "r"); ex = summ(rows, "excess")
            if st.get("n", 0) >= 5:
                seg[f"{kind}/{cap}"] = {"raw": st, "excess": ex}
                print(f"  {kind:5} {cap:16} n={st['n']:4} mean {st['mean_pct']:>6.2f}% win {st['win_pct']:>5.1f}% t {st['t']:>5.2f}"
                      f"  | exVsXBI {ex.get('mean_pct',0):>6.2f}% t {ex.get('t',0):>5.2f}", flush=True)
    res["headline_segmented"] = seg

    json.dump(res, open("/app/.data/studies/fda_runup_study.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="fda_runup_study", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[fda_runup_study]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
