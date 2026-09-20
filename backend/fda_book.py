#!/usr/bin/env python3
"""FDA CATALYST — Phase 3 book. Tradeable portfolio version of the PIT-validated 42->5 run-up: each mapped approval
enters the book 42 trading days before its action date and exits 5 days before (never holds the binary). Daily EW
across all names currently in-window, net of costs (turnover x cost_bps). Reports total/CAGR/Sharpe/maxDD, vs XBI over
the same days, both halves, avg concurrent names — for universe cuts: ALL, ORIG-only, ORIG small/mid-cap (<10B, where
Phase-1/acceptance segmentation showed the edge lives; large+sNDA dilute to beta). Saves BacktestResult[fda_book].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fda_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, FinancialReport

APPROVALS = "/app/.data/fda_approvals.json"
TMAP = "/app/.data/fda_ticker_map.json"
SPLIT = pd.Timestamp("2021-01-01")
T_ENTRY, T_EXIT = 42, 5
COST_BPS = 30.0                      # one-way spread+impact for small-cap biotech


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


def stats(r, bench=None):
    r = r.dropna()
    if len(r) < 60:
        return {"n_days": int(len(r))}
    eq = (1 + r).prod(); yrs = len(r) / 252.0
    o = {"n_days": int(len(r)), "total_pct": round(float((eq - 1) * 100)),
         "cagr_pct": round(float((eq ** (1 / yrs) - 1) * 100), 1) if eq > 0 else -100.0,
         "sharpe": round(float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0, 2),
         "maxdd_pct": round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), 1)}
    if bench is not None:
        b = bench.reindex(r.index).fillna(0.0); beq = (1 + b).prod()
        o["vs_xbi_pp"] = round(float((eq - beq) * 100))
        o["xbi_cagr_pct"] = round(float((beq ** (1 / yrs) - 1) * 100), 1) if beq > 0 else None
    return o


def main():
    events = json.load(open(APPROVALS))
    tmap = json.load(open(TMAP))
    ev = [dict(e, ticker=tmap[e["sponsor_name"]]) for e in events
          if e["action_date"] >= "2015-01-01" and tmap.get(e["sponsor_name"])]
    tickers = {e["ticker"] for e in ev}
    close = load_close(tickers | {"XBI"})
    xbi = close.get("XBI")
    xbi_ret = xbi.pct_change() if xbi is not None else None

    shr = {}
    for tk, av, sh in FinancialReport.objects.filter(ticker__in=list(tickers)).exclude(shares_outstanding=None
            ).values_list("ticker", "avail_date", "shares_outstanding"):
        shr.setdefault(tk, []).append((pd.Timestamp(av), float(sh)))
    for tk in shr:
        shr[tk].sort()

    def mktcap(tk, when):
        val = None
        for av, sh in shr.get(tk, []):
            if av <= when:
                val = sh
            else:
                break
        if val is None or when not in close.get(tk, pd.Series()).index:
            s = close.get(tk)
            if s is None:
                return None
            p = s.index.searchsorted(when, side="right") - 1
            if p < 0:
                return None
            return (val * float(s.iloc[p])) if val else None
        return val * float(close[tk].loc[when])

    # build per-event holding windows (entry/exit trading days from the ticker's own calendar)
    positions = []                         # (ticker, entry_date, exit_date, kind, cap_at_entry)
    for e in ev:
        s = close.get(e["ticker"])
        if s is None:
            continue
        d = pd.Timestamp(e["action_date"])
        p = s.index.searchsorted(d, side="right") - 1
        if p < 0 or p - T_ENTRY < 0:
            continue
        entry, exit_ = s.index[p - T_ENTRY], s.index[p - T_EXIT]
        if exit_ <= entry:
            continue
        positions.append({"tk": e["ticker"], "entry": entry, "exit": exit_, "kind": e["kind"],
                          "cap": mktcap(e["ticker"], entry)})
    print(f"positions built: {len(positions)} (from {len(ev)} events)", flush=True)

    # daily returns per ticker
    dret = {tk: close[tk].pct_change() for tk in tickers if tk in close}
    all_days = pd.DatetimeIndex(sorted(xbi.index)) if xbi is not None else None

    def run_book(sel, label):
        pos = [p for p in positions if sel(p)]
        if not pos:
            print(f"  {label}: no positions", flush=True); return None
        # held set per day via entry/exit; earn returns from day AFTER entry through exit
        held_days = {}
        enter_on, exit_on = {}, {}
        for p in pos:
            days = all_days[(all_days > p["entry"]) & (all_days <= p["exit"])]
            for dd in days:
                held_days.setdefault(dd, []).append(p["tk"])
            enter_on.setdefault(p["entry"], 0); enter_on[p["entry"]] += 1
            exit_on.setdefault(p["exit"], 0); exit_on[p["exit"]] += 1
        idx = pd.DatetimeIndex(sorted(held_days))
        port = pd.Series(0.0, index=idx); ncon = pd.Series(0, index=idx)
        for dd in idx:
            names = held_days[dd]
            rr = np.nanmean([dret[t].get(dd, np.nan) for t in names if t in dret])
            n = len(names)
            turn = (enter_on.get(dd, 0) + exit_on.get(dd, 0)) / max(1, n)
            port.loc[dd] = (0.0 if np.isnan(rr) else rr) - turn * (COST_BPS / 1e4)
            ncon.loc[dd] = n
        st = stats(port, bench=xbi_ret)
        h1 = stats(port[port.index < SPLIT], bench=xbi_ret); h2 = stats(port[port.index >= SPLIT], bench=xbi_ret)
        st["avg_concurrent"] = round(float(ncon.mean()), 1); st["n_positions"] = len(pos)
        st["h1_cagr"] = h1.get("cagr_pct"); st["h2_cagr"] = h2.get("cagr_pct")
        st["h1_vsxbi"] = h1.get("vs_xbi_pp"); st["h2_vsxbi"] = h2.get("vs_xbi_pp")
        print(f"  {label:26} pos={len(pos):4} conc={st['avg_concurrent']:>4} | net CAGR {st.get('cagr_pct')}%/Sh{st.get('sharpe')}"
              f"/DD{st.get('maxdd_pct')}% | XBI CAGR {st.get('xbi_cagr_pct')}% vs {st.get('vs_xbi_pp')}pp | H1/H2 vsXBI {st['h1_vsxbi']}/{st['h2_vsxbi']}", flush=True)
        return st

    print(f"\n=== FDA RUN-UP BOOK (42->5d window, daily EW, {COST_BPS:.0f}bps/side, net) — vs XBI ===", flush=True)
    res = {}
    res["all"] = run_book(lambda p: True, "ALL (ORIG+sNDA)")
    res["orig"] = run_book(lambda p: p["kind"] == "ORIG", "ORIG only")
    res["orig_smallmid"] = run_book(lambda p: p["kind"] == "ORIG" and (p["cap"] or 1e18) < 1e10, "ORIG small/mid (<10B)")
    res["orig_small"] = run_book(lambda p: p["kind"] == "ORIG" and (p["cap"] or 1e18) < 2e9, "ORIG small (<2B)")
    res["params"] = {"t_entry": T_ENTRY, "t_exit": T_EXIT, "cost_bps": COST_BPS}

    json.dump(res, open("/app/.data/studies/fda_book.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="fda_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[fda_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
