#!/usr/bin/env python3
"""MACRO-EVENT -> SECTOR-DRIFT (user: "sometimes we kinda know the future from current events — oil up because of
a war, rates up because of inflation. how do we use that?"). The HONEST, backtestable core of that idea: does a
PUBLIC macro event (observable the day it prints) predict WHICH sector outperforms next month — i.e. carry sector
info the flagship's price-momentum rotation doesn't already have? We do NOT hand-label "we knew"; every event is a
mechanical rule on a dated macro series (FRED / UST curve / oil). Hindsight is impossible here.

For each event we hold a directional hypothesis (a FAVORED sector basket that should lead, a DISFAVORED basket that
should lag) and measure the forward 1-MONTH long-short (fav - dis), month-end grid, EW baskets. Honesty gates:
  (1) event_LS vs UNCONDITIONAL LS (same baskets, all months) — is it an EVENT signal or just a permanent tilt?
  (2) both halves H1(2016-20)/H2(2021-26) SAME SIGN — real edges hold, noise flips.
  (3) t-stat on the event-month LS series (Newey-ish: monthly non-overlap at 1mo horizon).
Saves BacktestResult[macro_event_sector_drift].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/macro_event_sector_drift.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from seq_fundamental_study import load_candles
from core.models import MacroSeries, TreasuryRate

SPLIT = pd.Timestamp("2021-01-01")
START = "2016-06-01"

# event -> (favored basket, disfavored basket). Each observable the day the macro series prints.
EVENTS = {
    # oil shock (WTI 1-mo move) — energy vs the consumers/transports that pay for fuel
    "oil_up":        (["XLE", "XOP", "OIH"], ["XLY", "IYT", "XRT"]),
    "oil_down":      (["XLY", "IYT", "XRT"], ["XLE", "XOP", "OIH"]),
    # market inflation EXPECTATION rising (10y breakeven) — real-asset/value vs long-duration growth+bonds
    "infl_exp_up":   (["XLE", "XLF", "XLB", "IWD"], ["XLK", "QQQ", "IWF", "TLT"]),
    "infl_exp_down": (["XLK", "QQQ", "IWF", "TLT"], ["XLE", "XLF", "XLB", "IWD"]),
    # realized CPI YoY accelerating (publication-lagged) — same value-vs-growth thesis, slower signal
    "cpi_accel":     (["XLE", "XLF", "XLB", "IWD"], ["XLK", "QQQ", "IWF", "TLT"]),
    "cpi_decel":     (["XLK", "QQQ", "IWF", "TLT"], ["XLE", "XLF", "XLB", "IWD"]),
    # short-rate (2Y) rising — banks vs rate-sensitive (utilities/REITs/bonds/growth)
    "rate2y_up":     (["XLF", "KRE", "IWD"], ["XLU", "XLRE", "TLT", "QQQ"]),
    "rate2y_down":   (["XLU", "XLRE", "TLT", "QQQ"], ["XLF", "KRE", "IWD"]),
    # 10Y Treasury yield rising — the DURATION benchmark: hits long-duration growth/utilities/REITs/bonds,
    # helps banks/value. Slower, more persistent than the 2Y (growth+inflation expectations, not just Fed policy).
    "rate10y_up":    (["XLF", "KRE", "IWD", "XLE"], ["XLK", "QQQ", "XLU", "XLRE", "TLT"]),
    "rate10y_down":  (["XLK", "QQQ", "XLU", "XLRE", "TLT"], ["XLF", "KRE", "IWD", "XLE"]),
    # curve steepening (10y-2y) — cyclicals/banks vs defensives
    "curve_steepen": (["XLF", "KRE", "XLI"], ["XLU", "XLP"]),
    # net liquidity (WALCL - RRP - TGA) 3-mo direction — risk-on growth vs defensives
    "liq_up":        (["QQQ", "XLK", "IWF", "HYG"], ["XLP", "XLU", "GLD"]),
    "liq_down":      (["XLP", "XLU", "GLD"], ["QQQ", "XLK", "IWF", "HYG"]),
    # broad USD (DTWEXBGS) 3-mo direction — domestic vs commodity/multinational
    "usd_up":        (["XRT", "XLP"], ["XLB", "XLE", "GDX"]),
    "usd_down":      (["XLB", "XLE", "GDX"], ["XRT", "XLP"]),
}
ALL_ETFS = sorted({t for f, d in EVENTS.values() for t in (f + d)})


def macro_daily(series):
    q = MacroSeries.objects.filter(series=series).order_by("date").values_list("date", "value")
    s = pd.Series({pd.Timestamp(d): (float(v) if v is not None else np.nan) for d, v in q})
    return s.sort_index()


def rate_tenor(tenor):
    q = TreasuryRate.objects.filter(series="yield", tenor=tenor).order_by("date").values_list("date", "rate")
    s = pd.Series({pd.Timestamp(d): (float(v) if v is not None else np.nan) for d, v in q})
    return s.sort_index()


def tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


HORIZONS = [1, 3, 6, 12]                                         # months — persistence sweep


def main():
    ed = load_candles(ALL_ETFS + ["CL=F", "SPY"])
    me = ed["SPY"]["Close"].resample("ME").last().index
    me = me[me >= START]
    close_m = pd.DataFrame({t: ed[t]["Close"].resample("ME").last().reindex(me) for t in ALL_ETFS if t in ed})
    fwdH = {H: close_m.shift(-H) / close_m - 1.0 for H in HORIZONS}   # cumulative fwd sector returns per horizon

    def asof(daily):                                             # last known value at each month-end (PIT)
        return daily.reindex(daily.index.union(me)).ffill().reindex(me)

    oil = asof(ed["CL=F"]["Close"].resample("D").last().ffill())
    t10yie = asof(macro_daily("T10YIE"))
    t10y2y = asof(macro_daily("T10Y2Y"))
    dtwex = asof(macro_daily("DTWEXBGS"))
    r2y = asof(rate_tenor("2Y"))
    r10y = asof(rate_tenor("10Y"))
    # net liquidity in $B: WALCL is $M -> /1000; RRP & TGA already $B
    netliq = asof((macro_daily("WALCL") / 1000.0)) - asof(macro_daily("RRPONTSYD")) - asof(macro_daily("WTREGEN"))
    # CPI YoY, publication-lagged: at month-end d only CPI ref-months strictly before this month are public
    cpi_raw = macro_daily("CPIAUCSL")
    cpi_yoy = (cpi_raw / cpi_raw.shift(12) - 1.0)
    cpi_yoy_me = pd.Series(index=me, dtype=float)
    for d in me:
        avail = cpi_yoy[cpi_yoy.index < d.replace(day=1)]        # ref-month released mid next month -> safe
        cpi_yoy_me.loc[d] = avail.iloc[-1] if len(avail) else np.nan
    cpi_accel = cpi_yoy_me.diff(3)                               # YoY change (decimal) over last 3 prints

    cond = {
        "oil_up":        oil.pct_change(1) >= 0.08,
        "oil_down":      oil.pct_change(1) <= -0.08,
        "infl_exp_up":   t10yie.diff(3) >= 0.20,
        "infl_exp_down": t10yie.diff(3) <= -0.20,
        "cpi_accel":     cpi_accel >= 0.003,                     # YoY up >=0.3pp over 3 prints
        "cpi_decel":     cpi_accel <= -0.003,
        "rate2y_up":     r2y.diff(3) >= 0.40,
        "rate2y_down":   r2y.diff(3) <= -0.40,
        "rate10y_up":    r10y.diff(3) >= 0.30,
        "rate10y_down":  r10y.diff(3) <= -0.30,
        "curve_steepen": t10y2y.diff(3) >= 0.30,
        "liq_up":        netliq.pct_change(3) > 0.01,
        "liq_down":      netliq.pct_change(3) < -0.01,
        "usd_up":        dtwex.pct_change(3) >= 0.02,
        "usd_down":      dtwex.pct_change(3) <= -0.02,
    }

    def basket_ls(fav, dis, H):
        f = fwdH[H][[t for t in fav if t in close_m.columns]].mean(axis=1)
        d = fwdH[H][[t for t in dis if t in close_m.columns]].mean(axis=1)
        return f - d

    res = {"window": f"{str(me.min().date())}..{str(me.max().date())}", "horizons_m": HORIZONS, "events": {}}
    print(f"=== MACRO-EVENT -> SECTOR DRIFT — PERSISTENCE SWEEP (fav-dis LS, cumulative, {res['window']}) ===", flush=True)
    print("  cols: LS@Hmo = mean cumulative fav-dis over next H months when event active | (t) overlap-haircut",
          flush=True)
    print(f"{'event':>14} {'n':>4} | {'1mo':>13} {'3mo':>13} {'6mo':>13} {'12mo':>13} | {'12m both-halves':>16}", flush=True)
    for ev, (fav, dis) in EVENTS.items():
        c = cond[ev].reindex(me).fillna(False)
        rec = {"favored": fav, "disfavored": dis, "by_h": {}}
        n_ev = None; cells = []
        for H in HORIZONS:
            ls = basket_ls(fav, dis, H)
            valid = ls.notna()
            ev_ls = ls[c & valid]; un_ls = ls[valid]
            if len(ev_ls) < 4:
                cells.append("   n/a"); rec["by_h"][H] = {"n": int(len(ev_ls))}; continue
            n_ev = len(ev_ls)
            t_adj = tstat(ev_ls) / math.sqrt(H)                 # crude overlap haircut for cumulative returns
            rec["by_h"][H] = {
                "n": int(len(ev_ls)), "event_ls_pct": round(float(ev_ls.mean() * 100), 3),
                "t_adj": round(t_adj, 2), "uncond_ls_pct": round(float(un_ls.mean() * 100), 3),
                "edge_vs_uncond_pct": round(float((ev_ls.mean() - un_ls.mean()) * 100), 3),
            }
            cells.append(f"{ev_ls.mean()*100:>+7.2f}({t_adj:>+4.1f})")
        # both-halves at 12mo (persistence)
        ls12 = basket_ls(fav, dis, 12); ev12 = ls12[c & ls12.notna()]
        h1 = ev12[ev12.index < SPLIT]; h2 = ev12[ev12.index >= SPLIT]
        hh = f"{(h1.mean()*100 if len(h1) else float('nan')):>+6.1f}/{(h2.mean()*100 if len(h2) else float('nan')):>+6.1f}"
        rec["h1_12m_pct"] = round(float(h1.mean() * 100), 2) if len(h1) else None
        rec["h2_12m_pct"] = round(float(h2.mean() * 100), 2) if len(h2) else None
        res["events"][ev] = rec
        print(f"{ev:>14} {(n_ev or 0):>4} | {cells[0]:>13} {cells[1]:>13} {cells[2]:>13} {cells[3]:>13} | {hh:>16}", flush=True)

    # persistence verdict: LS grows with horizon AND both 12m halves same (+) sign AND adds vs uncond at 12m
    def persistent(r):
        h = r["by_h"]
        if not all(H in h and "event_ls_pct" in h[H] for H in HORIZONS):
            return False
        grows = h[12]["event_ls_pct"] > h[3]["event_ls_pct"] > 0
        adds = h[12]["edge_vs_uncond_pct"] > 0
        both = (r.get("h1_12m_pct") or 0) > 0 and (r.get("h2_12m_pct") or 0) > 0
        return grows and adds and both
    winners = [e for e, r in res["events"].items() if persistent(r)]
    res["persistent_events"] = winners
    print(f"\nPERSISTENT (LS grows to 12mo, adds vs uncond, both 12m halves +): {winners or 'NONE'}", flush=True)

    open("/app/.data/studies/macro_event_sector_drift.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="macro_event_sector_drift", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[macro_event_sector_drift]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
