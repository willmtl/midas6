#!/usr/bin/env python
"""STANDALONE "synthetic" strategy — fully separate from the flagship engine (survivorship_smallcap_study.py).

A two-stage relative-strength value strategy, EVENT-DRIVEN (buys WHEN THE SIGNAL FIRES, not at month-end):
  stage 1  find sectors OVERSOLD-AND-TURNING vs SPY   : sector/SPY weekly RS-RSI(14) crossed up through 30, still fresh (<= CONFL_WK weeks)
  stage 2  within those sectors, a stock OVERSOLD-AND-TURNING vs its sector : stock/sector weekly RS-RSI(14) crosses up through 30
  gate     cheap-P/B (0.1-1.5)  +  PROFITABLE (TTM P/E > 0 -> NO distress)  +  mktcap >= $100M (NO upper ceiling)
  entry    the trading day AFTER the signal week closes (no look-ahead); hold HOLD_D trading days, then exit
  sizing   equal-weight across all concurrently-open positions (daily attribution); CASH when nothing is open
  exits    delisting-aware; a held name that later confirms bankruptcy is wiped -100%; one position per name at a time

Data:  daily prices/signals via the shared adjusted-candle loader (load_candles). The expensive EDGAR point-in-time
       fundamentals (P/B, P/E, market cap) are REUSED from the flagship's generic EXPORT_PANELS bundle.

Run:   1) MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 env EXPORT_PANELS=/app/.data/panels.pkl python -u /app/survivorship_smallcap_study.py
       2) MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/synthetic_strategy.py            # single run (CONFL_WK/HOLD_D env)
          MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 env SWEEP=1 python -u /app/synthetic_strategy.py # widen sweep
Env:   CONFL_WK (default 8), HOLD_D (default 21), PANELS (default /app/.data/panels.pkl), SWEEP, SAVE_CW, SAVE_HD
"""
import os, json, pickle
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from seq_fundamental_study import load_candles

PANELS = os.environ.get("PANELS", "/app/.data/panels.pkl")
OS_THR = 30


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return (100.0 - 100.0 / (1.0 + ru / rd.replace(0.0, np.nan))).fillna(50.0)


def weekly_since_cross(ratio):
    r = rsi(ratio, 14)
    cx = (r.shift(1) < OS_THR) & (r >= OS_THR)
    num = pd.Series(range(len(ratio)), index=ratio.index, dtype=float)
    return num - num.where(cx).ffill()


def load_data():
    b = pickle.load(open(PANELS, "rb"))
    pb, pe, mc = b["pb"], b["pe_ttm"], b["mktcap_usd"]
    surv_sector, del_sector = b["surv_sector"], b["delisted_sector"]
    bankrupt_tk, BENCH = set(b["bankrupt_tk"]), b["bench"]
    universe = [t for t in b["common"] if (t in surv_sector or t in del_sector)]
    sec_of = {t: (surv_sector.get(t) or del_sector.get(t)) for t in universe}
    sectors = sorted(set(sec_of.values()))
    scd = load_candles(universe)
    ecd = load_candles(sectors + [BENCH])
    scl = pd.DataFrame({t: scd[t]["Close"] for t in universe if t in scd and "Close" in scd[t]}).sort_index()
    ecl = pd.DataFrame({e: ecd[e]["Close"] for e in sectors + [BENCH] if e in ecd and "Close" in ecd[e]}).sort_index()
    scl = scl[scl > 0]; ecl = ecl[ecl > 0]
    if BENCH not in ecl.columns:
        raise SystemExit("no SPY/bench candles")
    days = scl.index
    pb_d = pb.reindex(days, method="ffill"); pe_d = pe.reindex(days, method="ffill"); mc_d = mc.reindex(days, method="ffill")
    ecl_d = ecl.reindex(days, method="ffill")               # sector ETF closes aligned to the daily grid (for upturn check)
    MOM_D = int(os.environ.get("MOM_D", 10))                # "turning up" lookback (trading days)
    SPY_ONLY = int(os.environ.get("SPY_ONLY", 0))           # 1 = restrict to SPY-scale US large-caps (proxy for S&P 500 membership)
    MCAP_MIN = float(os.environ.get("MCAP_MIN", 1e10))      # large-cap floor for the SPY proxy ($10B)
    spy_ser = ecl_d[BENCH]; spy_ma = spy_ser.rolling(50, min_periods=20).mean()   # market-regime reference

    spy_w = ecl[BENCH].resample("W-FRI").last()
    sec_since = {}
    for e in sectors:
        if e not in ecl.columns:
            continue
        ew = ecl[e].resample("W-FRI").last()
        ix = ew.index.intersection(spy_w.index)
        rr = (ew.reindex(ix) / spy_w.reindex(ix)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rr) >= 40:
            sec_since[e] = weekly_since_cross(rr)

    # ALL stock RS-30 cross candidates that pass the (param-independent) value/profit/cap gates, tagged with sector recency.
    # STOCK_MODE=1: forget the sector entirely -> the RS ratio is stock/SPY and there is NO stage-1 sector-oversold gate.
    STOCK_MODE = int(os.environ.get("STOCK_MODE", 0))
    cands = []
    for t in universe:
        e = sec_of[t]
        if t not in scl.columns:
            continue
        if not STOCK_MODE and e not in sec_since:
            continue
        sc = scl[t].dropna()
        ew = ecl[BENCH] if STOCK_MODE else ecl[e]         # denominator: SPY (stock-only) or the stock's sector ETF
        sw = sc.resample("W-FRI").last(); ewk = ew.resample("W-FRI").last()
        ix = sw.index.intersection(ewk.index)
        ratio = (sw.reindex(ix) / ewk.reindex(ix)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) < 40:
            continue
        r = rsi(ratio, 14)
        cross = (r.shift(1) < OS_THR) & (r >= OS_THR)
        for wf in ratio.index[cross.fillna(False)]:
            ss = 0.0 if STOCK_MODE else sec_since[e].asof(wf)   # no sector gate in stock-only mode
            if not pd.notna(ss):
                continue
            di = days.searchsorted(wf, side="right")
            if di >= len(days):
                continue
            ed = days[di]
            pbv = pb_d.at[ed, t] if t in pb_d.columns else np.nan
            if not (pd.notna(pbv) and 0.1 <= pbv < 1.5):
                continue
            pev = pe_d.at[ed, t] if t in pe_d.columns else np.nan
            if not (pd.notna(pev) and pev > 0):
                continue
            mcv = mc_d.at[ed, t] if t in mc_d.columns else np.nan
            if pd.notna(mcv) and not (mcv >= 1e8):
                continue
            if SPY_ONLY and ("." in t or not (pd.notna(mcv) and mcv >= MCAP_MIN)):   # "in SPY" proxy: US-listed large-cap
                continue
            # "turning up" confirmation: short-term ABSOLUTE momentum positive for BOTH the stock and its sector at entry.
            # This is what separates a real early turn from a mid-crash oversold bounce (both still falling absolutely).
            stk_up = sec_up = None
            if di - MOM_D >= 0:
                p0 = sc.reindex([days[di - MOM_D]]).iloc[0] if days[di - MOM_D] in sc.index else scl[t].iloc[di - MOM_D]
                p1 = scl[t].iloc[di]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    stk_up = bool(p1 / p0 - 1 > 0)
                if e in ecl_d.columns:
                    e0 = ecl_d[e].iloc[di - MOM_D]; e1 = ecl_d[e].iloc[di]
                    if pd.notna(e0) and pd.notna(e1) and e0 > 0:
                        sec_up = bool(e1 / e0 - 1 > 0)
            # market regime at entry: is the broad market itself turning up? (SPY > its 50d MA = crash-avoidance / "tide turned")
            mkt_up = None
            s1 = spy_ser.iloc[di]; sm = spy_ma.iloc[di]
            if pd.notna(s1) and pd.notna(sm):
                mkt_up = bool(s1 > sm)
            cands.append(dict(entry_di=int(di), ticker=t, sector=e, pb=float(pbv), pe=float(pev),
                              mc=(float(mcv) if pd.notna(mcv) else None), sec_wk=float(ss),
                              stk_up=stk_up, sec_up=sec_up, mkt_up=mkt_up))
    cands.sort(key=lambda x: x["entry_di"])
    flm = {}
    try:
        flm = {d: r for d, r in json.load(open("/app/.data/studies/flagship_history.json")).get("monthly_net", [])}
    except Exception:
        pass
    return dict(scl=scl, days=days, del_sector=del_sector, bankrupt_tk=bankrupt_tk,
                cands=cands, flm=flm, n_universe=len(universe), n_sectors=len(sec_since))


def backtest(D, CONFL_WK, HOLD_D, want_trades=False):
    scl, days = D["scl"], D["days"]
    del_sector, bankrupt_tk, flm = D["del_sector"], D["bankrupt_tk"], D["flm"]
    UPTURN = int(os.environ.get("UPTURN", 0))                # 1 = require price turning up (stock AND sector) — REFUTED, off by default
    MKT_UP = int(os.environ.get("MKT_UP", 0))                # 1 = only enter when the broad market is above its 50d MA (crash-avoidance)
    n = len(days)
    dsum = np.zeros(n); dcnt = np.zeros(n)
    open_until = {}; trades = []
    for ev in D["cands"]:
        if ev["sec_wk"] > CONFL_WK:                          # stage-1 freshness window (the "widen" knob)
            continue
        if UPTURN and not (ev.get("stk_up") and ev.get("sec_up")):   # buy only what's genuinely turning UP (not a mid-crash bounce)
            continue
        if MKT_UP and not ev.get("mkt_up"):                 # only buy when the broad market itself has turned up
            continue
        t = ev["ticker"]; di = ev["entry_di"]
        if di < open_until.get(t, -1):
            continue
        sc = scl[t]
        seg = sc.iloc[di:di + HOLD_D + 1].dropna()
        if len(seg) < 2:
            continue
        segret = seg.pct_change().dropna()
        for d, rr in segret.items():
            j = days.get_loc(d); dsum[j] += rr; dcnt[j] += 1
        entry_price = float(seg.iloc[0]); exit_price = float(seg.iloc[-1])
        hold_ret = exit_price / entry_price - 1.0
        exit_di = days.get_loc(seg.index[-1])
        ended_early = (di + HOLD_D) >= n or seg.index[-1] < days[min(di + HOLD_D, n - 1)]
        is_bank = (t in bankrupt_tk) and ended_early and (sc.iloc[di:].dropna().index[-1] < days[-1] - pd.Timedelta(days=10))
        if is_bank:
            hold_ret = -1.0
            kj = min(exit_di + 1, n - 1); dsum[kj] += -1.0; dcnt[kj] += 1
        open_until[t] = exit_di + 1
        if want_trades:
            trades.append(dict(date=str(days[di].date()), exit=str(seg.index[-1].date()), ticker=t, sector=ev["sector"],
                               pb=round(ev["pb"], 3), pe=round(ev["pe"], 1),
                               mktcap_m=(round(ev["mc"] / 1e6, 1) if ev["mc"] else None),
                               sector_rs30_wk=round(ev["sec_wk"], 1), hold_days=int(exit_di - di),
                               ret_pct=round(hold_ret * 100, 2), delisted=(t in del_sector), bankrupt=is_bank))
    port = np.where(dcnt > 0, dsum / np.maximum(dcnt, 1), 0.0)
    pr = pd.Series(port, index=days).iloc[1:]
    eq = (1 + pr).cumprod()
    total = float(eq.iloc[-1] - 1) * 100
    yrs = (days[-1] - days[1]).days / 365.25
    cagr = ((1 + total / 100) ** (1 / yrs) - 1) * 100
    sh = float(pr.mean() / pr.std(ddof=1) * np.sqrt(252)) if pr.std(ddof=1) > 0 else float("nan")
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100
    inv = float((dcnt[1:] > 0).mean()) * 100
    avg_pos = float(dcnt[dcnt > 0].mean()) if (dcnt > 0).any() else 0.0
    corr = None
    if flm:
        mo = (1 + pr).resample("ME").prod() - 1
        idx = [str(x.date()) for x in mo.index]
        fv = np.array([flm.get(d, np.nan) for d in idx]); mv = mo.values.astype(float)
        m = np.isfinite(fv) & np.isfinite(mv)
        if m.sum() > 10:
            corr = round(float(np.corrcoef(mv[m], fv[m])[0, 1]), 3)
    rets = [x["ret_pct"] for x in trades] if want_trades else []
    res = dict(confl_wk=CONFL_WK, hold_d=HOLD_D, total=round(total, 1), cagr=round(cagr, 1), sharpe=round(sh, 2),
               dd=round(dd, 1), pct_days_invested=round(inv, 0), avg_concurrent=round(avg_pos, 1),
               n_trades=(len(trades) if want_trades else int(sum(1 for _ in _count(D, CONFL_WK, HOLD_D)))), corr_flagship=corr)
    if want_trades:
        res["win_rate"] = round(100 * sum(1 for x in rets if x > 0) / max(1, len(rets)), 1)
        res["mean_ret"] = round(float(np.mean(rets)) if rets else 0.0, 2)
        res["median_ret"] = round(float(np.median(rets)) if rets else 0.0, 2)
        res["monthly"] = [[str(d.date()), float(v)] for d, v in ((1 + pr).resample("ME").prod() - 1).items()]
    return res, trades


def _count(D, CW, HD):
    ou = {}
    for ev in D["cands"]:
        if ev["sec_wk"] > CW:
            continue
        if ev["entry_di"] < ou.get(ev["ticker"], -1):
            continue
        ou[ev["ticker"]] = ev["entry_di"] + HD + 1
        yield 1


def main():
    D = load_data()
    print(f"universe={D['n_universe']}  sectors={D['n_sectors']}  raw cross candidates (gated)={len(D['cands'])}", flush=True)
    if os.environ.get("SWEEP"):
        CWs = [8, 13, 26, 52, 104]; HDs = [21, 42, 63]
        print("\n=== WIDEN SWEEP — CONFL_WK (sector-freshness weeks) x HOLD_D (trading days) ===", flush=True)
        print(f"{'CW':>4}{'HD':>5}{'total%':>10}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}{'%inv':>6}{'avgPos':>8}{'corr':>7}", flush=True)
        for cw in CWs:
            for hd in HDs:
                r, _ = backtest(D, cw, hd, want_trades=False)
                print(f"{cw:>4}{hd:>5}{r['total']:>10,.0f}{r['cagr']:>7.1f}{r['sharpe']:>8.2f}{r['dd']:>8.1f}"
                      f"{r['pct_days_invested']:>6.0f}{r['avg_concurrent']:>8.1f}{str(r['corr_flagship']):>7}", flush=True)
        return
    CW = int(os.environ.get("CONFL_WK", 8)); HD = int(os.environ.get("HOLD_D", 21))
    res, trades = backtest(D, CW, HD, want_trades=True)
    print(f"\n=== STANDALONE SYNTHETIC (event-driven; profitable; mcap>=$100M)  CONFL_WK={CW} HOLD_D={HD} ===", flush=True)
    print(f"  UPTURN={os.environ.get('UPTURN','1')} MOM_D={os.environ.get('MOM_D','10')}", flush=True)
    print(f"  total={res['total']:,.0f}%  CAGR={res['cagr']:.1f}  Sharpe={res['sharpe']:.2f}  maxDD={res['dd']:.1f}%  "
          f"trades={res['n_trades']}  win={res['win_rate']}%  avg_concurrent={res['avg_concurrent']}  "
          f"%days_invested={res['pct_days_invested']:.0f}  CORR_vs_flagship={res['corr_flagship']}", flush=True)
    _mo = {d: v for d, v in res["monthly"]}
    _cov = [m for m in sorted(_mo) if "2020-0" in m and m <= "2020-06"]
    print("  COVID window: " + "  ".join(f"{m[-5:]} {_mo[m]*100:+.1f}%" for m in _cov), flush=True)
    summ = dict(strategy=f"standalone event-driven two-stage synthetic (cheap-P/B profitable, mcap>=$100M no-ceiling); CONFL_WK={CW} HOLD_D={HD}",
                confl_wk=CW, hold_d=HD, total=res["total"], cagr=res["cagr"], sharpe=res["sharpe"], dd=res["dd"],
                corr_flagship=res["corr_flagship"], n_trades=res["n_trades"], win_rate=res["win_rate"],
                avg_concurrent=res["avg_concurrent"], pct_days_invested=res["pct_days_invested"],
                mean_ret=res["mean_ret"], median_ret=res["median_ret"])
    json.dump(dict(summary=summ, trades=trades), open("/app/.data/studies/synth_trades.json", "w"), indent=0)
    print(f"saved synth_trades.json ({len(trades)} trades)", flush=True)
    try:
        from core.models import BacktestResult as BR
        from django.utils import timezone as tz
        BR.objects.update_or_create(kind="synth_strategy_standalone",
                                    defaults=dict(computed_at=tz.now(), payload=dict(summ, monthly=res["monthly"])))
        print("saved BacktestResult[synth_strategy_standalone]", flush=True)
    except Exception as _e:
        print("BR save skipped:", _e, flush=True)


if __name__ == "__main__":
    main()
