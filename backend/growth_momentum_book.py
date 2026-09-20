#!/usr/bin/env python3
"""GROWTH-MOMENTUM BOOK (standalone, user: "a separate momentum book you accept will underperform the flagship").
Chase the growth-sleeve upside directly with a momentum FACTOR book instead of forcing it through the value gate.
Universe = US/CA, $5M-liquid constituents of the 21 growth/innovation sleeves (Semis/Biotech/Genomics/Software/
Space/Cannabis/etc.). Monthly: long the top-quintile by price momentum, EW, hold 1mo, costed 20bps/side.

HONEST GATES (a growth book that just = QQQ beta is pointless — buy QQQ instead):
  - vs SPY AND vs QQQ (the growth-beta benchmark): does selection beat just owning growth?
  - LONG-SHORT spread (top - bottom quintile): isolates alpha from beta (should be >0 if momentum SELECTS).
  - BULL/BEAR split + both halves: beta shows up as all-return-in-bull + H1/H2 decay.
Signals: mom6, mom12, mom12_1 (12-1 skip-a-month, classic). Also a SLEEVE-GATED arm (name in a currently-top-10-accel
growth sleeve AND top-momentum) to mirror the flagship's sector-accel gate. Saves BacktestResult[growth_momentum_book].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/growth_momentum_book.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import config, sector_holdings
from seq_fundamental_study import load_candles

MIN_DVOL = 5e6; COST_BPS = 20.0; TOP_ACCEL = 10; SPLIT = "2021-01-01"
GROWTH = ['Semiconductors', 'Biotech', 'Genomics', 'Cybersecurity', 'AI & Robotics', 'Cloud Computing', 'Space',
          'Clean Energy', 'Solar', 'Fintech', 'Electric Vehicles', 'Internet', 'Software', 'Lithium & Battery',
          'Nanotechnology', 'Cannabis', 'Hydrogen', 'Psychedelics', 'E-Commerce', 'Social Media', 'Gaming & Esports']


def is_usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def stats(r, bench=None, bull=None):
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
    if bull is not None:
        bl = bull.reindex(r.index).fillna(True)
        o["bull_mean_pct"] = float(r[bl].mean() * 100); o["bear_mean_pct"] = float(r[~bl].mean() * 100)
    return o


def main():
    growth = [g for g in GROWTH if g in config.SECTOR_ETFS]
    etf_daily = load_candles(sorted(set(config.SECTOR_ETFS.values())) + ["SPY", "QQQ"])
    etf_m = pd.DataFrame({n: etf_daily[e]["Close"].resample("ME").last() for n, e in config.SECTOR_ETFS.items() if e in etf_daily and etf_daily[e] is not None})
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    midx = etf_m.index[etf_m.index >= "2016-01-01"]
    topaccel = {d: set(accel.loc[d].dropna().nlargest(TOP_ACCEL).index) if d in accel.index and accel.loc[d].notna().any() else set() for d in midx}
    spy_m = etf_daily["SPY"]["Close"].resample("ME").last().reindex(midx)
    qqq_m = etf_daily["QQQ"]["Close"].resample("ME").last().reindex(midx)
    spy_fwd = spy_m.shift(-1) / spy_m - 1.0
    qqq_fwd = qqq_m.shift(-1) / qqq_m - 1.0
    spy200 = etf_daily["SPY"]["Close"].rolling(200).mean().resample("ME").last().reindex(midx)
    bull = spy_m >= spy200

    cons = {g: [t for t in sector_holdings.get_holdings(g) if is_usca(t)] for g in growth}
    tk2sleeve = {}
    for g in growth:
        for t in cons[g]:
            tk2sleeve.setdefault(t, g)
    allc = sorted(tk2sleeve)
    cand = {}
    for i in range(0, len(allc), 40):
        cand.update(load_candles(allc[i:i + 40]))
    close_m, dvol_m = {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        close_m[tk] = df["Close"].resample("ME").last().reindex(midx)
        dvol_m[tk] = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m)
    tickers = list(close_m.columns)
    fret = close_m.shift(-1) / close_m - 1.0
    liquid = dvol_m >= MIN_DVOL
    SIG = {"mom6": close_m / close_m.shift(6) - 1, "mom12": close_m / close_m.shift(12) - 1,
           "mom12_1": close_m.shift(1) / close_m.shift(12) - 1}

    def book(sig, leg="long", q=0.2, sleeve_gate=False):
        rets, prev = [], set()
        for d in midx[:-1]:
            elig = liquid.loc[d] & close_m.loc[d].notna() & fret.loc[d].notna() & sig.loc[d].notna()
            if sleeve_gate:
                hot = topaccel.get(d, set())
                elig = elig & pd.Series({t: (config.SECTOR_ETFS.get(tk2sleeve[t]) in {config.SECTOR_ETFS[s] for s in hot if s in config.SECTOR_ETFS}) for t in tickers})
            names = sig.loc[d][elig.reindex(tickers).fillna(False)].dropna()
            if len(names) < 15:
                rets.append(np.nan); prev = set(); continue
            k = max(1, int(len(names) * q)); top = set(names.nlargest(k).index); bot = set(names.nsmallest(k).index)
            fr = fret.loc[d]
            if leg == "ls":
                rets.append(float(fr[list(top)].mean() - fr[list(bot)].mean())); prev = top
            else:
                turn = len(top ^ prev) / max(1, len(top)); rets.append(float(fr[list(top)].mean()) - (COST_BPS / 1e4) * turn); prev = top
        return pd.Series(rets, index=midx[:-1])

    def halves(r, bench):
        r = r.dropna(); mid = len(r) // 2
        if len(r) < 24:
            return {}
        return {"H1": round(stats(r.iloc[:mid]).get("sharpe", 0), 2), "H2": round(stats(r.iloc[mid:]).get("sharpe", 0), 2)}

    spyv = spy_fwd.reindex(midx[:-1]); qqqv = qqq_fwd.reindex(midx[:-1])
    res = {"spy": stats(spyv), "qqq": stats(qqqv, bench=spyv), "arms": {}}
    print(f"\n=== GROWTH-MOMENTUM BOOK (US/CA growth-sleeve names, top-quintile, monthly, costed {COST_BPS:.0f}bps) ===", flush=True)
    print(f"  SPY total {res['spy']['total_pct']:+.0f}%/Sh{res['spy']['sharpe']:.2f} | QQQ total {res['qqq']['total_pct']:+.0f}%/Sh{res['qqq']['sharpe']:.2f} (QQQ vsSPY {res['qqq'].get('vs_spy_pp',0):+.0f})", flush=True)
    print(f"{'arm':>22} {'total':>8} {'CAGR':>6} {'Sh':>5} {'DD':>7} {'vsSPY':>7} {'vsQQQ':>7} {'bull/bear':>11} {'LS_Sh':>6} {'H1/H2':>9}", flush=True)
    for sname, sig in SIG.items():
        for gate in (False, True):
            lab = f"{sname}{'+accelgate' if gate else ''}"
            lg = book(sig, "long", sleeve_gate=gate); ls = book(sig, "ls", sleeve_gate=gate)
            st = stats(lg, bench=spyv, bull=bull); ls_st = stats(ls); hv = halves(lg, spyv)
            eqb = (1 + lg.dropna()).prod(); vq = float((eqb - 1) * 100 - res["qqq"]["total_pct"])
            st["vs_qqq_pp"] = vq; st["long_short_sharpe"] = ls_st.get("sharpe", 0); st["halves"] = hv
            res["arms"][lab] = st
            print(f"{lab:>22} {st.get('total_pct',0):>+7.0f}% {st.get('cagr_pct',0):>+5.1f}% {st.get('sharpe',0):>5.2f} "
                  f"{st.get('maxdd_pct',0):>6.1f}% {st.get('vs_spy_pp',0):>+6.0f} {vq:>+6.0f} "
                  f"{st.get('bull_mean_pct',0):>+5.1f}/{st.get('bear_mean_pct',0):>+4.1f} {ls_st.get('sharpe',0):>6.2f} {hv.get('H1',0):>4.2f}/{hv.get('H2',0):>4.2f}", flush=True)

    open("/app/.data/studies/growth_momentum_book.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="growth_momentum_book", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[growth_momentum_book]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
