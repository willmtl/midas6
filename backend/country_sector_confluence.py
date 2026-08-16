#!/usr/bin/env python3
"""COUNTRY x SECTOR CONFLUENCE — test the hypothesis: a hot SECTOR delivers MORE next month when its most-
correlated COUNTRY is ALSO hot (e.g. Korea #1 + Semis #2 -> buy Semis). And the decomposition: when a COUNTRY
is hot, is its DOMINANT SECTOR the better buy than the country basket itself?

Method (monthly, ETF-only, fast): accel = 3mo momentum minus its 3mo-lagged self (same as flagship). Country->
sector links are DATA-DRIVEN: full-sample return correlation, each sector's single most-correlated country.
  Q1 CONFLUENCE: sector fwd-ret | (sector in accel top-tercile AND its linked country also top-tercile)
                 vs sector fwd-ret | (sector top-tercile, linked country NOT) -> lift + t.
  Q2 DECOMPOSITION: when a country is in accel top-tercile, compare next-month return of the country ETF vs its
                 dominant sector ETF -> which is the better expression?
-> BacktestResult[country_sector_confluence] + JSON. (First pass: full-sample corr = mild look-ahead in the
   LINK MAP only, not the signal; flagged. Refine with trailing corr if the effect is real.)
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/country_sector_confluence.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
from seq_fundamental_study import load_candles
from backtest_lowpb import _monthly_close, _tstat_from_returns

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "country_sector_confluence.json"

COUNTRY = {"EWZ", "EWG", "EWJ", "EWY", "MCHI", "INDA", "EWW", "EWA", "EWU", "VGK", "EFA", "EEM", "ILF", "AFK",
           "EWC", "EWQ", "EWI", "EWP", "EWL", "EWN", "EWD", "EWK", "EWO", "EIRL", "NORW", "EDEN", "EWH", "EWS",
           "EIS", "ENZL", "EZU", "EWT", "EWM", "THD", "EIDO", "EPHE", "VNM", "EZA", "TUR", "KSA", "EPOL", "ECH",
           "EPU", "GXG", "ARGT", "GREK", "QAT", "UAE", "PAK", "ACWI", "VEU", "AAXJ", "EPP", "EMXC", "FM"}
# non-thematic sleeves to exclude from the SECTOR side (bonds, pure commodities, factors, broad indices)
NON_THEME = {"TLT", "AGG", "HYG", "TIP", "BIL", "SHV", "SHY", "IEI", "IEF", "LQD", "FLOT", "MUB", "CWB",
             "GLD", "SLV", "USO", "UNG", "DBA", "CORN", "WEAT", "PPLT", "BTC-USD", "IBIT",
             "SCHD", "MTUM", "VTV", "VUG", "SPLV", "MAGS", "IWM", "IWC", "QQQ", "IVV", "DIA", "IJH", "VTI",
             "IWB", "RSP"}


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    tick_to_name = {e: n for n, e in config.SECTOR_ETFS.items()}
    all_etfs = [e for e in config.SECTOR_ETFS.values() if e != "BTC-USD"]
    # exclude regional/global AGGREGATES so "country" = a real single country (else ACWI = global beta dominates links)
    AGG = {"ACWI", "VEU", "AAXJ", "EPP", "EMXC", "FM", "EEM", "EFA", "VGK", "ILF", "EZU", "AFK"}
    countries = [e for e in all_etfs if e in COUNTRY and e not in AGG]
    sectors = [e for e in all_etfs if e not in COUNTRY and e not in NON_THEME]

    daily = load_candles(all_etfs)
    m = _monthly_close({t: d for t, d in daily.items() if t in daily})
    m = m.dropna(axis=1, how="all")
    countries = [c for c in countries if c in m.columns]
    sectors = [s for s in sectors if s in m.columns]
    ret = m.pct_change()
    accel = m.pct_change(3) - m.pct_change(3).shift(3)
    fwd = ret.shift(-1)                              # next-month return
    midx = m.index

    # DATA-DRIVEN LINK: each sector's single most-correlated country (full-sample return corr)
    corr = ret[countries + sectors].corr()
    link = {}
    for s in sectors:
        cc = corr.loc[s, countries].dropna()
        if len(cc):
            c = cc.idxmax()
            link[s] = (c, float(cc.max()))

    # accel tercile membership per month (top third = "hot")
    def hot(row, universe):
        v = row[universe].dropna()
        if len(v) < 6:
            return set()
        thr = v.quantile(2 / 3)
        return set(v[v >= thr].index)

    # Q1 CONFLUENCE
    conf_ret, alone_ret, rows = [], [], []
    per_pair = {}
    for i in range(9, len(midx) - 1):
        date = midx[i]
        hot_s = hot(accel.loc[date], sectors)
        hot_c = hot(accel.loc[date], countries)
        for s in hot_s:
            if s not in link:
                continue
            c, rho = link[s]
            fr = fwd.loc[date, s]
            if not np.isfinite(fr):
                continue
            if c in hot_c:
                conf_ret.append(fr); per_pair.setdefault((s, c), []).append(("conf", fr))
            else:
                alone_ret.append(fr); per_pair.setdefault((s, c), []).append(("alone", fr))

    def avg(x):
        return float(np.mean(x)) * 100 if x else float("nan")

    # Q2 DECOMPOSITION: country hot -> country ETF vs its dominant sector next month
    # invert link: for each country, the sector most correlated to it
    csec = {}
    for c in countries:
        cc = corr.loc[c, sectors].dropna()
        if len(cc):
            csec[c] = (cc.idxmax(), float(cc.max()))
    ctry_ret, dom_sec_ret = [], []
    for i in range(9, len(midx) - 1):
        date = midx[i]
        hot_c = hot(accel.loc[date], countries)
        for c in hot_c:
            if c not in csec:
                continue
            s, rho = csec[c]
            rc, rs = fwd.loc[date, c], fwd.loc[date, s]
            if np.isfinite(rc) and np.isfinite(rs):
                ctry_ret.append(rc); dom_sec_ret.append(rs)

    print("=== Q1 CONFLUENCE: hot sector, does a co-hot linked country add forward return? ===", flush=True)
    print(f"  sector hot + linked country ALSO hot : {avg(conf_ret):+.2f}%/mo  n={len(conf_ret)}  t={_tstat_from_returns(conf_ret):.2f}", flush=True)
    print(f"  sector hot, linked country NOT hot   : {avg(alone_ret):+.2f}%/mo  n={len(alone_ret)}  t={_tstat_from_returns(alone_ret):.2f}", flush=True)
    diff = [x for x in conf_ret]
    lift = avg(conf_ret) - avg(alone_ret)
    print(f"  CONFLUENCE LIFT = {lift:+.2f}%/mo", flush=True)

    print("\n=== Q2 DECOMPOSITION: country hot -> buy country basket or its dominant sector? ===", flush=True)
    print(f"  country ETF next-month     : {avg(ctry_ret):+.2f}%/mo  n={len(ctry_ret)}  t={_tstat_from_returns(ctry_ret):.2f}", flush=True)
    print(f"  its DOMINANT SECTOR next-mo: {avg(dom_sec_ret):+.2f}%/mo  n={len(dom_sec_ret)}  t={_tstat_from_returns(dom_sec_ret):.2f}", flush=True)
    print(f"  SECTOR-OVER-COUNTRY EDGE = {avg(dom_sec_ret) - avg(ctry_ret):+.2f}%/mo", flush=True)

    print("\n=== strongest country<->sector links (full-sample corr) ===", flush=True)
    top_links = sorted(link.items(), key=lambda kv: -kv[1][1])[:15]
    for s, (c, rho) in top_links:
        print(f"  {tick_to_name.get(s, s)[:26]:<27} ~ {tick_to_name.get(c, c)[:20]:<21} corr={rho:.2f}", flush=True)

    payload = {
        "q1_confluence": {"conf_mean_pct": avg(conf_ret), "conf_n": len(conf_ret), "conf_t": _tstat_from_returns(conf_ret),
                          "alone_mean_pct": avg(alone_ret), "alone_n": len(alone_ret), "alone_t": _tstat_from_returns(alone_ret),
                          "lift_pct_mo": lift},
        "q2_decomposition": {"country_mean_pct": avg(ctry_ret), "dom_sector_mean_pct": avg(dom_sec_ret),
                             "sector_over_country_pct_mo": avg(dom_sec_ret) - avg(ctry_ret), "n": len(ctry_ret)},
        "links": {tick_to_name.get(s, s): {"country": tick_to_name.get(c, c), "corr": round(rho, 3)} for s, (c, rho) in link.items()},
        "caveat": "First pass: link map uses FULL-SAMPLE corr (mild look-ahead in WHICH country pairs with WHICH "
                  "sector, not in the accel signal itself). ETF-only, monthly, no fees. If lift is real, redo with "
                  "trailing-window corr for a clean OOS read.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="country_sector_confluence",
                                                 defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[country_sector_confluence]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_CONFLUENCE", flush=True)


if __name__ == "__main__":
    main()
