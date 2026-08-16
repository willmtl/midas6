#!/usr/bin/env python3
"""INDEX vs STOCK-PICK — would we have done better just buying the sector/country INDEX ETF (or SPY) instead of
the cheapest-P/B stock inside it? Replays the flagship (usca_small AND global with_delisted_small) and for every
monthly pick records: the pick's USD return, the SECTOR ETF's own return (the 'index' for that sleeve), and SPY's
return. Reports (1) portfolio total: stock-picks vs hold-the-index-instead vs all-SPY; (2) per-pick win-rate vs its
index; (3) the sectors where the INDEX beat the pick most (the cases the question asks about).
-> /app/.data/studies/index_vs_pick.json + printed tables.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/index_vs_pick.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH
from survivorship_smallcap_study import (TOP_N, CONV, MIN_DVOL, SMALL, MIN_PRICE, MIN_PB, MAJOR_EXCH,
                                         PHARMA_ETFS, MICRO_PHARMA_MIN, GIC_FILE, GIC_TO_ETF,
                                         _usd_factor_matrix, _is_usca)

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "index_vs_pick.json"


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    tick_to_name = {e: n for n, e in config.SECTOR_ETFS.items()}
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    surv_sector, all_holds = {}, set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                surv_sector.setdefault(t, e); all_holds.add(t)
    gic_raw = json.load(open(GIC_FILE))
    from core.models import FinancialReport, DelistedCompany
    dl_have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    dl_exch = {d.ticker: (d.exchange or "").strip() for d in DelistedCompany.objects.filter(ticker__in=list(gic_raw))}
    delisted_sector = {}
    for tk, gic in gic_raw.items():
        e = GIC_TO_ETF.get((gic or "").strip())
        if e and tk in dl_have and dl_exch.get(tk) in MAJOR_EXCH:
            delisted_sector[tk] = e

    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)

    universe = sorted(all_holds | set(delisted_sector))
    stock_daily = load_candles(universe)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(universe)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    as_traded = price_basis.as_traded_close(px)
    mktcap = as_traded * sh
    pb = mktcap / eq.where(eq != 0)
    usd_factor_m, _ = _usd_factor_matrix(list(common), midx)
    ret_factor = usd_factor_m.fillna(1.0)
    px_usd = px * ret_factor; as_traded_usd = as_traded * ret_factor
    mktcap_usd = mktcap * usd_factor_m
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 60:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    dvol_usd = dvol * ret_factor
    adl_m = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or not {"High", "Low", "Close", "Volume"}.issubset(getattr(d, "columns", [])):
            continue
        rng = (d["High"] - d["Low"]).replace(0, np.nan)
        mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
        adl_m[t] = (mfm.fillna(0) * d["Volume"]).cumsum().resample("ME").last().reindex(midx)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def replay(country_ok):
        pick_port, etf_port, spy_port = [], [], []
        beat_etf = beat_spy = npk = 0
        per_sector = {}                       # etf -> [pick_ret - etf_ret]
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = pr = er = 0.0
            for etf in top:
                pharma = etf in PHARMA_ETFS
                pool = [t for t, e in surv_sector.items() if e == etf and t in common]
                pool += [t for t, e in delisted_sector.items() if e == etf and t in common]
                cands = [h for h in pool if h not in held and (country_ok is None or country_ok(h))
                         and _available_at(px_usd[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > MIN_PB
                         and pd.notna(as_traded_usd.loc[date, h]) and as_traded_usd.loc[date, h] >= MIN_PRICE
                         and not bool(trap.loc[date, h]) and pd.notna(dvol_usd.loc[date, h]) and dvol_usd.loc[date, h] >= MIN_DVOL
                         and not (pharma and (pd.isna(mktcap_usd.loc[date, h]) or mktcap_usd.loc[date, h] < MICRO_PHARMA_MIN))]
                g = [x for x in cands if bool(low.loc[date, x])] or cands
                sm = [x for x in g if pd.notna(mktcap_usd.loc[date, x]) and mktcap_usd.loc[date, x] < SMALL]
                g = sm or g
                if not g:
                    continue
                p = min(g, key=lambda h: pb.loc[date, h]); held.add(p)
                r = _ret_delist(px_usd[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                etf_r = etf_m[etf].iloc[i + 1] / etf_m[etf].iloc[i] - 1     # the sector/country INDEX return
                if not np.isfinite(etf_r):
                    etf_r = 0.0
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; pr += w * float(r); er += w * float(etf_r)
                per_sector.setdefault(str(etf), []).append(float(r) - float(etf_r))
                npk += 1; beat_etf += (r > etf_r); beat_spy += (r > sp)
            if wsum <= 0:
                continue
            pick_port.append(pr / wsum); etf_port.append(er / wsum); spy_port.append(float(sp))
        tot = lambda x: float(np.prod(1 + np.asarray(x, float)) - 1) * 100
        return dict(pick=tot(pick_port), etf=tot(etf_port), spy=tot(spy_port), n=npk,
                    beat_etf_pct=round(100 * beat_etf / npk, 1), beat_spy_pct=round(100 * beat_spy / npk, 1),
                    per_sector=per_sector, months=len(pick_port))

    out = {}
    for label, cok in (("usca_small", _is_usca), ("global_small", None)):
        rr = replay(cok)
        out[label] = {k: v for k, v in rr.items() if k != "per_sector"}
        print(f"\n=== {label}: STOCK-PICK vs INDEX-INSTEAD vs SPY ({rr['months']} months, {rr['n']} picks) ===", flush=True)
        print(f"  hold the STOCK PICK   : {rr['pick']:8.1f}%   <- flagship", flush=True)
        print(f"  hold the INDEX instead: {rr['etf']:8.1f}%   (same sectors, buy the ETF not the stock)", flush=True)
        print(f"  hold SPY only         : {rr['spy']:8.1f}%", flush=True)
        print(f"  picks that BEAT their index: {rr['beat_etf_pct']}%   |  picks that beat SPY: {rr['beat_spy_pct']}%", flush=True)
        # sectors where the INDEX beat the pick most (avg pick-minus-index < 0), min 4 picks
        rows = [(tick_to_name.get(e, e), np.mean(v) * 100, len(v)) for e, v in rr["per_sector"].items() if len(v) >= 4]
        worst = sorted(rows, key=lambda x: x[1])[:12]
        print(f"  --- sectors where the INDEX would have beaten the pick (avg pick−index %/mo, n) ---", flush=True)
        for nm, d, n in worst:
            tag = "INDEX better" if d < 0 else "pick better"
            print(f"    {nm[:30]:<31}{d:+7.2f}%/mo  n={n:<3}  [{tag}]", flush=True)
        out[label]["sectors_index_better"] = [{"sector": nm, "pick_minus_index_pct_mo": round(d, 2), "n": n}
                                              for nm, d, n in worst if d < 0]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {OUT}", flush=True)
    print("DONE_INDEX_VS_PICK", flush=True)


if __name__ == "__main__":
    main()
