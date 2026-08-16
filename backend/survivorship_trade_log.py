#!/usr/bin/env python3
"""FULL TRADE LOG for the honest survivorship-adjusted flagship (with_delisted_small): major-exchange gate,
NO price floor (keep penny if big-board listed), cheapest-P/B among <$2B names in accel top-10 sectors, div_2x
conviction on A/D-accumulation. Mirrors survivorship_smallcap_study.build() exactly, but logs EVERY pick:
  date | sector | ticker | entry as-traded price | entry market cap | P/B | weight | 1mo return | delisted?
-> /app/.data/studies/survivorship_trade_log.json  + printed table.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/survivorship_trade_log.py
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
from backtest_lowpb import _monthly_close, BENCH
from survivorship_smallcap_study import (TOP_N, CONV, MIN_DVOL, SMALL, MIN_PRICE, MAJOR_EXCH,
                                         GIC_FILE, GIC_TO_ETF)

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "survivorship_trade_log.json"


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    etf_to_name = {e: n for n, e in config.SECTOR_ETFS.items()}
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    surv_sector, all_holds = {}, set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                surv_sector.setdefault(t, e); all_holds.add(t)

    gic_raw = json.load(open(GIC_FILE))
    from core.models import FinancialReport, DelistedCompany
    dl_have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    dl_exch = {d.ticker: (d.exchange or "").strip()
               for d in DelistedCompany.objects.filter(ticker__in=list(gic_raw))}
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
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol, adl_m = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 60:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    trades = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set()
        for etf in top:
            pool = [t for t, e in surv_sector.items() if e == etf and t in common]
            pool += [t for t, e in delisted_sector.items() if e == etf and t in common]
            cands = [h for h in pool if h not in held
                     and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                     and pd.notna(as_traded.loc[date, h]) and as_traded.loc[date, h] >= MIN_PRICE
                     and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
            g = [x for x in cands if bool(low.loc[date, x])] or cands
            sm = [x for x in g if pd.notna(mktcap.loc[date, x]) and mktcap.loc[date, x] < SMALL]
            g = sm or g
            if not g:
                continue
            p = min(g, key=lambda h: pb.loc[date, h]); held.add(p)
            r = _ret_delist(px[p], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            w = CONV if accumulating(p, date) else 1.0
            mc = float(mktcap.loc[date, p])
            trades.append({
                "date": str(date.date()), "sector_etf": str(etf),
                "sector": etf_to_name.get(str(etf), str(etf)), "ticker": p,
                "entry_price": round(float(as_traded.loc[date, p]), 4),
                "entry_mktcap_musd": round(mc / 1e6, 1),
                "pb": round(float(pb.loc[date, p]), 3),
                "weight": w, "ret_1mo_pct": round(float(r) * 100, 2),
                "delisted": p in delisted_sector,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"config": "with_delisted_small (major-exchange, no price floor)",
                               "n_trades": len(trades), "trades": trades}, indent=2))
    hdr = f"{'date':<12}{'sector':<26}{'ticker':<8}{'entry$':>10}{'mktcap$M':>11}{'P/B':>7}{'wt':>5}{'ret%':>8}  flag"
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    for t in trades:
        flag = "DELISTED" if t["delisted"] else ""
        print(f"{t['date']:<12}{t['sector'][:25]:<26}{t['ticker']:<8}{t['entry_price']:>10.4f}"
              f"{t['entry_mktcap_musd']:>11.1f}{t['pb']:>7.2f}{t['weight']:>5.1f}{t['ret_1mo_pct']:>8.2f}  {flag}",
              flush=True)
    print(f"\nTOTAL TRADES: {len(trades)}  | delisted: {sum(1 for t in trades if t['delisted'])}  "
          f"| months: {len({t['date'] for t in trades})}", flush=True)
    print(f"Saved {OUT}", flush=True)
    print("DONE_TRADELOG", flush=True)


if __name__ == "__main__":
    main()
