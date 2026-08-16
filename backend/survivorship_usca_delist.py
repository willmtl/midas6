#!/usr/bin/env python3
"""US/CANADA picks that DELISTED WHILE WE HELD THEM — replays the winning usca_small config and flags every
pick whose price series ENDS inside the held month (no price at the next rebalance = delisted/bankrupt during
the hold), with entry price (USD), P/B, mktcap (USD), the return we actually realized (_ret_delist = exit at
last traded price), last traded date/price, and the DelistedCompany record (exchange, delisted_date).
-> /app/.data/studies/survivorship_usca_delist.json  + printed table.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/survivorship_usca_delist.py
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
from survivorship_smallcap_study import (TOP_N, CONV, MIN_DVOL, SMALL, MIN_PRICE, MIN_PB, MAJOR_EXCH,
                                         GIC_FILE, GIC_TO_ETF, _usd_factor_matrix, _is_usca)

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "survivorship_usca_delist.json"


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
    dl_meta = {d.ticker: d for d in DelistedCompany.objects.filter(ticker__in=list(gic_raw))}
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

    # last valid monthly close date per ticker
    last_valid = {t: (px[t].dropna().index[-1] if px[t].notna().any() else None) for t in common}

    events = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set()
        for etf in top:
            pool = [t for t, e in surv_sector.items() if e == etf and t in common]
            pool += [t for t, e in delisted_sector.items() if e == etf and t in common]
            cands = [h for h in pool if h not in held and _is_usca(h)
                     and _available_at(px_usd[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > MIN_PB
                     and pd.notna(as_traded_usd.loc[date, h]) and as_traded_usd.loc[date, h] >= MIN_PRICE
                     and not bool(trap.loc[date, h]) and pd.notna(dvol_usd.loc[date, h]) and dvol_usd.loc[date, h] >= MIN_DVOL]
            g = [x for x in cands if bool(low.loc[date, x])] or cands
            sm = [x for x in g if pd.notna(mktcap_usd.loc[date, x]) and mktcap_usd.loc[date, x] < SMALL]
            g = sm or g
            if not g:
                continue
            p = min(g, key=lambda h: pb.loc[date, h]); held.add(p)
            lv = last_valid.get(p)
            # DELISTED WHILE HELD: series ends inside the held month (no price at next rebalance)
            if not (p in delisted_sector and lv is not None and date <= lv < ndate):
                continue
            r = _ret_delist(px_usd[p], date, ndate)
            m = dl_meta.get(p)
            events.append({
                "buy_date": str(date.date()), "sector": etf_to_name.get(str(etf), str(etf)),
                "ticker": p, "company": (m.name if m else ""), "exchange": (dl_exch.get(p) or ""),
                "entry_price_usd": round(float(as_traded_usd.loc[date, p]), 4),
                "pb": round(float(pb.loc[date, p]), 3),
                "entry_mktcap_musd": round(float(mktcap_usd.loc[date, p]) / 1e6, 1),
                "realized_ret_pct": (round(float(r) * 100, 2) if r is not None and np.isfinite(r) else None),
                "last_traded_date": str(lv.date()), "last_price_usd": round(float(px_usd[p].dropna().iloc[-1]), 4),
                "delisted_date": (str(m.delisted_date) if m and m.delisted_date else None),
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"config": "usca_small (US+Canada, <$2B USD, P/B>=0.1, FX-USD returns)",
                               "n_events": len(events), "events": events}, indent=2))
    hdr = (f"{'buy_date':<12}{'sector':<22}{'ticker':<8}{'entry$':>9}{'P/B':>6}{'mcap$M':>9}"
           f"{'realized%':>10}{'last_date':>12}{'last$':>9}  company")
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    for e in sorted(events, key=lambda x: x["buy_date"]):
        print(f"{e['buy_date']:<12}{e['sector'][:21]:<22}{e['ticker']:<8}{e['entry_price_usd']:>9.3f}{e['pb']:>6.2f}"
              f"{e['entry_mktcap_musd']:>9.1f}{str(e['realized_ret_pct']):>10}{e['last_traded_date']:>12}"
              f"{e['last_price_usd']:>9.3f}  {e['company']}", flush=True)
    print(f"\nUS/CANADA DELISTED-WHILE-HELD: {len(events)} events", flush=True)
    print(f"Saved {OUT}", flush=True)
    print("DONE_USCA_DELIST", flush=True)


if __name__ == "__main__":
    main()
