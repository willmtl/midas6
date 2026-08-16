#!/usr/bin/env python3
"""CONCRETE EXAMPLES: which DELISTED names would the cheapest-P/B small-cap engine actually have BOUGHT?
Replays the with_delisted_small arm of survivorship_smallcap_study and, for every pick that is a delisted
name, records entry date/sector/P/B/mktcap, the 1-month realized return, and the FULL damage from entry
price to the last traded price before delisting (the value-trap punchline). -> prints top losers + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/delisted_picks_examples.py
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
from survivorship_smallcap_study import GIC_TO_ETF, GIC_FILE, TOP_N, CONV, MIN_DVOL, SMALL

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "delisted_picks_examples.json"


def build():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    surv_sector, all_holds = {}, set()
    for n, e in etfs.items():
        for t in sector_holdings.get_holdings(n):
            if t not in (e, BENCH) and t not in CRYPTO:
                surv_sector.setdefault(t, e); all_holds.add(t)
    gic_raw = json.load(open(GIC_FILE))
    from core.models import FinancialReport
    dl_have = set(FinancialReport.objects.values_list("ticker", flat=True).distinct())
    delisted_sector = {tk: GIC_TO_ETF.get((g or "").strip()) for tk, g in gic_raw.items()}
    delisted_sector = {tk: e for tk, e in delisted_sector.items() if e and tk in dl_have}
    print(f"survivors {len(surv_sector)} | delisted mapped {len(delisted_sector)}", flush=True)

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
    dvol = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 60:
            continue
        dvol[t] = (d["Close"] * d["Volume"]).rolling(20).mean().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)

    def sector_cands(etf):
        return ([t for t, e in surv_sector.items() if e == etf and t in common] +
                [t for t, e in delisted_sector.items() if e == etf and t in common])

    picks = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set()
        for etf in top:
            cands = [h for h in sector_cands(etf) if h not in held
                     and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                     and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
            g = [x for x in cands if bool(low.loc[date, x])] or cands
            sm = [x for x in g if pd.notna(mktcap.loc[date, x]) and mktcap.loc[date, x] < SMALL]
            g = sm or g
            if not g:
                continue
            p = min(g, key=lambda h: pb.loc[date, h])
            held.add(p)
            if p not in delisted_sector:
                continue
            r1 = _ret_delist(px[p], date, ndate)                      # realized 1-month (rotation) return
            ser = px[p].dropna()
            entry = float(px.loc[date, p]) if pd.notna(px.loc[date, p]) else (float(ser.loc[:date].iloc[-1]) if len(ser.loc[:date]) else np.nan)
            after = ser.loc[date:]
            last_px = float(after.iloc[-1]) if len(after) else np.nan
            last_dt = after.index[-1] if len(after) else None
            damage = (last_px / entry - 1) if (entry and np.isfinite(entry) and np.isfinite(last_px)) else np.nan
            months_to_end = int(round((last_dt - date).days / 30.4)) if last_dt is not None else None
            picks.append(dict(
                date=str(date.date()), ticker=p, sector_etf=etf,
                pb=round(float(pb.loc[date, p]), 2),
                mktcap_musd=round(float(mktcap.loc[date, p]) / 1e6, 1) if pd.notna(mktcap.loc[date, p]) else None,
                ret_1mo_pct=round(float(r1) * 100, 1) if (r1 is not None and np.isfinite(r1)) else None,
                entry_px=round(entry, 2) if np.isfinite(entry) else None,
                last_px=round(last_px, 2) if np.isfinite(last_px) else None,
                last_date=str(last_dt.date()) if last_dt is not None else None,
                months_to_delist=months_to_end,
                total_damage_pct=round(float(damage) * 100, 1) if pd.notna(damage) else None,
            ))
    return picks


def main():
    picks = build()
    picks_sorted = sorted(picks, key=lambda x: (x["total_damage_pct"] is None, x["total_damage_pct"] if x["total_damage_pct"] is not None else 0))
    print(f"\n=== DELISTED NAMES THE SMALL-CAP ENGINE WOULD HAVE BOUGHT ({len(picks)} picks) ===", flush=True)
    print(f"  {'entry':<11}{'ticker':<9}{'sec':<6}{'P/B':>6}{'mcap$M':>9}{'1mo%':>7}{'entry':>9}{'last':>9}{'mo->DL':>7}{'DAMAGE%':>9}", flush=True)
    for p in picks_sorted[:25]:
        print(f"  {p['date']:<11}{p['ticker']:<9}{p['sector_etf']:<6}{p['pb']:>6}{str(p['mktcap_musd']):>9}"
              f"{str(p['ret_1mo_pct']):>7}{str(p['entry_px']):>9}{str(p['last_px']):>9}{str(p['months_to_delist']):>7}{str(p['total_damage_pct']):>9}", flush=True)
    dmg = [p["total_damage_pct"] for p in picks if p["total_damage_pct"] is not None]
    wipeouts = [p for p in picks if p["total_damage_pct"] is not None and p["total_damage_pct"] <= -80]
    print(f"\n  {len(picks)} delisted picks | median total damage {np.median(dmg):.0f}% | "
          f"{len(wipeouts)} went to <=-80% (near-total wipeout to delisting)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n": len(picks), "picks": picks_sorted}, indent=2, default=str))
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
