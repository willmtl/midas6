#!/usr/bin/env python3
"""SPECULATIVE-SECTOR REFINEMENT — index_vs_pick showed the cheapest-P/B stock is a TRAP in story-stock sectors
(Space −9.7%/mo, Quantum −7.6, ARKK −3.6, Software −1.6): the value name is the laggard while the index rides the
momentum leaders. Test whether, in those sectors, we should HOLD THE ETF or SKIP entirely instead of stock-picking.
Modes per SPEC set: baseline (always stock-pick) | skip (no position in SPEC sectors) | etf (hold the ETF there).
Runs on the usca_small flagship config. -> BacktestResult[speculative_sector] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/speculative_sector_study.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "speculative_sector.json"
STORY = {"UFO", "QTUM", "ARKK", "IGV"}                       # pure story/growth sectors
WIDE = STORY | {"SLV", "XLE", "SLX", "COPX", "TAN", "PRNT"}  # + commodity/miner + solar/3dprint laggards


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

    def perf(rets, spies):
        r = np.asarray(rets, float); n = len(r)
        tot = float(np.prod(1 + r) - 1) * 100
        ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
        sh_ = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
        eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
        return dict(total=round(tot, 1), annual=round(ann, 1), sharpe=round(sh_, 2), dd=round(dd, 1),
                    t=round(_tstat_from_returns(list(r)) or 0, 2), months=n)

    def run(mode, spec):
        rets, spies = [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
                is_spec = str(etf) in spec
                if is_spec and mode == "skip":
                    continue
                if is_spec and mode == "etf":
                    er = etf_m[etf].iloc[i + 1] / etf_m[etf].iloc[i] - 1
                    if np.isfinite(er):
                        wsum += 1.0; rr += 1.0 * float(er)
                    continue
                pharma = etf in PHARMA_ETFS
                pool = [t for t, e in surv_sector.items() if e == etf and t in common]
                pool += [t for t, e in delisted_sector.items() if e == etf and t in common]
                cands = [h for h in pool if h not in held and _is_usca(h)
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
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp))
        return perf(rets, spies)

    variants = {
        "baseline": run("baseline", set()),
        "skip_story": run("skip", STORY),
        "etf_story": run("etf", STORY),
        "skip_wide": run("skip", WIDE),
        "etf_wide": run("etf", WIDE),
    }
    print("=== SPECULATIVE-SECTOR REFINEMENT (usca_small flagship) ===", flush=True)
    print(f"  STORY={sorted(STORY)}", flush=True)
    print(f"  WIDE = STORY + {sorted(WIDE - STORY)}", flush=True)
    print(f"  {'variant':<14}{'total':>9}{'annual':>8}{'Sharpe':>8}{'DD':>8}{'t':>7}", flush=True)
    for k, r in variants.items():
        print(f"  {k:<14}{r['total']:>8}%{r['annual']:>7}%{r['sharpe']:>8}{r['dd']:>7}%{r['t']:>7}", flush=True)
    base = variants["baseline"]["total"]
    print(f"\n  baseline {base}% -> best refinement: "
          f"{max(variants, key=lambda k: variants[k]['total'])} {max(v['total'] for v in variants.values())}%", flush=True)

    payload = {"variants": variants, "STORY": sorted(STORY), "WIDE": sorted(WIDE),
               "note": "index_vs_pick showed cheapest-P/B lags the index in story sectors; test skip vs etf-substitute."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="speculative_sector",
                                                 defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[speculative_sector]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_SPEC", flush=True)


if __name__ == "__main__":
    main()
