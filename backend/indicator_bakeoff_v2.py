#!/usr/bin/env python3
"""INDICATOR BAKEOFF v2 — test Mom(10), Awesome Oscillator, CCI(20), Stochastic(14,3,3), MACD(12,26) as ENTRY
overlays on the honest usca_small flagship (790%). Each indicator is computed on the PICKED stock's daily series
as-of entry and used as a gate: take the value-pick only if the condition holds, else no position that sleeve.
Tested in BOTH directions (confirmation/bullish vs oversold) because the project prior ([[indicator-bakeoff]],
[[entry-signal-value-pick]]) is 'oversold helps tiny, trend/confirmation SUBTRACTS'. -> BacktestResult[indicator_bakeoff_v2].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/indicator_bakeoff_v2.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "indicator_bakeoff_v2.json"


def _indicator_panels(stock_daily, common, midx):
    """month-end value of each daily indicator per ticker."""
    mom, ao, cci, stk, macdh = {}, {}, {}, {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or len(d) < 60 or not {"High", "Low", "Close"}.issubset(d.columns):
            continue
        c, h, l = d["Close"], d["High"], d["Low"]
        me = lambda s: s.resample("ME").last().reindex(midx)
        mom[t] = me(c.pct_change(10) * 100)                                   # Mom(10) %
        med = (h + l) / 2
        ao[t] = me(med.rolling(5).mean() - med.rolling(34).mean())            # Awesome Oscillator
        tp = (h + l + c) / 3
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        cci[t] = me((tp - tp.rolling(20).mean()) / (0.015 * mad.replace(0, np.nan)))  # CCI(20)
        ll, hh = l.rolling(14).min(), h.rolling(14).max()
        k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()  # Stoch %K (14,3,3)
        stk[t] = me(k)
        e12, e26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
        line = e12 - e26
        macdh[t] = me(line - line.ewm(span=9).mean())                          # MACD(12,26) histogram
    P = lambda dd: pd.DataFrame(dd).reindex(index=midx, columns=common)
    return {"mom": P(mom), "ao": P(ao), "cci": P(cci), "stochk": P(stk), "macdh": P(macdh)}


def main():
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
    sh, eq, ni, dtb = (_pit_monthly_panel(reps, f, midx) for f in
                       ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dtb = R(sh), R(eq), R(ni), R(dtb)
    as_traded = price_basis.as_traded_close(px)
    mktcap = as_traded * sh
    pb = mktcap / eq.where(eq != 0)
    usd_factor_m, _ = _usd_factor_matrix(list(common), midx)
    ret_factor = usd_factor_m.fillna(1.0)
    px_usd = px * ret_factor; as_traded_usd = as_traded * ret_factor
    mktcap_usd = mktcap * usd_factor_m
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dtb / eq.where(eq != 0)) < 1.0
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
    IND = _indicator_panels(stock_daily, list(common), midx)
    print("indicator panels built", flush=True)

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    def gate_ok(gate, p, date):
        if gate == "baseline":
            return True
        v = IND[gate.split(":")[0]].loc[date, p]
        if pd.isna(v):
            return True                       # missing indicator -> don't filter (neutral)
        cond = gate.split(":")[1]
        return {"pos": v > 0, "neg": v < 0, "gt100": v > 100, "lt-100": v < -100,
                "lt20": v < 20, "gt80": v > 80}[cond]

    GATES = ["baseline",
             "mom:pos", "mom:neg", "ao:pos", "ao:neg", "cci:gt100", "cci:lt-100",
             "stochk:lt20", "stochk:gt80", "macdh:pos", "macdh:neg"]

    def run(gate):
        rets = []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0
            for etf in top:
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
                if not gate_ok(gate, p, date):
                    continue                                    # indicator gate rejects entry
                r = _ret_delist(px_usd[p], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                w = CONV if accumulating(p, date) else 1.0
                wsum += w; rr += w * float(r)
            if wsum > 0:
                rets.append(rr / wsum)
        r = np.asarray(rets, float); n = len(r)
        tot = float(np.prod(1 + r) - 1) * 100
        shp = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
        eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
        return dict(total=round(tot, 1), sharpe=round(shp, 2), dd=round(dd, 1),
                    t=round(_tstat_from_returns(list(r)) or 0, 2), months=n)

    results = {ggate: run(ggate) for ggate in GATES}
    base = results["baseline"]["total"]
    print("\n=== INDICATOR BAKEOFF v2 (entry overlay on usca_small; baseline=no gate) ===", flush=True)
    print(f"  {'gate':<16}{'total':>9}{'Sharpe':>8}{'DD':>8}{'t':>7}{'vs base':>10}", flush=True)
    for gt in GATES:
        r = results[gt]
        print(f"  {gt:<16}{r['total']:>8}%{r['sharpe']:>8}{r['dd']:>7}%{r['t']:>7}{r['total']-base:>+9.1f}", flush=True)
    winners = [g for g in GATES if g != "baseline" and results[g]["total"] > base]
    print(f"\n  gates that BEAT baseline ({base}%): {winners or 'NONE'}", flush=True)

    payload = {"baseline": base, "results": results, "note": "entry overlay; gate rejects the pick if indicator fails."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="indicator_bakeoff_v2",
                                                 defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[indicator_bakeoff_v2]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_INDV2", flush=True)


if __name__ == "__main__":
    main()
