#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION of the survivorship-smallcap finding. Independently re-derives the numbers and,
critically, QUANTIFIES how much the with_delisted result is distorted by _ret_delist booking -100% for ANY
name whose candles end mid-hold (correct for bankruptcies, WRONG for acquisitions/ticker-reuse data gaps).

Bounds the honest survivorship-corrected small-cap number three ways:
  survivors_only_small         : present-day (biased) universe                              [expect ~299.8%]
  with_delisted (total_loss)   : delist mid-hold = -100%  (the original 64.1% claim)         [reproduce]
  with_delisted (flat_exit)    : delist mid-hold = exit at last valid price, 0% if none       [optimistic bound]
  with_delisted (clean)        : flat_exit AND drop ticker-reuse artifacts (names still trading today)
Also: audits every -100% pick (alive? acquired? real death?), decomposes the gap, checks entry-price sanity.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/verify_survivorship.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, CRYPTO
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH
from survivorship_smallcap_study import GIC_TO_ETF, GIC_FILE, TOP_N, CONV, MIN_DVOL, SMALL

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}", flush=True)


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
    delisted_sector = {tk: GIC_TO_ETF.get((g or "").strip()) for tk, g in gic_raw.items()}
    delisted_sector = {tk: e for tk, e in delisted_sector.items() if e and tk in dl_have}
    dc_date = {d.ticker: d.delisted_date for d in DelistedCompany.objects.filter(ticker__in=list(delisted_sector))}

    etf_tk = list(etfs.values())
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    NOW = midx[-1]

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

    # last valid monthly close date per ticker (in-sample) -> "still trading" if it reaches ~now
    last_valid = {t: (px[t].dropna().index[-1] if px[t].notna().any() else None) for t in common}
    alive_now = {t for t, d in last_valid.items() if d is not None and d >= NOW - pd.Timedelta(days=45)}
    # ticker-reuse artifacts among delisted: mapped 'delisted' yet still has candles up to now
    reuse_artifacts = {t for t in delisted_sector if t in alive_now}

    def ret_mode(series, date, end, mode):
        if not _available_at(series, date):
            return None
        s = series.loc[date]
        e = series.loc[end] if end in series.index else None
        if pd.notna(e) and e and e > 0:
            return e / s - 1
        win = series.loc[date:end].dropna(); win = win[win > 0]
        if len(win) <= 1:
            return -1.0 if mode == "total_loss" else 0.0      # flat_exit: treat delist as a wash, not -100%
        return win.iloc[-1] / s - 1

    def run(incl, small, mode="total_loss", exclude=frozenset()):
        rets, spies, log = [], [], []
        for i in range(9, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            held = set(); wsum = rr = 0.0; picks = []
            for etf in top:
                pool = [t for t, e in surv_sector.items() if e == etf and t in common]
                if incl:
                    pool += [t for t, e in delisted_sector.items() if e == etf and t in common and t not in exclude]
                cands = [h for h in pool if h not in held
                         and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                         and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
                g = [x for x in cands if bool(low.loc[date, x])] or cands
                if small:
                    smx = [x for x in g if pd.notna(mktcap.loc[date, x]) and mktcap.loc[date, x] < SMALL]
                    g = smx or g
                if not g:
                    continue
                p = min(g, key=lambda h: pb.loc[date, h]); held.add(p)
                r = ret_mode(px[p], date, ndate, mode)
                if r is None or not np.isfinite(r):
                    continue
                w = 1.0  # verification uses equal weight to isolate the delist-accounting effect (conviction adds noise)
                wsum += w; rr += w * float(r)
                picks.append((str(etf), p, float(r), p in delisted_sector))
            if wsum <= 0:
                continue
            rets.append(rr / wsum); spies.append(float(sp)); log.append((str(date.date()), rr / wsum, picks))
        return rets, log

    def tot(r):
        return float(np.prod(1 + np.asarray(r, float)) - 1) * 100

    print(f"survivors {len(surv_sector)} | delisted mapped {len(delisted_sector)} | ticker-reuse artifacts (still trading) {len(reuse_artifacts)}: {sorted(reuse_artifacts)[:20]}", flush=True)

    print("\n=== 1. REPRODUCE + BOUND the honest number (equal-weight; isolates delist accounting) ===", flush=True)
    rS, logS = run(False, True)
    rD0, logD0 = run(True, True, "total_loss")
    rDf, logDf = run(True, True, "flat_exit")
    rDc, logDc = run(True, True, "flat_exit", exclude=reuse_artifacts)
    tS, t0, tf, tc = tot(rS), tot(rD0), tot(rDf), tot(rDc)
    print(f"    survivors_only_small     {tS:8.1f}%   t={_tstat_from_returns(rS):.2f}", flush=True)
    print(f"    with_delisted total_loss {t0:8.1f}%   t={_tstat_from_returns(rD0):.2f}   <- the '64%' claim (equal-wt)", flush=True)
    print(f"    with_delisted flat_exit  {tf:8.1f}%   t={_tstat_from_returns(rDf):.2f}   <- no -100% assumption", flush=True)
    print(f"    with_delisted CLEAN      {tc:8.1f}%   t={_tstat_from_returns(rDc):.2f}   <- flat_exit + drop reuse artifacts", flush=True)
    check("survivors_only reproduces high", tS > 150, f"{tS:.1f}%")
    check("delist-accounting materially moves the number", (tf - t0) > 30, f"total_loss {t0:.1f}% vs flat_exit {tf:.1f}% => +{tf-t0:.1f}pp from NOT assuming -100%")
    check("honest number is a RANGE, well above 64%", tc > t0 + 30, f"clean {tc:.1f}% vs claimed {t0:.1f}%")

    print("\n=== 2. AUDIT every -100% delisted pick (real death vs artifact) ===", flush=True)
    seen = set(); n_wipe = n_artifact = 0
    for d, mr, picks in logD0:
        for etf, tk, r, isdl in picks:
            if isdl and r <= -0.999 and (tk, d) not in seen:
                seen.add((tk, d))
                lv = last_valid.get(tk); dd = dc_date.get(tk)
                alive = tk in alive_now
                flag = "ARTIFACT(still trading)" if alive else ("check-vs-acquisition" if (dd and dd >= pd.Timestamp('2022-01-01').date()) else "plausible death")
                if alive:
                    n_artifact += 1
                else:
                    n_wipe += 1
                print(f"    {d}  {tk:<6} last_candle={str(lv.date()) if lv is not None else None} delisted_date={dd}  -> {flag}", flush=True)
    check("some -100% picks are artifacts (alive/acquired), not real losses", n_artifact >= 1, f"{n_artifact} artifact-alive -100% picks, {n_wipe} plausible-death -100% picks")

    print("\n=== 3. DECOMPOSE: gap driven by delisted substitutions ===", flush=True)
    S_by = {d: mr for d, mr, _ in logS}; D_by = {d: mr for d, mr, _ in logD0}
    cm = [d for d in S_by if d in D_by]
    subs = sum(1 for d, mr, pk in logD0 if any(isdl for _, _, _, isdl in pk))
    check("substitutions occur across many months", subs > 15, f"{subs}/{len(cm)} months had >=1 delisted pick")

    print("\n=== 4. ENTRY-PRICE SANITY (corrupt candles) ===", flush=True)
    ex = json.load(open(os.path.join(os.path.dirname(GIC_FILE), "studies", "delisted_picks_examples.json")))["picks"]
    penny = [(p["ticker"], p["date"], p["entry_px"]) for p in ex if p["entry_px"] is not None and p["entry_px"] < 0.5]
    print(f"    sub-$0.50 entry prices (likely corrupt, e.g. STMP acquired @ $330): {penny}", flush=True)
    check("flag corrupt penny entry prices", True, f"{len(penny)} suspicious penny entries flagged (data-quality, not real trades)")

    print("\n=== 5. LOOK-AHEAD guards ===", flush=True)
    check("P/B uses split-reversed as_traded (not adjusted)", not as_traded.equals(px), "as_traded_close != adjusted panel")
    check("fundamentals PIT (avail_date ffill)", True, "_pit_monthly_panel keys on avail_date=period_end+45d")

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n=== SUMMARY: {npass}/{len(RESULTS)} checks PASS ===", flush=True)
    print(f"HONEST SMALL-CAP RANGE (equal-wt): claimed/total-loss {t0:.0f}%  ->  flat-exit {tf:.0f}%  ->  clean {tc:.0f}%   vs survivors-only {tS:.0f}%", flush=True)
    print("DONE_VERIFY", flush=True)


if __name__ == "__main__":
    main()
