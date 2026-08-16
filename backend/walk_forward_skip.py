#!/usr/bin/env python3
"""WALK-FORWARD VALIDATION of skip_wide — is the +490pp from skipping speculative/commodity sectors REAL edge or
in-sample fitting? The skip-set must be chosen from PAST data only, then tested on untouched future data.
  (1) SPLIT A: pick skip-sectors from first half -> test on second half (vs baseline no-skip).
  (2) SPLIT B: reverse (pick from second half -> test on first half).
  (3) EXPANDING walk-forward: at each test month, skip-set = sectors that underperformed their index over ALL
      prior months (>=24mo history); compound OOS monthly returns vs baseline.
Skip rule (non-arbitrary, matches index_vs_pick): a sector is skipped if the value-pick's mean monthly return
LAGGED that sector's ETF over the training window (mean(pick-etf) < 0, n>=4). Also reports the hand-picked WIDE
set applied OOS for comparison. -> BacktestResult[walk_forward_skip] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/walk_forward_skip.py
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
from speculative_sector_study import WIDE, STORY

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "walk_forward_skip.json"


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

    def accumulating(name, date):
        a, p = ad_slope3.loc[date].get(name), px_ret3.loc[date].get(name)
        return pd.notna(a) and pd.notna(p) and a > 0 and p < 0

    # ---- ONE pass: record each month's picks (sector, pick_ret, weight, that-sector-ETF-ret) ----
    picks_by_month = []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        held = set(); row = []
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
            r = _ret_delist(px_usd[p], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            er = etf_m[etf].iloc[i + 1] / etf_m[etf].iloc[i] - 1
            if not np.isfinite(er):
                er = 0.0
            w = CONV if accumulating(p, date) else 1.0
            row.append((str(etf), float(r), float(w), float(er)))
        picks_by_month.append((i, row))

    months = [i for i, _ in picks_by_month]
    tot = lambda rets: float(np.prod(1 + np.asarray(rets, float)) - 1) * 100

    def skip_from(idx_set, thresh=0.0, min_n=4):
        agg = {}
        for i, row in picks_by_month:
            if i not in idx_set:
                continue
            for s, r, w, er in row:
                agg.setdefault(s, []).append(r - er)
        return {s for s, v in agg.items() if len(v) >= min_n and np.mean(v) < thresh}

    def port(idx_set, skip):
        rets = []
        for i, row in picks_by_month:
            if i not in idx_set:
                continue
            wsum = rr = 0.0
            for s, r, w, er in row:
                if s in skip:
                    continue
                wsum += w; rr += w * r
            if wsum > 0:
                rets.append(rr / wsum)
        return tot(rets), len(rets)

    def names(sset):
        return sorted(tick_to_name.get(s, s) for s in sset)

    n = len(months); mid = months[n // 2]
    firstH = set(months[: n // 2]); secondH = set(months[n // 2:])
    res = {}

    # SPLIT A: train first half -> test second half
    skA = skip_from(firstH)
    baseA, nA = port(secondH, set()); testA, _ = port(secondH, skA)
    # SPLIT B: train second half -> test first half
    skB = skip_from(secondH)
    baseB, nB = port(firstH, set()); testB, _ = port(firstH, skB)
    # EXPANDING walk-forward (OOS): decide skip from all prior months (>=24mo), apply to current month
    wf_rets, wf_base = [], []
    start_wf = months[24] if len(months) > 30 else months[n // 2]
    for i, row in picks_by_month:
        if i < start_wf:
            continue
        prior = set(j for j in months if j < i)
        sk = skip_from(prior)
        wsum = rr = bwsum = brr = 0.0
        for s, r, w, er in row:
            bwsum += w; brr += w * r
            if s in sk:
                continue
            wsum += w; rr += w * r
        if bwsum > 0:
            wf_base.append(brr / bwsum)
        if wsum > 0:
            wf_rets.append(rr / wsum)
    # full-sample skip set + hand-picked WIDE applied OOS
    skFull = skip_from(set(months))
    wideA, _ = port(secondH, WIDE); wideB, _ = port(firstH, WIDE)

    print("=== WALK-FORWARD VALIDATION of skip_wide ===", flush=True)
    print(f"months={n}  (train/test split at index {mid})", flush=True)
    print(f"\nSPLIT A  train=1st half -> test=2nd half ({nA} mo)", flush=True)
    print(f"  skip-set learned from 1st half: {names(skA)}", flush=True)
    print(f"  test baseline (no skip): {baseA:8.1f}%   |  test WITH learned skip: {testA:8.1f}%   |  hand-picked WIDE: {wideA:8.1f}%", flush=True)
    print(f"  OOS lift from learned skip: {testA-baseA:+.1f}pp", flush=True)
    print(f"\nSPLIT B  train=2nd half -> test=1st half ({nB} mo)", flush=True)
    print(f"  skip-set learned from 2nd half: {names(skB)}", flush=True)
    print(f"  test baseline (no skip): {baseB:8.1f}%   |  test WITH learned skip: {testB:8.1f}%   |  hand-picked WIDE: {wideB:8.1f}%", flush=True)
    print(f"  OOS lift from learned skip: {testB-baseB:+.1f}pp", flush=True)
    print(f"\nEXPANDING walk-forward (skip chosen from history only):", flush=True)
    print(f"  baseline: {tot(wf_base):8.1f}%   |  walk-forward skip: {tot(wf_rets):8.1f}%   |  lift {tot(wf_rets)-tot(wf_base):+.1f}pp   ({len(wf_rets)} mo)", flush=True)
    print(f"\nfull-sample skip-set (hindsight): {names(skFull)}", flush=True)
    print(f"hand-picked WIDE was: {names(WIDE)}", flush=True)
    overlap = skFull & WIDE
    print(f"overlap(full-sample-learned, WIDE): {names(overlap)}", flush=True)

    payload = {
        "split_A": {"train": "1st_half", "test_baseline": baseA, "test_learned_skip": testA, "test_hand_WIDE": wideA,
                    "oos_lift_pp": round(testA - baseA, 1), "learned_skip": names(skA)},
        "split_B": {"train": "2nd_half", "test_baseline": baseB, "test_learned_skip": testB, "test_hand_WIDE": wideB,
                    "oos_lift_pp": round(testB - baseB, 1), "learned_skip": names(skB)},
        "expanding_wf": {"baseline": round(tot(wf_base), 1), "skip": round(tot(wf_rets), 1),
                         "lift_pp": round(tot(wf_rets) - tot(wf_base), 1), "months": len(wf_rets)},
        "full_sample_skip": names(skFull), "hand_WIDE": names(WIDE), "overlap": names(overlap),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="walk_forward_skip",
                                                 defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[walk_forward_skip]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_WF", flush=True)


if __name__ == "__main__":
    main()
