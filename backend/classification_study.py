#!/usr/bin/env python3
"""CLASSIFICATION GROUPING (parallel approach — does NOT replace the ETF-membership strategy).

Instead of grouping stocks by curated ETF holdings, group them by their INTRINSIC GICS classification
(Fundamental.sector / .industry) — an attribute that exists for every stock INCLUDING delisted names, so
it dissolves the ETF-membership survivorship problem. Rotate into the top-momentum GROUPS (group momentum
= equal-weight mean of member 6mo returns), then pick the cheapest positive-P/B name passing guard +
low-debt in each — same selection rules as the validated engine, only the GROUPING changes.

Runs both grains: industry (~127 groups, top-15) and sector (~12 groups, top-4). Reports vs SPY / t /
Sharpe / DD next to the ETF-membership baseline (+229% vs SPY) so the grouping method is compared apples
-to-apples. NOTE: current classified universe only — delisted-inclusive (fully survivorship-correct) is
the next data step; this prototype tests whether classification-grouping matches ETF-grouping.
-> BacktestResult[classification_study] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/classification_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
from core.models import Fundamental

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "classification_study.json"
LOOKBACK = 6
GRAINS = {"industry": {"top_n": 15, "min_members": 4}, "sector": {"top_n": 4, "min_members": 8}}
ETF_BASELINE_VS_SPY = 229.4          # the validated ETF-membership engine, for side-by-side reference


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


def build():
    # classification map (current snapshot; a stock's sector/industry is ~static -> fine as grouping key)
    cls = {}
    for tk, sec, ind in Fundamental.objects.exclude(sector__isnull=True).exclude(sector="") \
            .values_list("ticker", "sector", "industry"):
        if sec == "ETF":
            continue
        cls[tk] = {"sector": (sec or "").strip(), "industry": (ind or "").strip() or (sec or "").strip()}
    tickers = sorted(cls.keys())
    print(f"classified stocks: {len(tickers)}", flush=True)

    daily = load_candles(tickers + [BENCH])
    stock_m = _monthly_close({t: d for t, d in daily.items() if t in tickers})
    spy_daily = daily.get(BENCH)
    midx = stock_m.index
    spy_m = spy_daily["Close"].resample("ME").last().reindex(midx)

    reps = load_financial_reports(tickers)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    mktcap = px * shares
    pb = mktcap / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    stock_mom = px.pct_change(LOOKBACK)
    print(f"months {len(midx)} | stocks with PIT {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def run(grain, mom_mode):
        key = grain
        top_n = GRAINS[grain]["top_n"]; min_m = GRAINS[grain]["min_members"]
        # group -> member columns present in `common`
        groups = {}
        for t in common:
            g = cls.get(t, {}).get(key)
            if g:
                groups.setdefault(g, []).append(t)
        groups = {g: m for g, m in groups.items() if len(m) >= min_m}
        # group momentum panel: equal-weight mean, or CAP-WEIGHTED (mirrors how an ETF's momentum works)
        if mom_mode == "capwt":
            gcols = {}
            for g, m in groups.items():
                num = (stock_mom[m] * mktcap[m]).sum(axis=1, min_count=1)
                den = mktcap[m].where(stock_mom[m].notna()).sum(axis=1, min_count=1)
                gcols[g] = num / den
            gmom = pd.DataFrame(gcols, index=midx)
        else:
            gmom = pd.DataFrame({g: stock_mom[m].mean(axis=1) for g, m in groups.items()}, index=midx)
        rets, spies, nn = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = gmom.loc[date].dropna().sort_values(ascending=False).head(top_n).index
            slot = []
            for g in ranks:
                cands = [h for h in groups[g] if _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                if not use:
                    continue
                pick = pb.loc[date, use].idxmin()
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); nn.append(len(slot))
        s = _stats(rets, spies)
        s["avg_names"] = round(float(np.mean(nn)), 1) if nn else 0
        s["groups"] = len(groups)
        s["top_n"] = top_n
        s["grain"] = grain
        s["mom_mode"] = mom_mode
        return s

    results = {f"{g}_{mode}": run(g, mode) for g in GRAINS for mode in ("equal", "capwt")}
    print("\n=== CLASSIFICATION GROUPING — fair rematch (equal vs CAP-WEIGHTED momentum) ===", flush=True)
    print(f"  ETF-membership baseline (reference): vsSPY +{ETF_BASELINE_VS_SPY}%", flush=True)
    for k, s in results.items():
        print(f"  {k:16} ({s['groups']} groups, top {s['top_n']}, {s['mom_mode']}-mom): vsSPY {s['vs_spy']:>7}%  "
              f"t={s['t_stat']}  Sh {s['sharpe']}  DD {s['max_drawdown']}%  names {s['avg_names']}", flush=True)

    best = max(results, key=lambda g: results[g]["vs_spy"])
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "benchmark": BENCH, "months": int(len(midx)),
                   "grains": GRAINS, "note": "PARALLEL approach; does not replace ETF-membership strategy. "
                   "Groups by intrinsic GICS classification (survivorship-friendly). Current classified "
                   "universe only — delisted-inclusive is the next data step."},
        "etf_membership_baseline_vs_spy": ETF_BASELINE_VS_SPY,
        "results": results, "best_grain": best,
        "verdict": (f"Classification grouping by {best} = {results[best]['vs_spy']}% vs SPY "
                    f"({'matches/beats' if results[best]['vs_spy'] >= ETF_BASELINE_VS_SPY - 20 else 'trails'} "
                    f"the ETF-membership baseline +{ETF_BASELINE_VS_SPY}%). Grouping method is viable; the real "
                    "win comes when delisted names are added (survivorship-correct), which ETF membership can't do."),
        "caveat": "Static current classification applied over history (industries ~stable). Current universe "
                  "only (survivorship-limited like the ETF version until delisted names are backfilled). "
                  "Directional, no fees, ~5y single regime.",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="classification_study", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                                   "computed_at": timezone.now()})
        print("Saved BacktestResult[classification_study]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
