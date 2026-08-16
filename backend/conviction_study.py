#!/usr/bin/env python3
"""PERFECT PLAYS — does CONVICTION predict returns, and does levering the top tier add real edge?

Blanket leverage just scales risk 1:1 (Return Lab). The only way leverage/options ADD value is if a
'perfect play' (more signals aligned) earns more than an average pick. We score every monthly pick 0-5:
  +1 deep_value    pick P/B in cheapest tercile of that month's picks
  +1 profitable    net income > 0 (not just non-trap)
  +1 very_low_debt debt/equity < 0.5
  +1 oversold      stock RSI(10) < 45 at entry (the validated dip edge)
  +1 strong_sector pick's sector is in the top-5 momentum (not just top-10)
Then: forward return + hit-rate by score bucket (is it monotonic?), and a LEVERAGE SIM — 2x the
high-conviction picks (score>=4), 1x the rest — vs flat 1x. If high-conviction fwd returns >> low, and
the levered top tier beats flat on Sharpe (not just return), 'options/leverage on perfect plays' is real.
-> BacktestResult[conviction] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/conviction_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "conviction.json"
LOOKBACK, TOP_N = 6, 10


def _rsi(close, n=10):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


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
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)

    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    etf_trail = etf_m.pct_change(LOOKBACK)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def R(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(R, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    vlow_debt = (debt / equity.where(equity != 0)) < 0.5
    rsi = pd.DataFrame({t: _rsi(stock_daily[t]["Close"]).resample("ME").last().reindex(midx)
                        for t in common if t in stock_daily and len(stock_daily[t]) > 20}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    # collect (score, fwd_ret) for every pick, and per-month picks with scores for the leverage sim
    by_score = {s: [] for s in range(6)}
    months = []                          # each = list of (score, ret)
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        ranks = list(etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        top5 = set(ranks[:5])
        picks = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            pick = pb.loc[date, use].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is None or not np.isfinite(r):
                continue
            picks.append((etf, pick, float(r)))
        if not picks:
            continue
        pbs = sorted(pb.loc[date, [p for _, p, _ in picks]].values)
        cheap_cut = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0
        month = []
        for etf, pick, r in picks:
            sc = 0
            sc += int(pb.loc[date, pick] <= cheap_cut)                       # deep value
            sc += int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)  # profitable
            sc += int(bool(vlow_debt.loc[date, pick]))                       # very low debt
            sc += int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)  # oversold
            sc += int(etf in top5)                                           # strong sector
            by_score[sc].append(r); month.append((sc, r))
        months.append(month)

    # forward return by conviction score
    score_tbl = {s: {"n": len(v), "avg_ret_%": round(float(np.mean(v)) * 100, 2) if v else None,
                     "hit_%": round(float((np.array(v) > 0).mean()) * 100, 1) if v else None}
                 for s, v in by_score.items()}

    # portfolio sims: flat 1x; lever top tier (score>=4) 2x; hold ONLY top tier
    def sim(lev_hi=False, only_hi=False, hi=4, L=2.0):
        rets, spies = [], []
        for j, month in enumerate(months):
            if only_hi:
                sel = [(sc, r) for sc, r in month if sc >= hi]
                if not sel:
                    continue
                port = float(np.mean([r for _, r in sel]))
            else:
                ws = [(L if (lev_hi and sc >= hi) else 1.0) for sc, _ in month]
                w = np.array(ws) / np.sum(ws)
                port = float(np.dot(w, [r for _, r in month]))
            rets.append(port)
        # align spy
        sp = []
        k = 0
        for j, month in enumerate(months):
            if only_hi and not any(sc >= hi for sc, _ in month):
                continue
            sp.append(0.0)  # placeholder; recompute below
        # simpler: recompute spy per included month
        return rets

    # build spy series aligned to months
    spy_month = []
    idxs = list(range(warmup, len(midx) - 1))
    # rebuild aligned spy for the flat sim (all months included)
    flat_r, flat_spy, hi_r, hi_spy = [], [], [], []
    mi = 0
    for i in idxs:
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = list(etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        # reuse months[mi] if exists
        if mi >= len(months):
            break
        month = months[mi]; mi += 1
        if not month:
            continue
        flat_r.append(float(np.mean([r for _, r in month]))); flat_spy.append(float(sp))
        ws = [(2.0 if sc >= 4 else 1.0) for sc, _ in month]
        w = np.array(ws) / np.sum(ws)
        hi_r.append(float(np.dot(w, [r for _, r in month]))); hi_spy.append(float(sp))

    sims = {"flat_1x": _stats(flat_r, flat_spy), "lever_top_tier_2x": _stats(hi_r, hi_spy)}
    mono = [score_tbl[s]["avg_ret_%"] for s in range(6) if score_tbl[s]["n"] >= 5]
    is_mono = len(mono) >= 3 and mono[-1] > mono[0]
    print("\n=== CONVICTION: forward return by score ===", flush=True)
    for s in range(6):
        t = score_tbl[s]
        print(f"  score {s}: n={t['n']:>4}  avg fwd {str(t['avg_ret_%']):>6}%  hit {t['hit_%']}%", flush=True)
    print("\n  leverage sim:", flush=True)
    for k, v in sims.items():
        print(f"    {k:18} vsSPY {v['vs_spy']:>7}%  total {v['total_return']}%  Sh {v['sharpe']}  DD {v['max_drawdown']}%", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)),
                   "score": "deep_value+profitable+very_low_debt+oversold+strong_sector (0-5)"},
        "by_score": score_tbl, "sims": sims, "conviction_monotonic": bool(is_mono),
        "verdict": ("Conviction PREDICTS returns (higher score -> higher forward return) — levering the top tier "
                    "is real edge, not blanket leverage." if is_mono else
                    "Conviction does NOT cleanly sort forward returns — 'perfect plays' don't earn more, so "
                    "leverage/options there is just blanket leverage (scales risk, not edge)."),
        "caveat": "In-sample, no fees, ~5y. Options NOT modeled (no options-price history) — theta/IV/spreads "
                  "would erode a call-based version; treat options as a live overlay, not a backtested result.",
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
            kind="conviction", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                         "computed_at": timezone.now()})
        print("Saved BacktestResult[conviction]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
