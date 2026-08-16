#!/usr/bin/env python3
"""ARE WE ENTERING THE SECTOR TOO LATE? We pick sectors by trailing 6mo momentum — maybe by then the
sector has already run. Three tests on the value pick (cheapest-P/B guarded low-debt in top-N sectors):

  1. LOOKBACK sweep (3/6/9/12mo momentum for sector selection): if SHORTER wins -> we're late with 6mo;
     if LONGER wins -> momentum persists, 6mo is fine or even early.
  2. FRESHNESS: per-pick forward return by how many CONSECUTIVE months the sector has been in the top-N
     (fresh 1-2 = just turned strong; established 5+ = has run a while / possibly late).
  3. ACCELERATION: sector momentum still rising (recent > earlier) vs fading.
-> BacktestResult[sector_timing] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/sector_timing_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings, price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "sector_timing.json"
TOP_N = 10
LBS = [3, 6, 9, 12]


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total_return=0, vs_spy=0, sharpe=0, max_drawdown=0, t_stat=None, win_pct=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total_return=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2),
                max_drawdown=round(dd, 1), t_stat=round(t, 2) if t is not None else None, periods=n)


def _pk(rets):
    a = np.array(rets, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2),
                                        win_pct=round((a > 0).mean() * 100, 1))


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
    mom = {lb: etf_m.pct_change(lb) for lb in LBS}
    mom3_prev = etf_m.pct_change(3).shift(3)          # 3mo momentum as of 3 months ago (for acceleration)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 12

    def pick_in(etf, date, ndate):
        _, holds = sector_map.get(etf, (etf, []))
        cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                 and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
        ld = [c for c in cands if bool(low_debt.loc[date, c])]
        use = ld or cands
        if not use:
            return None
        pick = pb.loc[date, use].idxmin()
        r = _ret_delist(px[pick], date, ndate)
        return float(r) if (r is not None and np.isfinite(r)) else None

    # ---- 1. lookback sweep ----
    lb_res = {}
    for lb in LBS:
        rets, spies, picks_r = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = mom[lb].loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot = [pick_in(e, date, ndate) for e in ranks]
            slot = [x for x in slot if x is not None]
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); picks_r += slot
        s = _stats(rets, spies); s["pick_win_pct"] = round(float((np.array(picks_r) > 0).mean() * 100), 1)
        lb_res[f"{lb}mo"] = s

    # ---- 2. freshness + 3. acceleration (on the 6mo config) ----
    run_len = {e: 0 for e in etf_tk}
    fresh = {"fresh_1-2": [], "mid_3-4": [], "estab_5+": []}
    accel = {"accelerating": [], "fading": []}
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        ranks = list(mom[6].loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        rset = set(ranks)
        for e in etf_tk:
            run_len[e] = run_len[e] + 1 if e in rset else 0
        for etf in ranks:
            r = pick_in(etf, date, ndate)
            if r is None:
                continue
            rl = run_len[etf]
            (fresh["fresh_1-2"] if rl <= 2 else fresh["mid_3-4"] if rl <= 4 else fresh["estab_5+"]).append(r)
            m_now = mom[3].loc[date, etf]; m_prev = mom3_prev.loc[date, etf]
            if pd.notna(m_now) and pd.notna(m_prev):
                (accel["accelerating"] if m_now > m_prev else accel["fading"]).append(r)

    fresh_res = {k: _pk(v) for k, v in fresh.items()}
    accel_res = {k: _pk(v) for k, v in accel.items()}
    print("\n=== 1. MOMENTUM LOOKBACK sweep (sector selection) ===", flush=True)
    for k, s in lb_res.items():
        print(f"  {k:5} vsSPY {s['vs_spy']:>7}%  Sh {s['sharpe']}  DD {s['max_drawdown']}%  pick-win {s['pick_win_pct']}%", flush=True)
    print("\n=== 2. FRESHNESS (months sector in top-10) — per-pick ===", flush=True)
    for k, s in fresh_res.items():
        print(f"  {k:10} mean {s['mean_pct']}%  win {s['win_pct']}%  (n{s['n']})" if s else f"  {k}: –", flush=True)
    print("\n=== 3. ACCELERATION ===", flush=True)
    for k, s in accel_res.items():
        print(f"  {k:12} mean {s['mean_pct']}%  win {s['win_pct']}%  (n{s['n']})" if s else f"  {k}: –", flush=True)

    best_lb = max(lb_res, key=lambda k: lb_res[k]["vs_spy"])
    fresh_better = (fresh_res.get("fresh_1-2") and fresh_res.get("estab_5+")
                    and fresh_res["fresh_1-2"]["win_pct"] > fresh_res["estab_5+"]["win_pct"] + 2)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)), "lookbacks": LBS},
        "lookback_sweep": lb_res, "freshness": fresh_res, "acceleration": accel_res, "best_lookback": best_lb,
        "verdict": (f"Best sector-momentum lookback = {best_lb} (vsSPY {lb_res[best_lb]['vs_spy']}%). "
                    + ("FRESH sectors (just entered top-10) beat established ones -> we ARE entering late; favor "
                       "shorter/fresher momentum." if fresh_better else
                       "Fresh vs established sectors similar -> not clearly entering late; the value pick neutralizes "
                       "sector-timing. Momentum persists enough that 6mo isn't 'late'.")),
        "caveat": "In-sample, no fees, ~5y, 12mo warmup. Freshness/accel are per-pick (overlapping).",
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
            kind="sector_timing", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                            "computed_at": timezone.now()})
        print("Saved BacktestResult[sector_timing]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
