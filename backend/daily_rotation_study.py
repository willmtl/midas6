#!/usr/bin/env python3
"""DAILY vs MONTHLY — run the ENTIRE honest flagship on a DAILY grid instead of month-end. Every trading day:
rank 91 sectors by accel (daily: pct_change(63)-pct_change(63).shift(63) = same 6mo lookback in trading days),
top-10 -> cheapest positive as-traded-P/B guard low-debt pick, $5M floor, div_2x; hold to NEXT day; rebalance
whenever the optimal basket changes. Reports GROSS + TURNOVER + NET-of-cost (daily rebalancing lives or dies on
cost). Compared against the monthly engine (+313%). -> BacktestResult[daily_rotation].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/daily_rotation_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
MIN_DVOL = 5e6
LB = 63                 # ~3 trading months
WARMUP = 2 * LB + 5


def _perf(r, spy, ppy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (ppy / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1),
                sharpe=round(sh, 2), dd=round(dd, 1), periods=n, spy_total=round(sp, 1))


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
    stock_daily = load_candles(all_holds)

    # common daily calendar = SPY trading days
    didx = etf_daily[BENCH].index
    etf_close = pd.DataFrame({t: d["Close"] for t, d in etf_daily.items() if t in etf_tk}).reindex(didx)
    spy = etf_daily[BENCH]["Close"].reindex(didx)
    accel = etf_close.pct_change(LB) - etf_close.pct_change(LB).shift(LB)

    px = pd.DataFrame({t: stock_daily[t]["Close"] for t in all_holds if t in stock_daily}).reindex(didx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, didx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = px.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=didx, columns=common)
    px = px[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    px_at = price_basis.as_traded_close(px, price_basis.refresh_splits(list(common)))
    pb = (px_at * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(252))) & (~(ni > ni.shift(63)))   # 12mo/1q in trading days
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         for t in common if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=didx, columns=common)
    # A/D divergence (63d)
    adl = {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or not {"High", "Low", "Close", "Volume"}.issubset(d.columns):
            continue
        rng = (d["High"] - d["Low"]).replace(0, np.nan)
        mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
        adl[t] = (mfm.fillna(0) * d["Volume"]).cumsum()
    adl = pd.DataFrame(adl).reindex(index=didx, columns=common)
    ad_slope = adl - adl.shift(LB); px_ret = px.pct_change(LB)
    print(f"trading days {len(didx)} | stocks {len(common)}", flush=True)

    # numpy for speed
    cidx = {t: j for j, t in enumerate(common)}
    PX = px.to_numpy(); PB = pb.to_numpy(); TRAP = trap.to_numpy(); LOW = low.to_numpy()
    DV = dvol.to_numpy(); ADS = ad_slope.to_numpy(); PRR = px_ret.to_numpy()
    spy_np = spy.to_numpy()
    sec_cols = {e: [cidx[h] for h in hs if h in cidx] for e, (_, hs) in sector_map.items()}
    etf_cols = list(etf_close.columns); accel_np = accel.to_numpy()
    etf_idx = {e: k for k, e in enumerate(etf_cols)}

    def basket_at(i):
        row = accel_np[i]
        order = np.argsort(row)[::-1]
        top = [etf_cols[k] for k in order if np.isfinite(row[k])][:TOP_N]
        picks = []
        for e in top:
            best = None; best_pb = np.inf; cands_low = []
            for j in sec_cols.get(e, []):
                p = PX[i, j]
                if not (np.isfinite(p) and p > 0):
                    continue
                pbv = PB[i, j]
                if not (np.isfinite(pbv) and pbv > 0):
                    continue
                if TRAP[i, j] == True:
                    continue
                dv = DV[i, j]
                if not (np.isfinite(dv) and dv >= MIN_DVOL):
                    continue
                cands_low.append((j, pbv, bool(LOW[i, j] == True)))
            low_only = [c for c in cands_low if c[2]] or cands_low
            if not low_only:
                continue
            j, pbv, _ = min(low_only, key=lambda c: c[1])
            w = CONVICTION_MULT if (np.isfinite(ADS[i, j]) and np.isfinite(PRR[i, j]) and ADS[i, j] > 0 and PRR[i, j] < 0) else 1.0
            picks.append((j, w))
        return picks

    rets, spies, turns = [], [], []
    prev = set()
    for i in range(WARMUP, len(didx) - 1):
        s = spy_np[i + 1] / spy_np[i] - 1 if (np.isfinite(spy_np[i]) and spy_np[i] > 0) else np.nan
        if not np.isfinite(s):
            continue
        picks = basket_at(i)
        if not picks:
            continue
        tw = sum(w for _, w in picks); pr = 0.0; ok = 0
        for j, w in picks:
            a, b = PX[i, j], PX[i + 1, j]
            if np.isfinite(a) and a > 0 and np.isfinite(b) and b > 0:
                pr += w * (b / a - 1); ok += 1
        if ok == 0:
            continue
        rets.append(pr / tw); spies.append(float(s))
        cur = set(j for j, _ in picks)
        turns.append(1.0 - len(prev & cur) / max(len(cur), 1)); prev = cur

    ppy = 252.0
    gross = _perf(rets, spies, ppy)
    avg_turn = float(np.mean(turns)) if turns else 0.0
    # net: subtract one-sided turnover * cost each day
    def net(cost):
        nr = [r - avg_turn * cost for r in rets]     # approx: pay turnover*cost per day
        return _perf(nr, spies, ppy)
    net5, net10 = net(0.0005), net(0.0010)
    ann_turn = avg_turn * ppy * 100
    print(f"\n=== DAILY rebalance (1-day hold, {gross['periods']} days) ===", flush=True)
    print(f"  GROSS  total {gross['total']}%  vsSPY {gross['vs_spy']}  Sh {gross['sharpe']}  DD {gross['dd']}%", flush=True)
    print(f"  avg 1-sided turnover/day {round(avg_turn*100,1)}%  (~{round(ann_turn)}%/yr)", flush=True)
    print(f"  NET @5bps/side  total {net5['total']}%  Sh {net5['sharpe']}", flush=True)
    print(f"  NET @10bps/side total {net10['total']}%  Sh {net10['sharpe']}", flush=True)
    print(f"  (monthly engine for comparison: +313% / Sh 1.50)", flush=True)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "lookback_days": LB, "min_dvol": MIN_DVOL, "hold": "1 day", "pb_basis": "as-traded"},
        "gross": gross, "net_5bps": net5, "net_10bps": net10,
        "avg_turnover_per_day_pct": round(avg_turn * 100, 2), "annual_turnover_pct": round(ann_turn, 0),
        "monthly_reference": {"total": 313.2, "sharpe": 1.50},
        "verdict": (f"DAILY rebalance: gross +{gross['total']}%/Sh{gross['sharpe']} vs MONTHLY +313%/Sh1.50. "
                    f"1-sided turnover ~{round(avg_turn*100,1)}%/day (~{round(ann_turn)}%/yr) -> NET @5bps +{net5['total']}%/Sh{net5['sharpe']}, "
                    f"@10bps +{net10['total']}%/Sh{net10['sharpe']}. " + (
                    "Daily gross beats monthly but costs decide it — see net." if gross['total'] > 313 else
                    "Daily does NOT beat monthly even GROSS — the slow accel/value signal barely changes day-to-day, so daily "
                    "rebalance just churns; monthly (month-end turn-of-month effect) is the right frequency.")),
        "caveat": "1-day hold, entry/exit at daily close (no intraday). Turnover cost approximated as avg_turnover*cost/day. "
                  "as-traded pb, div_2x, $5M floor, PIT, present-day-holdings survivorship.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/daily_rotation.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="daily_rotation", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[daily_rotation]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
