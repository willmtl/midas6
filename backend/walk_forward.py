#!/usr/bin/env python3
"""WALK-FORWARD / SUBPERIOD VALIDATION — is the validated engine's +229% persistent or concentrated?

The engine is rules-based (no fitted parameters), so out-of-sample validation = subperiod stability: does
the alpha show up in EVERY slice of the sample, or does one lucky stretch carry it? Runs the clean top-20
engine (rotation 6mo momentum -> cheapest-P/B guarded low-debt), then decomposes its monthly return series:
  - first half vs second half (vs SPY, t, monthly hit-rate vs SPY)
  - per calendar year
  - rolling 12-month vs-SPY (fraction of windows the strategy beats SPY)
A robust edge beats SPY in BOTH halves and most years; a fragile one is carried by a single period.
-> BacktestResult[walk_forward] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/walk_forward.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "walk_forward.json"
LOOKBACK, TOP_N = 6, 10


def _seg(rets, spies, dates):
    r = np.asarray(rets, float); s = np.asarray(spies, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, strat=0, spy=0, t_stat=None, hit_rate=0, months=0)
    strat = float(np.prod(1 + r) - 1) * 100
    spy = float(np.prod(1 + s) - 1) * 100
    hit = float((r > s).mean() * 100)
    t = _tstat_from_returns(list(r - s))            # t of the monthly EXCESS return vs SPY
    return dict(vs_spy=round(strat - spy, 1), strat=round(strat, 1), spy=round(spy, 1),
                t_stat=round(t, 2) if t is not None else None, hit_rate=round(hit, 1), months=n)


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.HOLDINGS.get(n, {}).get("holdings", []) if t not in (e, BENCH) and t not in CRYPTO]
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
    warmup = max(LOOKBACK, 1)

    rets, spies, dates = [], [], []
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        slot = []
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
            if r is not None and np.isfinite(r):
                slot.append(float(r))
        if slot:
            rets.append(float(np.mean(slot))); spies.append(float(sp)); dates.append(ndate)
    rets, spies, dates = np.array(rets), np.array(spies), pd.to_datetime(dates)
    n = len(rets)
    print(f"months {len(midx)} | realized periods {n}", flush=True)

    full = _seg(rets, spies, dates)
    h = n // 2
    halves = {"first_half": _seg(rets[:h], spies[:h], dates[:h]),
              "second_half": _seg(rets[h:], spies[h:], dates[h:])}
    thirds = {}
    for j, (a, b) in enumerate([(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]):
        thirds[f"third_{j+1}"] = _seg(rets[a:b], spies[a:b], dates[a:b])
    years = {}
    for y in sorted(set(dates.year)):
        m = dates.year == y
        if m.sum() >= 3:
            years[str(y)] = _seg(rets[m], spies[m], dates[m])
    # rolling 12-month vs-SPY: fraction of windows strategy beats SPY
    roll_beat, roll_n = 0, 0
    for i in range(0, n - 12 + 1):
        st = float(np.prod(1 + rets[i:i + 12]) - 1)
        sp = float(np.prod(1 + spies[i:i + 12]) - 1)
        roll_n += 1; roll_beat += int(st > sp)
    roll = {"windows": roll_n, "beat_spy": roll_beat,
            "beat_pct": round(roll_beat / roll_n * 100, 1) if roll_n else 0}

    both_halves_pos = halves["first_half"]["vs_spy"] > 0 and halves["second_half"]["vs_spy"] > 0
    years_beat = sum(1 for y in years.values() if y["vs_spy"] > 0)
    robust = both_halves_pos and years_beat >= 0.6 * len(years) and roll["beat_pct"] >= 60

    print("\n=== WALK-FORWARD / SUBPERIOD (validated top-20 engine) ===", flush=True)
    print(f"  FULL         vsSPY {full['vs_spy']:>7}%  t(excess)={full['t_stat']}  hit {full['hit_rate']}%  n={full['months']}", flush=True)
    for k, s in {**halves, **thirds}.items():
        print(f"  {k:12} vsSPY {s['vs_spy']:>7}%  t={s['t_stat']}  hit {s['hit_rate']}%  n={s['months']}", flush=True)
    for y, s in years.items():
        print(f"  {y:12} vsSPY {s['vs_spy']:>7}%  strat {s['strat']}%  spy {s['spy']}%  hit {s['hit_rate']}%", flush=True)
    print(f"  rolling-12mo beats SPY in {roll['beat_pct']}% of {roll['windows']} windows", flush=True)
    print(f"\n  ROBUST? {robust}  (both halves +, {years_beat}/{len(years)} years beat SPY, "
          f"{roll['beat_pct']}% rolling windows)", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "full": full, "halves": halves, "thirds": thirds, "years": years, "rolling_12mo": roll,
        "robust": bool(robust),
        "verdict": ("Edge is PERSISTENT across subperiods — beats SPY in both halves, most years, and a majority "
                    "of rolling windows." if robust else "Edge is CONCENTRATED — one or more subperiods carry it; "
                    "treat the headline number with caution and size accordingly."),
        "caveat": "Rules-based (no fitted params) so this is subperiod stability, the right OOS proxy. Still one "
                  "market regime (~5y); a true bear-regime test needs more history. Directional, no fees.",
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
            kind="walk_forward", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                           "computed_at": timezone.now()})
        print("Saved BacktestResult[walk_forward]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
