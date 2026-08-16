#!/usr/bin/env python3
"""'WEAKNESS TO MOMENTUM IS A THING' — test the TRANSITION. The value pick fades pure stock acceleration
(buy weakness) but pure weakness can be a falling knife. The sweet spot may be the INFLECTION: a cheap
stock that WAS weak and is NOW turning up (weakness -> momentum). Split value picks into 4 quadrants by
(weak? = stock 6mo momentum < 0) x (accelerating? = stock accel > 0), report forward return + win rate:

  weak + accel   WEAKNESS->MOMENTUM transition (inflecting up from a weak base)  <- the hypothesis
  weak + fade    falling knife (weak, still sliding)
  strong + accel already-running momentum (late)
  strong + fade  topping (strong, rolling over)
Plus a strategy that TILTS the pick toward weak+accelerating names vs the plain cheapest-P/B.
-> BacktestResult[stock_transition] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/stock_transition_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "stock_transition.json"
LOOKBACK, TOP_N = 6, 10


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


def _q(x):
    a = np.array(x, float)
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
    etf_accel = (etf_m.pct_change(3) - etf_m.pct_change(3).shift(3))     # sector signal (validated)
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
    s_mom6 = px.pct_change(6)
    s_accel = px.pct_change(3) - px.pct_change(3).shift(3)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = 9

    # ---- A. quadrant analysis on the value pick ----
    quad = {"weak_accel": [], "weak_fade": [], "strong_accel": [], "strong_fade": []}
    # ---- B. strategy: baseline cheapest-P/B vs transition-tilt (prefer weak+accel) ----
    base_r, base_s, tr_r, tr_s = [], [], [], []
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        b_slot, t_slot = [], []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            # baseline pick = cheapest P/B
            bpick = pb.loc[date, use].idxmin()
            br = _ret_delist(px[bpick], date, ndate)
            if br is not None and np.isfinite(br):
                b_slot.append(float(br))
                w = pd.notna(s_mom6.loc[date, bpick]) and s_mom6.loc[date, bpick] < 0
                a = pd.notna(s_accel.loc[date, bpick]) and s_accel.loc[date, bpick] > 0
                key = ("weak_" if w else "strong_") + ("accel" if a else "fade")
                quad[key].append(float(br))
            # transition-tilt = cheapest P/B among WEAK+ACCELERATING; fallback to plain
            trans = [c for c in use if (pd.notna(s_mom6.loc[date, c]) and s_mom6.loc[date, c] < 0)
                     and (pd.notna(s_accel.loc[date, c]) and s_accel.loc[date, c] > 0)]
            tpick = pb.loc[date, trans].idxmin() if trans else bpick
            tr = _ret_delist(px[tpick], date, ndate)
            if tr is not None and np.isfinite(tr):
                t_slot.append(float(tr))
        if b_slot:
            base_r.append(float(np.mean(b_slot))); base_s.append(float(sp))
        if t_slot:
            tr_r.append(float(np.mean(t_slot))); tr_s.append(float(sp))

    quads = {k: _q(v) for k, v in quad.items()}
    base = _stats(base_r, base_s)
    trans = _stats(tr_r, tr_s)
    print("\n=== A. QUADRANTS on the value pick (weak/strong x accel/fade) ===", flush=True)
    order = ["weak_accel", "weak_fade", "strong_accel", "strong_fade"]
    for k in order:
        q = quads[k]
        lbl = {"weak_accel": "WEAK->MOMENTUM (transition)", "weak_fade": "weak+fade (falling knife)",
               "strong_accel": "strong+accel (late)", "strong_fade": "strong+fade (topping)"}[k]
        print(f"  {lbl:30} mean {q['mean_pct']:>6}%  win {q['win_pct']}%  (n{q['n']})" if q else f"  {lbl}: –", flush=True)
    print("\n=== B. STRATEGY: baseline vs transition-tilt ===", flush=True)
    print(f"  baseline cheapest-P/B   vsSPY {base['vs_spy']:>7}%  Sh {base['sharpe']}  DD {base['max_drawdown']}%", flush=True)
    print(f"  transition-tilt         vsSPY {trans['vs_spy']:>7}%  Sh {trans['sharpe']}  DD {trans['max_drawdown']}%  "
          f"({'+' if trans['vs_spy']-base['vs_spy']>=0 else ''}{round(trans['vs_spy']-base['vs_spy'],1)}pp)", flush=True)

    wa = quads["weak_accel"]; best_quad = max(quads, key=lambda k: quads[k]["mean_pct"] if quads[k] else -9)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)),
                   "weak": "stock 6mo momentum < 0", "accel": "stock 3mo-now minus prior-3mo > 0"},
        "quadrants": quads, "best_quadrant": best_quad, "baseline": base, "transition_tilt": trans,
        "tilt_lift_pp": round(trans["vs_spy"] - base["vs_spy"], 1),
        "verdict": (f"Best quadrant = {best_quad}. Weakness->momentum (weak+accel) picks: {wa['mean_pct']}%/"
                    f"{wa['win_pct']}% win. Transition-tilt strategy {'+' if trans['vs_spy']>=base['vs_spy'] else ''}"
                    f"{round(trans['vs_spy']-base['vs_spy'],1)}pp vs baseline. "
                    + ("'Weakness to momentum' IS a thing — tilting the value pick toward weak-but-turning-up "
                       "names helps." if best_quad == "weak_accel" or trans["vs_spy"] > base["vs_spy"] + 5 else
                       "The transition edge is modest here; pure weakness (falling-knife-and-all) still carries "
                       "the value pick.")),
        "caveat": "In-sample, no fees, ~5y, 9mo warmup. weak=6mo mom<0 is one proxy (RSI<45 is another).",
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
            kind="stock_transition", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                               "computed_at": timezone.now()})
        print("Saved BacktestResult[stock_transition]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
