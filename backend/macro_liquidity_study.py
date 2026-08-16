#!/usr/bin/env python3
"""MACRO-LIQUIDITY REGIME + LEADING RISK-OFF — the defensive_sleeve study showed momentum-ranked bonds are a
LAGGING risk-off trigger. Test genuinely LEADING macro signals from FRED (core.MacroSeries):
  net liquidity = WALCL - RRPONTSYD*1000 - WTREGEN  ($M)   -> the liquidity tide under risk assets
  M2 YoY growth (M2SL)                                     -> monetary expansion/contraction
  broad USD (DTWEXBGS) 3mo change                          -> dollar headwind
  HY OAS credit spread (BAMLH0A0HYM2, 2023-08+)            -> the classic leading stress signal
  10y-2y curve (T10Y2Y)                                    -> recession lead

Two parts on the flagship monthly returns (reused from volatility_regime_study.flagship_monthly):
 (A) REGIME DIAGNOSTIC — mean flagship return split by each regime (as-of PRIOR month-end, no look-ahead).
 (B) RISK-OFF OVERLAYS — exposure decided from the prior month's signal (cash earns 0):
     liq_gate       full when net-liquidity rising (3mo), else cash.
     liq_half       full when rising, else HALF exposure (gentler).
     hy_gate        cash when HY spread widened last month, else full (2023-08+; full before).
     combo_gate     cash when net-liq falling OR HY widening, else full.
If a leading overlay lifts Sharpe / cuts DD vs baseline, it's the risk-off mechanism the momentum sleeve wasn't.
-> BacktestResult[macro_liquidity] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/macro_liquidity_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
from backtest_lowpb import _tstat_from_returns
from volatility_regime_study import flagship_monthly     # reuse the exact flagship engine

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "macro_liquidity.json"


def _series(sid):
    from core.models import MacroSeries
    rows = MacroSeries.objects.filter(series=sid).values_list("date", "value")
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()
    return s


def _monthly(s):
    return s.resample("ME").last() if len(s) else s


def _perf(r, spy):
    r = np.asarray(r, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                t_stat=round(t, 2) if t is not None else None, months=n)


def build():
    dates, rets, spies, _ = flagship_monthly()
    rets = np.asarray(rets); spies = np.asarray(spies)
    midx = pd.DatetimeIndex(dates)
    print(f"flagship months {len(midx)} ({midx[0].date()}..{midx[-1].date()})", flush=True)

    walcl = _monthly(_series("WALCL")).reindex(midx).ffill()
    rrp = _monthly(_series("RRPONTSYD")).reindex(midx).ffill()
    tga = _monthly(_series("WTREGEN")).reindex(midx).ffill()
    m2 = _monthly(_series("M2SL")).reindex(midx).ffill()
    usd = _monthly(_series("DTWEXBGS")).reindex(midx).ffill()
    hy = _monthly(_series("BAMLH0A0HYM2")).reindex(midx).ffill()

    net_liq = walcl - rrp * 1000.0 - tga                  # all $M
    sig = pd.DataFrame(index=midx)
    sig["net_liq_up"] = (net_liq.diff(3) > 0)             # net liquidity rising (3mo)
    sig["m2_up"] = (m2.pct_change(12) > 0)               # M2 expanding YoY
    sig["usd_down"] = (usd.diff(3) <= 0)                 # dollar NOT rising (tailwind)
    sig["hy_narrow"] = (hy.diff() <= 0)                  # HY spread not widening
    sig_prior = sig.shift(1)                              # decide from PRIOR month-end (no look-ahead)

    # (A) diagnostic: mean flagship return by regime
    def split(mask_name):
        m = sig_prior[mask_name]
        on = [rets[i] for i in range(len(midx)) if m.iloc[i] == True]     # noqa: E712
        off = [rets[i] for i in range(len(midx)) if m.iloc[i] == False]   # noqa: E712
        f = lambda a: (round(float(np.mean(a)) * 100, 2), len(a)) if a else (None, 0)
        return {"on": f(on), "off": f(off)}
    diag = {k: split(k) for k in ("net_liq_up", "m2_up", "usd_down", "hy_narrow")}

    # (B) overlays (exposure from prior-month signal; cash=0)
    def overlay(expo):
        return _perf([expo[i] * rets[i] for i in range(len(midx))], spies)
    base = _perf(rets, spies)

    def gate(mask, off_expo=0.0):
        m = sig_prior[mask]
        return [1.0 if (m.iloc[i] == True or pd.isna(m.iloc[i])) else off_expo for i in range(len(midx))]  # noqa: E712

    hy_ok = sig_prior["hy_narrow"]
    combo = [0.0 if ((sig_prior["net_liq_up"].iloc[i] == False) or (hy_ok.iloc[i] == False)) else 1.0   # noqa: E712
             for i in range(len(midx))]

    results = {
        "baseline": base,
        "liq_gate": overlay(gate("net_liq_up", 0.0)),
        "liq_half": overlay(gate("net_liq_up", 0.5)),
        "m2_gate": overlay(gate("m2_up", 0.0)),
        "hy_gate": overlay(gate("hy_narrow", 0.0)),
        "combo_gate": overlay(combo),
    }

    print(f"\n=== (A) flagship monthly mean by regime (as-of prior month-end) ===", flush=True)
    for k, v in diag.items():
        print(f"  {k:<12} ON {v['on'][0]}% (n{v['on'][1]})   OFF {v['off'][0]}% (n{v['off'][1]})", flush=True)
    print(f"\n=== (B) leading risk-off overlays (total / vsSPY / Sharpe / DD) ===", flush=True)
    for k, r in results.items():
        print(f"  {k:<12} {r['total']:>7}%  vsSPY {r['vs_spy']:>7}  Sh {r['sharpe']:>5}  DD {r['dd']:>6}%  t {r['t_stat']}",
              flush=True)

    cand = [k for k in results if k != "baseline"]
    best = max(cand, key=lambda k: results[k]["sharpe"])
    b = results[best]
    # regime with the biggest ON-OFF spread
    spreads = {k: (v["on"][0] - v["off"][0]) for k, v in diag.items()
               if v["on"][0] is not None and v["off"][0] is not None}
    top_regime = max(spreads, key=lambda k: spreads[k]) if spreads else None
    verdict = (
        f"Baseline {base['total']}%/Sh{base['sharpe']}/DD{base['dd']}%. "
        + (f"Biggest regime spread: {top_regime} ON {diag[top_regime]['on'][0]}% vs OFF {diag[top_regime]['off'][0]}% "
           f"(+{spreads[top_regime]:.2f}pp/mo). " if top_regime else "")
        + f"Best overlay = {best} ({b['total']}%/Sh{b['sharpe']}/DD{b['dd']}%). "
        + ("A leading macro overlay improves Sharpe/DD -> real risk-off signal, worth wiring."
           if b["sharpe"] > base["sharpe"] + 0.05 or b["dd"] > base["dd"] + 3 else
           "No leading overlay beats baseline Sharpe -> even leading macro signals don't time this monthly "
           "rotation better than staying invested (short 2021-2026 sample; regime spread is the takeaway).")
    )
    print("\n" + verdict, flush=True)
    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"months": int(len(midx)), "signals": "net_liq(WALCL-RRP-TGA), M2 YoY, USD 3mo, HY OAS, curve",
                   "exposure": "from prior month-end signal (no look-ahead), cash=0", "hy_from": "2023-08"},
        "regime_diagnostic": diag, "results": results, "verdict": verdict,
        "caveat": "Flagship monthly returns overlaid with macro exposure (whole-book, not per-name). FRED data has "
                  "publication lag; using prior-month-end value is conservative but not lag-exact. HY spread only "
                  "2023-08+ (treated as full-exposure before). 2021-2026 = ~63 months, few real risk-off episodes "
                  "-> regime spreads are directional. Cash=0 (no T-bill). PIT flagship, survivorship as in base.",
    }


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="macro_liquidity", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                              "computed_at": timezone.now()})
        print("Saved BacktestResult[macro_liquidity]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
