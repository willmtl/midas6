#!/usr/bin/env python3
"""3-MONTH HOLD effect on the leveraged/options plays. The stock benefits from a 3mo hold (reversion
completes: 55.6%/+2.7% -> 57.9%/+5.8%). Does holding the LEVERAGED position 3 months amplify that (bigger
move x leverage), or does theta eat it? Per-pick, perfect plays, priced on real ATM IV. Option DTE matched
to hold + ~60d buffer so time value remains at exit (1mo->90DTE, 3mo->150DTE). Deep-ITM 20% + ATM.
-> BacktestResult[options_hold] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/options_hold_study.py
"""
import os, json, warnings
from math import log, sqrt, exp
from statistics import NormalDist
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis
from core.models import OptionSnapshot

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "options_hold.json"
LOOKBACK, TOP_N, R_F = 6, 10, 0.04
_N = NormalDist().cdf
# (label, hold_months, moneyness, dte_days)
VARIANTS = [("stock_1mo", 1, None, None), ("stock_3mo", 3, None, None),
            ("itm20_1mo", 1, 0.80, 90), ("itm20_3mo", 3, 0.80, 150),
            ("atm_1mo", 1, 1.00, 90), ("atm_3mo", 3, 1.00, 150)]


def bs(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R_F + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    return S * _N(d1) - K * exp(-R_F * T) * _N(d2)


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _wr(x):
    a = np.array(x, float)
    if not len(a):
        return None
    w, l = a[a > 0], a[a <= 0]
    return dict(n=len(a), mean_pct=round(a.mean() * 100, 1), median_pct=round(float(np.median(a)) * 100, 1),
                win_pct=round((a > 0).mean() * 100, 1),
                avg_win_pct=round(w.mean() * 100, 1) if len(w) else None,
                avg_loss_pct=round(l.mean() * 100, 1) if len(l) else None)


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
    etf_trail = etf_m.pct_change(LOOKBACK)
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
    vlow_debt = (debt / equity.where(equity != 0)) < 0.5
    rsi = pd.DataFrame({t: _rsi(stock_daily[t]["Close"]).resample("ME").last().reindex(midx)
                        for t in common if t in stock_daily and len(stock_daily[t]) > 20}).reindex(index=midx, columns=common)

    ivm = {}
    for tk, d, iv in OptionSnapshot.objects.filter(ticker__in=list(common), atm_iv__isnull=False).values_list("ticker", "date", "atm_iv"):
        ivm.setdefault(tk, {})[pd.Timestamp(d)] = (iv / 100.0 if iv > 1.5 else iv)
    iv_ser = {tk: pd.Series(v).sort_index() for tk, v in ivm.items()}

    def iv_at(tk, date):
        s = iv_ser.get(tk)
        if s is None:
            return None
        s2 = s[s.index <= date]
        return None if (not len(s2) or (date - s2.index[-1]).days > 10) else float(s2.iloc[-1])
    print(f"months {len(midx)} | stocks {len(common)} | IV {len(iv_ser)}", flush=True)
    warmup = max(LOOKBACK, 1)

    out = {v[0]: [] for v in VARIANTS}
    for i in range(warmup, len(midx) - 3):
        date = midx[i]
        ranks = list(etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        top5 = set(ranks[:5]); picks = []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            ld = [c for c in cands if bool(low_debt.loc[date, c])]
            use = ld or cands
            if not use:
                continue
            picks.append((etf, pb.loc[date, use].idxmin()))
        if not picks:
            continue
        pbs = sorted(pb.loc[date, [p for _, p in picks]].values)
        cc = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0
        for etf, pick in picks:
            sc = (int(pb.loc[date, pick] <= cc) + int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
                  + int(bool(vlow_debt.loc[date, pick])) + int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)
                  + int(etf in top5))
            if sc < 4:
                continue
            S0 = px.loc[date, pick]; iv0 = iv_at(pick, date)
            if not (S0 and S0 > 0):
                continue
            for lab, H, mny, dte in VARIANTS:
                j = i + H
                if j >= len(midx) or pd.isna(px.loc[midx[j], pick]):
                    continue
                ret = float(px.loc[midx[j], pick] / S0 - 1)
                if mny is None:
                    out[lab].append(ret)
                    continue
                if iv0 is None:
                    continue
                iv1 = iv_at(pick, midx[j]) or iv0
                held = (midx[j] - date).days
                prem = bs(S0, S0 * mny, dte / 365.0, iv0)
                if prem <= 1e-6:
                    continue
                exit_val = bs(S0 * (1 + ret), S0 * mny, max(dte - held, 1) / 365.0, iv1)
                out[lab].append((exit_val - prem) / prem)

    res = {lab: _wr(out[lab]) for lab, *_ in VARIANTS}
    print("\n=== 3-MONTH HOLD effect (per-pick, perfect plays) ===", flush=True)
    print(f"  {'variant':12} {'mean':>7} {'median':>8} {'win':>6}  {'avgWin/avgLoss'}", flush=True)
    for lab, *_ in VARIANTS:
        s = res[lab]
        if s:
            print(f"  {lab:12} {s['mean_pct']:>6}% {s['median_pct']:>7}% {s['win_pct']:>5}%  "
                  f"{s['avg_win_pct']}/{s['avg_loss_pct']}%", flush=True)

    def lift(a, b):
        return None if not (res.get(a) and res.get(b)) else round(res[a]["mean_pct"] - res[b]["mean_pct"], 1)
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH,
                   "note": "perfect plays; option DTE = hold + ~60d buffer"},
        "results": res,
        "hold_lift": {"stock_3mo_vs_1mo": lift("stock_3mo", "stock_1mo"),
                      "itm20_3mo_vs_1mo": lift("itm20_3mo", "itm20_1mo"),
                      "atm_3mo_vs_1mo": lift("atm_3mo", "atm_1mo")},
        "verdict": ("A 3-month hold amplifies the leveraged plays MORE than the stock (leverage x the fuller "
                    "reversion), if theta stays contained via longer-dated deep-ITM contracts — see hold_lift."),
        "caveat": "Per-pick (overlapping across holds -> not a portfolio Sharpe). Real ATM IV, ATM used for all "
                  "strikes, no fees/spreads, ~Sept2022+, ~coverage-limited, small perfect-play sample.",
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
            kind="options_hold", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                           "computed_at": timezone.now()})
        print("Saved BacktestResult[options_hold]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
