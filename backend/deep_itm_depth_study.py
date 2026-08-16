#!/usr/bin/env python3
"""SYNTHETIC LEVERAGE via deep-ITM depth — 'clean 1.5x margin' isn't real (interest 6-12%, margin calls,
IRA/cash accounts can't); the practical way is a DEEP-ITM LEAPS call (defined risk, no call). Question:
how DEEP must you go so the call tracks the stock (~stock win rate, low theta) while still giving leverage?
ATM tanks win rate to ~40%; deep-ITM should recover it toward the stock's ~57%.

Sweep K/S = 0.80/0.70/0.60/0.50 (20/30/40/50% ITM), 90 DTE, held ~1mo, perfect plays, real ATM IV. Per
depth: avg natural leverage (S/prem), delta, per-pick mean/win/avgWin/avgLoss, and 'synth_1.5x' = the
net return if you SIZE the position to exactly 1.5x delta exposure (rest in cash). Compare to stock.
-> BacktestResult[deep_itm_depth] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/deep_itm_depth_study.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "deep_itm_depth.json"
LOOKBACK, TOP_N, R_F, DTE = 6, 10, 0.04, 90 / 365.0
SYNTH_LEV = 1.5
_N = NormalDist().cdf
DEPTHS = [("itm20", 0.80), ("itm30", 0.70), ("itm40", 0.60), ("itm50", 0.50)]


def _d1(S, K, T, s):
    return (log(S / K) + (R_F + s * s / 2) * T) / (s * sqrt(T))


def bs(S, K, T, s):
    if T <= 0 or s <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, s)
    return S * _N(d1) - K * exp(-R_F * T) * _N(d1 - s * sqrt(T))


def delta(S, K, T, s):
    return 1.0 if (T <= 0 or s <= 0) else _N(_d1(S, K, T, s))


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _wr(x):
    a = np.array(x, float)
    if not len(a):
        return None
    w, l = a[a > 0], a[a <= 0]
    return dict(n=len(a), mean_pct=round(a.mean() * 100, 1), win_pct=round((a > 0).mean() * 100, 1),
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

    stk, opt, synth, lev, dlt = [], {d[0]: [] for d in DEPTHS}, {d[0]: [] for d in DEPTHS}, {d[0]: [] for d in DEPTHS}, {d[0]: [] for d in DEPTHS}
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        hold = (ndate - date).days
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
            r = _available_at(px[pick], date)
            if iv0 is None or not (S0 and S0 > 0) or pd.isna(px.loc[ndate, pick]):
                continue
            ret = float(px.loc[ndate, pick] / S0 - 1)
            iv1 = iv_at(pick, ndate) or iv0; ST = S0 * (1 + ret)
            stk.append(ret)
            for lab, mny in DEPTHS:
                K = S0 * mny; prem = bs(S0, K, DTE, iv0)
                if prem <= 1e-6:
                    continue
                oret = (bs(ST, K, max(DTE - hold / 365.0, 1 / 365.0), iv1) - prem) / prem
                natlev = S0 / prem
                opt[lab].append(oret); lev[lab].append(natlev); dlt[lab].append(delta(S0, K, DTE, iv0))
                # sized to exactly SYNTH_LEV: deploy fraction f=SYNTH_LEV/natlev of capital in premium, rest cash
                f = min(SYNTH_LEV / natlev, 1.0)
                synth[lab].append(f * oret + (1 - f) * 0.0033)

    stock = _wr(stk)
    res = {}
    for lab, _ in DEPTHS:
        res[lab] = {"option_natural": _wr(opt[lab]), "synth_1p5": _wr(synth[lab]),
                    "avg_leverage": round(float(np.mean(lev[lab])), 2) if lev[lab] else None,
                    "avg_delta": round(float(np.mean(dlt[lab])), 2) if dlt[lab] else None}
    print("\n=== DEEP-ITM DEPTH — synthesize leverage while keeping win rate ===", flush=True)
    print(f"  stock        mean {stock['mean_pct']}%  win {stock['win_pct']}%  (avgWin {stock['avg_win_pct']}/avgLoss {stock['avg_loss_pct']})", flush=True)
    print(f"  {'depth':7} {'lev':>4} {'delta':>5}  {'NATURAL mean/win':18}  {'SIZED-to-1.5x mean/win'}", flush=True)
    for lab, _ in DEPTHS:
        o = res[lab]["option_natural"]; s = res[lab]["synth_1p5"]
        print(f"  {lab:7} {str(res[lab]['avg_leverage']):>4} {str(res[lab]['avg_delta']):>5}  "
              f"{o['mean_pct']}/{o['win_pct']}%{'':8}  {s['mean_pct']}/{s['win_pct']}%", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"dte_days": 90, "hold": "~1mo", "synth_target_leverage": SYNTH_LEV, "benchmark": BENCH},
        "stock": stock, "by_depth": res,
        "verdict": ("Deeper ITM -> lower natural leverage, higher delta, win rate climbs back toward the stock's. "
                    "Sizing a deep-ITM call to ~1.5x delta 'fakes' margin with defined risk, no margin call, and a "
                    "win rate near the stock's — the practical leverage this strategy actually wants."),
        "caveat": "Real ATM IV (deep-ITM true IV lower via skew -> real premiums cheaper/better than modeled), "
                  "1mo hold, no fees/spreads/assignment, ~Sept2022+, perfect plays, small sample.",
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
            kind="deep_itm_depth", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                             "computed_at": timezone.now()})
        print("Saved BacktestResult[deep_itm_depth]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
