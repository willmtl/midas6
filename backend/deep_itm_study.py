#!/usr/bin/env python3
"""MULTIPLY PROFIT WITH OPTIONS — the RIGHT structures for a slow/modest value grind. ATM short-dated
calls die to theta (already shown). Deep-ITM and long-dated calls act as leveraged STOCK: high delta
(~move 1:1), mostly intrinsic (low theta), cost a fraction of the stock (leverage). Held only to the
monthly rebalance (not expiry), so little theta is paid.

Structures on the picks (priced via Black-Scholes on real ATM IV, exit re-priced at rebalance):
  stock       hold the shares (1x)
  itm20_90    90-DTE call, 20% ITM (K=0.80*S) — stock replacement, ~0.85 delta
  itm10_90    90-DTE call, 10% ITM
  atm_90      90-DTE ATM call (long-dated, more leverage, more theta)
  itm20_180   180-DTE 20% ITM (LEAPS-like)
Reports return-on-premium mean/median/win + effective LEVERAGE (S/premium) + profit-multiple vs stock,
for ALL picks and PERFECT plays. -> BacktestResult[deep_itm] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/deep_itm_study.py
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
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis
from core.models import OptionSnapshot

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "deep_itm.json"
LOOKBACK, TOP_N, R_F = 6, 10, 0.04
_N = NormalDist().cdf
# (label, moneyness K/S, DTE days)
STRUCTS = [("itm20_90", 0.80, 90), ("itm10_90", 0.90, 90), ("atm_90", 1.00, 90), ("itm20_180", 0.80, 180)]


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R_F + sig * sig / 2) * T) / (sig * sqrt(T)); d2 = d1 - sig * sqrt(T)
    return S * _N(d1) - K * exp(-R_F * T) * _N(d2)


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _summ(x):
    a = np.array(x, float)
    if not len(a):
        return None
    w, l = a[a > 0], a[a <= 0]
    return dict(n=len(a), mean_pct=round(a.mean() * 100, 1), median_pct=round(float(np.median(a)) * 100, 1),
                win_pct=round((a > 0).mean() * 100, 1),
                avg_win_pct=round(w.mean() * 100, 1) if len(w) else None,
                avg_loss_pct=round(l.mean() * 100, 1) if len(l) else None,
                worst_pct=round(float(a.min()) * 100, 1))


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

    iv_rows = OptionSnapshot.objects.filter(ticker__in=list(common), atm_iv__isnull=False).values_list("ticker", "date", "atm_iv")
    ivm = {}
    for tk, d, iv in iv_rows:
        ivm.setdefault(tk, {})[pd.Timestamp(d)] = (iv / 100.0 if iv > 1.5 else iv)
    iv_ser = {tk: pd.Series(v).sort_index() for tk, v in ivm.items()}

    def iv_at(tk, date):
        s = iv_ser.get(tk)
        if s is None:
            return None
        s2 = s[s.index <= date]
        if not len(s2) or (date - s2.index[-1]).days > 10:
            return None
        return float(s2.iloc[-1])
    print(f"months {len(midx)} | stocks {len(common)} | IV tickers {len(iv_ser)}", flush=True)
    warmup = max(LOOKBACK, 1)

    B = {"stock": {"all": [], "perfect": []}}
    lev = {}
    for lab, _, _ in STRUCTS:
        B[lab] = {"all": [], "perfect": []}; lev[lab] = []
    n_pick = n_opt = 0
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        if not np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1):
            continue
        hold_days = (ndate - date).days
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
            pick = pb.loc[date, use].idxmin()
            r = _ret_delist(px[pick], date, ndate)
            if r is not None and np.isfinite(r):
                picks.append((etf, pick, float(r)))
        if not picks:
            continue
        pbs = sorted(pb.loc[date, [p for _, p, _ in picks]].values)
        cheap_cut = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0
        for etf, pick, ret in picks:
            n_pick += 1
            sc = (int(pb.loc[date, pick] <= cheap_cut) + int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
                  + int(bool(vlow_debt.loc[date, pick])) + int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)
                  + int(etf in top5))
            iv0 = iv_at(pick, date); iv1 = iv_at(pick, ndate)
            S0 = px.loc[date, pick]
            if iv0 is None or not (S0 and S0 > 0):
                continue
            iv1 = iv1 if iv1 is not None else iv0
            n_opt += 1
            ST = S0 * (1 + ret)
            grp = "perfect" if sc >= 4 else None
            B["stock"]["all"].append(ret)
            if grp:
                B["stock"]["perfect"].append(ret)
            for lab, mny, dte in STRUCTS:
                K = S0 * mny; T0 = dte / 365.0; T1 = max(dte - hold_days, 1) / 365.0
                prem = bs_call(S0, K, T0, iv0)
                if prem <= 1e-6:
                    continue
                exit_val = bs_call(ST, K, T1, iv1)
                oret = (exit_val - prem) / prem
                B[lab]["all"].append(oret)
                lev[lab].append(S0 / prem)
                if grp:
                    B[lab]["perfect"].append(oret)

    res = {k: {g: _summ(v) for g, v in d.items()} for k, d in B.items()}
    avg_lev = {lab: round(float(np.mean(v)), 1) if v else None for lab, v in lev.items()}
    cov = round(n_opt / n_pick * 100, 1) if n_pick else 0
    print(f"\n=== DEEP-ITM / LEAPS (leveraged stock replacement) — coverage {cov}% ===", flush=True)
    print(f"  {'struct':11} {'lev':>4}  {'PERFECT mean/med/win':26}  {'avgWin/avgLoss/worst'}", flush=True)
    sp = res["stock"]["perfect"]
    print(f"  {'stock':11} {'1.0':>4}  {sp['mean_pct']}/{sp['median_pct']}/{sp['win_pct']}%{'':6}  "
          f"{sp['avg_win_pct']}/{sp['avg_loss_pct']}/{sp['worst_pct']}%", flush=True)
    for lab, _, _ in STRUCTS:
        p = res[lab]["perfect"]
        if not p:
            continue
        print(f"  {lab:11} {str(avg_lev[lab]):>4}  {p['mean_pct']}/{p['median_pct']}/{p['win_pct']}%{'':6}  "
              f"{p['avg_win_pct']}/{p['avg_loss_pct']}/{p['worst_pct']}%", flush=True)

    # pick the structure with best perfect-play mean that also keeps median > -50 (not a lottery)
    cand = [(lab, res[lab]["perfect"]) for lab, _, _ in STRUCTS if res[lab]["perfect"]]
    best = max(cand, key=lambda kv: kv[1]["mean_pct"])[0] if cand else None
    bp = res[best]["perfect"] if best else None
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"structs": [{"label": l, "moneyness": m, "dte": d} for l, m, d in STRUCTS],
                   "avg_leverage": avg_lev, "coverage_pct": cov, "hold": "to monthly rebalance"},
        "results": res, "best_mean_struct": best,
        "verdict": (f"Best profit-multiplier on perfect plays = {best}: mean {bp['mean_pct']}% (stock {sp['mean_pct']}%), "
                    f"median {bp['median_pct']}%, win {bp['win_pct']}%, ~{avg_lev.get(best)}x leverage. Deep-ITM/long-"
                    "dated multiplies the modest stock move with far less theta than the ATM short-dated version — "
                    "BUT the leverage cuts BOTH ways (losses multiply too); size accordingly." if best else "n/a"),
        "caveat": "Real ATM IV used for all strikes (ignores skew — deep-ITM true IV is lower, so real premiums a bit "
                  "cheaper/better than modeled); held to rebalance; no fees/slippage/spreads/assignment; ~Sept2022+; "
                  "coverage-limited. Leverage multiplies drawdown too.",
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
            kind="deep_itm", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                       "computed_at": timezone.now()})
        print("Saved BacktestResult[deep_itm]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
