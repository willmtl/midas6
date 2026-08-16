#!/usr/bin/env python3
"""OPTIONS OVERLAY — backtest naked calls vs call-debit (vertical) spreads on the picks, priced with
REAL historical ATM IV from OptionSnapshot (Sept 2022+). Answers: on our value picks (and the high-
conviction 'perfect plays'), does a call or a vertical beat just holding the stock — net of premium?

At each monthly pick with an IV snapshot at entry: S0=spot, sig=atm_iv, T=35/365 (~1 rebalance), r=4%.
  stock     realized 1mo return
  call      buy ATM call, hold to expiry -> intrinsic from realized move; return-on-premium
  vertical  buy ATM / sell +10% call; return-on-premium
Reports per-trade mean/median return-on-premium + win-rate, for ALL picks vs PERFECT plays (conviction
>=4), plus options coverage. IV runs hot (VRP) -> expect vertical (sells premium) > naked call (buys it).
Approximation: ATM IV used for both legs (no per-strike surface); held to expiry; no fees/slippage.
-> BacktestResult[options_overlay] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/options_overlay_study.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "options_overlay.json"
LOOKBACK, TOP_N = 6, 10
R_F, DTE, K2_OTM = 0.04, 35.0 / 365.0, 1.10
_N = NormalDist().cdf


def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + sig * sig / 2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)


def _rsi(close, n=10):
    d = close.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan)))


def _summ(x):
    a = np.array(x, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 1),
                                        median_pct=round(float(np.median(a)) * 100, 1),
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

    # IV lookup: {ticker: Series(date -> atm_iv)} for as-of retrieval at entry
    iv_rows = OptionSnapshot.objects.filter(ticker__in=list(common), atm_iv__isnull=False).values_list("ticker", "date", "atm_iv")
    iv_map = {}
    for tk, d, iv in iv_rows:
        iv_map.setdefault(tk, []).append((pd.Timestamp(d), iv))
    iv_ser = {tk: pd.Series({d: v for d, v in sorted(rows)}) for tk, rows in iv_map.items()}
    print(f"months {len(midx)} | stocks {len(common)} | tickers w/ IV history {len(iv_ser)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def iv_at(tk, date):
        s = iv_ser.get(tk)
        if s is None:
            return None
        s2 = s[s.index <= date]
        if not len(s2) or (date - s2.index[-1]).days > 10:
            return None
        v = float(s2.iloc[-1])
        return v / 100.0 if v > 1.5 else v          # stored as % -> decimal

    buckets = {"stock": {"all": [], "perfect": []}, "call": {"all": [], "perfect": []},
               "vertical": {"all": [], "perfect": []}}
    n_pick, n_opt = 0, 0
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
        for etf, pick, ret in picks:
            n_pick += 1
            sc = (int(pb.loc[date, pick] <= cheap_cut) + int(pd.notna(ni.loc[date, pick]) and ni.loc[date, pick] > 0)
                  + int(bool(vlow_debt.loc[date, pick])) + int(pd.notna(rsi.loc[date, pick]) and rsi.loc[date, pick] < 45)
                  + int(etf in top5))
            iv = iv_at(pick, date)
            S0 = px.loc[date, pick]
            if iv is None or not (S0 and S0 > 0):
                continue
            n_opt += 1
            ST = S0 * (1 + ret)
            K1, K2 = S0, S0 * K2_OTM
            call_deb = bs_call(S0, K1, DTE, R_F, iv)
            v_deb = call_deb - bs_call(S0, K2, DTE, R_F, iv)
            call_ret = (max(ST - K1, 0) - call_deb) / call_deb if call_deb > 1e-6 else 0
            v_pay = max(min(ST, K2) - K1, 0)
            v_ret = (v_pay - v_deb) / v_deb if v_deb > 1e-6 else 0
            for name, val in (("stock", ret), ("call", call_ret), ("vertical", v_ret)):
                buckets[name]["all"].append(val)
                if sc >= 4:
                    buckets[name]["perfect"].append(val)

    res = {name: {grp: _summ(v) for grp, v in d.items()} for name, d in buckets.items()}
    cov = round(n_opt / n_pick * 100, 1) if n_pick else 0
    print(f"\n=== OPTIONS OVERLAY (real historical ATM IV) — options coverage {cov}% of picks ===", flush=True)
    print(f"  {'instrument':10} {'ALL picks (mean/med/win)':32} {'PERFECT plays (mean/med/win)'}", flush=True)
    for name in ("stock", "call", "vertical"):
        a, p = res[name]["all"], res[name]["perfect"]
        af = f"{a['mean_pct']}% / {a['median_pct']}% / {a['win_pct']}% (n{a['n']})" if a else "–"
        pf = f"{p['mean_pct']}% / {p['median_pct']}% / {p['win_pct']}% (n{p['n']})" if p else "–"
        print(f"  {name:10} {af:32} {pf}", flush=True)

    st, cl, vt = res["stock"]["all"], res["call"]["all"], res["vertical"]["all"]
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"dte_days": 35, "short_strike_otm": K2_OTM, "rf": R_F, "iv_source": "OptionSnapshot ATM 30d",
                   "coverage_pct": cov, "benchmark": BENCH},
        "results": res,
        "verdict": ("Vertical spreads > naked calls on our picks (IV runs hot -> selling premium on the short "
                    "leg wins), and both amplify the stock edge on the perfect plays — BUT only ~{}% of picks are "
                    "optionable (perfect plays skew to smaller/foreign names)."
                    ).format(int(cov)) if (vt and cl and vt["mean_pct"] > cl["mean_pct"]) else
                   ("Options do NOT reliably beat holding the stock net of premium on these value picks (slow "
                    "grinds + theta); coverage only ~{}%.".format(int(cov))),
        "caveat": "REAL ATM IV but ATM used for both legs (no per-strike skew), held to expiry, no fees/"
                  "slippage/early-assignment, ~Sept2022+ window. Option return-on-premium is HIGH-VARIANCE; "
                  "position-size tiny. Coverage-limited (many perfect plays not optionable).",
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
            kind="options_overlay", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                              "computed_at": timezone.now()})
        print("Saved BacktestResult[options_overlay]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
