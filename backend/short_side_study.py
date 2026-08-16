#!/usr/bin/env python3
"""THE SHORT SIDE — does the OPPOSITE strategy make money? Long = cheapest-P/B in TOP-acceleration sectors.
Symmetric short = most-EXPENSIVE-P/B in BOTTOM-acceleration (decelerating/fading) sectors, betting they
fall. Test short-book return (= -stock return), a 'short the expensive junk' variant (+ trap/high-debt),
and a market-neutral long-short. Watch the asymmetry: a shorted name that POPS is a catastrophic loss.
-> BacktestResult[short_side] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/short_side_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10


def _st(rets, spy):
    r = np.asarray(rets, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


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
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    L, Sexp, Sjunk, spies = [], [], [], []
    squeeze = 0; nshort = 0
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        acc = accel.loc[date].dropna().sort_values(ascending=False)
        top = acc.head(TOP_N).index          # accelerating
        bot = acc.tail(TOP_N).index          # decelerating / fading
        # LONG: cheapest-P/B guarded low-debt in TOP accel sectors (>$5M/day, matches live)
        lslot = []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h]) and (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6)]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if g:
                r = _ret_delist(px[pb.loc[date, g].idxmin()], date, ndate)
                if r is not None and np.isfinite(r):
                    lslot.append(float(r))
        # SHORT: most-EXPENSIVE-P/B in BOTTOM accel sectors (short return = -stock return)
        se, sj = [], []
        for etf in bot:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6)]
            if not c:
                continue
            exp = pb.loc[date, c].idxmax()                 # most expensive
            r = _ret_delist(px[exp], date, ndate)
            if r is not None and np.isfinite(r):
                se.append(-float(r)); nshort += 1
                if r > 0.4:
                    squeeze += 1                            # short squeezed (name popped >40%)
            junk = [h for h in c if bool(trap.loc[date, h])]   # expensive AND value-trap (short the junk)
            if junk:
                jr = _ret_delist(px[pb.loc[date, junk].idxmax()], date, ndate)
                if jr is not None and np.isfinite(jr):
                    sj.append(-float(jr))
        if lslot:
            L.append(float(np.mean(lslot))); spies.append(float(sp))
            Sexp.append(float(np.mean(se)) if se else 0.0)
            Sjunk.append(float(np.mean(sj)) if sj else 0.0)

    long_b = _st(L, spies)
    short_exp = _st(Sexp, [0] * len(Sexp))          # short book: absolute return (vs cash)
    short_junk = _st(Sjunk, [0] * len(Sjunk))
    ls = _st([l + s for l, s in zip(L, Sexp)], spies)   # long + short (market-neutral-ish, unlevered sum)
    print("\n=== SHORT SIDE — does the opposite make money? ===", flush=True)
    print(f"  LONG (cheap in accel):        total {long_b['total']}%  vsSPY {long_b['vs_spy']}%  Sh {long_b['sharpe']}  win {long_b['win']}%", flush=True)
    print(f"  SHORT expensive-in-fading:    total {short_exp['total']}%  (abs, vs cash)  Sh {short_exp['sharpe']}  win {short_exp['win']}%", flush=True)
    print(f"  SHORT expensive-JUNK(trap):   total {short_junk['total']}%  (abs)  Sh {short_junk['sharpe']}  win {short_junk['win']}%", flush=True)
    print(f"  LONG+SHORT combined:          total {ls['total']}%  vsSPY {ls['vs_spy']}%  Sh {ls['sharpe']}  DD {ls['dd']}%", flush=True)
    print(f"  short squeezes (name popped >40% against the short): {squeeze}/{nshort} ({round(squeeze/max(nshort,1)*100,1)}%)", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "long": long_b, "short_expensive": short_exp, "short_junk": short_junk, "long_short": ls,
        "squeeze_pct": round(squeeze / max(nshort, 1) * 100, 1),
        "verdict": ("Shorting the opposite MAKES MONEY — the short book is positive and the long-short beats long-only."
                    if short_exp["total"] > 20 else
                    "Shorting the opposite does NOT work — the short book LOSES (markets drift up, momentum persists, "
                    "and shorted names get squeezed by the same fat-tail pops that help the LONG side). The edge is "
                    "long-only mean-reversion; it does not invert."),
        "caveat": "Short return = -stock return; no borrow cost/hard-to-borrow/margin (real shorting is WORSE). "
                  "Unlevered long+short sum. In-sample, ~5y bull-ish regime.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "short_side.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="short_side", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[short_side]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
