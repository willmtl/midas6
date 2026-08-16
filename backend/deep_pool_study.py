#!/usr/bin/env python3
"""MORE STOCKS PER ETF — does picking cheapest-P/B from the ETF's FULL holdings (top-~50) beat picking
from our hardcoded top-20? Keeps the ETF grouping + curation (liquid, thematic) — only the candidate
POOL depth changes. Same engine both ways: rotation top-10 by 6mo momentum -> cheapest positive-P/B
passing guard + low-debt, monthly, PIT. top20 must ~reproduce the honest split-corrected baseline (~+105% vs SPY; was +229% on the buggy adjusted-close P/B) (self-check).

  top20  candidate pool = hardcoded sector_holdings.HOLDINGS[name]['holdings'] (original 20)
  full   candidate pool = get_holdings(name) (top-20 + EODHD-expanded constituents, now PIT-backfilled)

CAVEAT: 'full' uses CURRENT expanded constituents backfilled over history -> mild survivorship inflation
(but within the ETF's curated liquid names, far less than classification). Directional.
-> BacktestResult[deep_pool] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/deep_pool_study.py
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

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "deep_pool.json"
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


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sec_top20, sec_full, all_holds = {}, {}, set()
    for n, e in etfs.items():
        base = [t for t in sector_holdings.HOLDINGS.get(n, {}).get("holdings", []) if t not in (e, BENCH) and t not in CRYPTO]
        full = [t for t in sector_holdings.get_holdings(n, expanded=True) if t not in (e, BENCH) and t not in CRYPTO]
        sec_top20[e] = (n, base); sec_full[e] = (n, full)
        all_holds.update(full)
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
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)
    warmup = max(LOOKBACK, 1)

    def run(sector_map):
        rets, spies, pool = [], [], []
        for i in range(warmup, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(sp):
                continue
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
            slot, psz = [], []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                         and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
                ld = [c for c in cands if bool(low_debt.loc[date, c])]
                use = ld or cands
                if not use:
                    continue
                psz.append(len(use))
                pick = pb.loc[date, use].idxmin()
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    slot.append(float(r))
            if slot:
                rets.append(float(np.mean(slot))); spies.append(float(sp)); pool.append(float(np.mean(psz)))
        s = _stats(rets, spies)
        s["avg_pool"] = round(float(np.mean(pool)), 1) if pool else 0
        return s

    top20 = run(sec_top20)
    full = run(sec_full)
    print("\n=== MORE STOCKS PER ETF (top-20 pool vs full expanded pool) ===", flush=True)
    for tag, s in (("top20", top20), ("full ", full)):
        print(f"  {tag}  vsSPY {s['vs_spy']:>7}%  total {s['total_return']}%  t={s['t_stat']}  "
              f"Sh {s['sharpe']}  DD {s['max_drawdown']}%  avg pool/sector {s['avg_pool']}", flush=True)
    delta = round(full["vs_spy"] - top20["vs_spy"], 1)
    print(f"\n  [SELFCHECK] top20 should ~= +105% vs SPY (honest split-corrected P/B): got +{top20['vs_spy']}%", flush=True)
    print(f"  deeper pool effect: {'+' if delta >= 0 else ''}{delta}pp vs SPY", flush=True)

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx))},
        "top20": top20, "full": full, "delta_vs_spy": delta,
        "verdict": (f"Deepening the candidate pool per ETF from ~{top20['avg_pool']} to ~{full['avg_pool']} names "
                    f"changed vs-SPY by {'+' if delta >= 0 else ''}{delta}pp ({top20['vs_spy']}% -> {full['vs_spy']}%). "
                    + ("A deeper pool finds cheaper value names that help." if delta > 15 else
                       "Deeper pool does NOT materially help — our top-20 already contains the cheapest name in "
                       "most sectors (recon showed it binds only in a few growth sectors).")),
        "caveat": "'full' uses current expanded constituents over history -> mild survivorship inflation "
                  "(within ETF-curated liquid names). In-sample, no fees, ~5y single regime.",
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
            kind="deep_pool", defaults={"payload": json.loads(json.dumps(p, default=str)),
                                        "computed_at": timezone.now()})
        print("Saved BacktestResult[deep_pool]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
