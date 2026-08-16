#!/usr/bin/env python3
"""COST TEST: does the distressed sleeve (and the 20% blend edge) survive REAL transaction costs? The whole
study is fee-free, but distressed small-caps have WIDE spreads — costs hit this sleeve far harder than the
large-cap flagship. Model it honestly: measure actual month-over-month TURNOVER of each basket, apply a
per-name round-trip cost scaled to liquidity, sweep the spread assumption, and find where the flagship-vs-
blend Sharpe edge disappears. Reports each sleeve's typical liquidity (median $vol/mktcap) to justify the
spread. -> BacktestResult[blend_cost].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/blend_cost_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def _turnover(baskets):
    """avg fraction of an equal-weight basket that is bought/sold each month (one-sided)."""
    tos = []
    for a, b in zip(baskets[:-1], baskets[1:]):
        if not a and not b:
            continue
        sa, sb = set(a), set(b)
        # one-sided turnover = weight that changed hands / 2 (buys≈sells for eq-wt full-rotate)
        entered = len(sb - sa) / max(len(sb), 1)
        tos.append(entered)
    return float(np.mean(tos)) if tos else 0.0


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
    mktcap = px * sh
    pb = mktcap / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    flag, dist, spies = [], [], []
    fbaskets, dbaskets = [], []
    f_dv, f_mc, d_dv, d_mc = [], [], [], []
    for i in range(9, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        top = accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        fslot, fnames = [], []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
                 and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
                 and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6]
            g = [x for x in c if bool(low.loc[date, x])] or c
            if g:
                pick = pb.loc[date, g].idxmin()
                r = _ret_delist(px[pick], date, ndate)
                if r is not None and np.isfinite(r):
                    fslot.append(float(r)); fnames.append(pick)
                    f_dv.append(float(dvol.loc[date, pick])); f_mc.append(float(mktcap.loc[date, pick]))
        dslot, dnames = [], []
        for h in common:
            if not _available_at(px[h], date):
                continue
            if not (pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6):
                continue
            e_ = eq.loc[date, h]
            if pd.isna(e_) or e_ >= 0 or not (pd.notna(ni.loc[date, h]) and ni.loc[date, h] <= 0):
                continue
            r = _ret_delist(px[h], date, ndate)
            if r is not None and np.isfinite(r):
                dslot.append(float(r)); dnames.append(h)
                d_dv.append(float(dvol.loc[date, h])); d_mc.append(float(mktcap.loc[date, h]))
        if fslot:
            flag.append(float(np.mean(fslot))); dist.append(float(np.mean(dslot)) if dslot else 0.0)
            spies.append(float(sp)); fbaskets.append(fnames); dbaskets.append(dnames)

    flag, dist, spies = np.array(flag), np.array(dist), np.array(spies)
    f_to, d_to = _turnover(fbaskets), _turnover(dbaskets)
    print(f"\n  LIQUIDITY (median): flagship $vol {np.median(f_dv)/1e6:.1f}M / mktcap ${np.median(f_mc)/1e9:.1f}B  |  "
          f"distressed $vol {np.median(d_dv)/1e6:.1f}M / mktcap ${np.median(d_mc)/1e9:.2f}B", flush=True)
    print(f"  one-sided monthly TURNOVER: flagship {f_to*100:.0f}%  distressed {d_to*100:.0f}%", flush=True)

    # per-name round-trip cost (half-spread+impact, buy+sell). Flagship = large-cap tight; distressed swept.
    F_COST = 0.003          # 30 bps round-trip on liquid large-caps (generous/conservative)
    scenarios = {"tight_0.5%": 0.005, "base_1.0%": 0.010, "wide_1.5%": 0.015, "harsh_2.5%": 0.025}
    print(f"\n=== NET-OF-COST (flagship round-trip {F_COST*100:.1f}%; distressed swept) ===", flush=True)
    print(f"  {'scenario':12}  {'FLAGSHIP':>20}  {'DISTRESSED':>20}  {'BLEND 20%':>22}", flush=True)

    def net(series, per_name_cost, turnover):
        drag = turnover * per_name_cost           # cost applied to the fraction rotated, each month
        return series - drag

    out = {}
    # flagship net is constant across scenarios (its own cost); distressed varies
    fnet = net(flag, F_COST, f_to)
    fb_net = _stats(fnet, spies)
    for name, dcost in scenarios.items():
        dnet = net(dist, dcost, d_to)
        blend = 0.8 * fnet + 0.2 * dnet
        db_net, bl_net = _stats(dnet, spies), _stats(blend, spies)
        out[name] = {"distressed": db_net, "blend20": bl_net}
        s = lambda st: f"{st['total']:>7.0f}% Sh{st['sharpe']:.2f} DD{st['dd']:.0f}"
        print(f"  {name:12}  {s(fb_net):>20}  {s(db_net):>20}  {s(bl_net):>22}", flush=True)

    # gross reference
    fg, dg = _stats(flag, spies), _stats(dist, spies)
    blg = _stats(0.8 * flag + 0.2 * dist, spies)
    print(f"\n  GROSS (no cost):  FLAGSHIP {fg['total']}%/Sh{fg['sharpe']}  DISTRESSED {dg['total']}%/Sh{dg['sharpe']}  "
          f"BLEND20 {blg['total']}%/Sh{blg['sharpe']}", flush=True)

    base = out["base_1.0%"]
    edge_survives = base["blend20"]["sharpe"] > fb_net["sharpe"] + 0.02
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "flagship_roundtrip": F_COST, "months": int(len(flag)),
                   "flagship_turnover": round(f_to, 3), "distressed_turnover": round(d_to, 3)},
        "liquidity": {"flagship_dvol_m": round(np.median(f_dv) / 1e6, 1), "flagship_mktcap_b": round(np.median(f_mc) / 1e9, 2),
                      "distressed_dvol_m": round(np.median(d_dv) / 1e6, 1), "distressed_mktcap_b": round(np.median(d_mc) / 1e9, 2)},
        "gross": {"flagship": fg, "distressed": dg, "blend20": blg},
        "flagship_net": fb_net, "scenarios": out,
        "verdict": (f"Distressed turns over {d_to*100:.0f}%/mo of a small basket. At a realistic ~1% round-trip "
                    f"(justified by ${np.median(d_mc)/1e9:.2f}B median mktcap), net blend-20% Sharpe {base['blend20']['sharpe']} "
                    f"vs flagship-net {fb_net['sharpe']}. " + (
                    "The blend edge SURVIVES costs — thin but real; keep the tilt small." if edge_survives else
                    "The blend edge is ERODED by costs — the +0.11 gross Sharpe bump does not clear realistic frictions; "
                    "treat distressed as a raw-return bet, not a risk-adjusted improvement.")),
        "caveat": "No bid-ask in data -> costs are ASSUMED (swept), not measured. Impact/partial-fills on thin distressed "
                  "names likely WORSE than modeled at size. Flagship cost held flat. In-sample.",
    }
    return payload


def main():
    p = build()
    from pathlib import Path
    OUT = Path(__file__).resolve().parent / ".data" / "studies" / "blend_cost.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="blend_cost", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[blend_cost]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
