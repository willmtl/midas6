#!/usr/bin/env python3
"""REGIME CONDITIONING on the HONEST (split-corrected) flagship — the one orthogonal lever with prior evidence
of lift (macro regime -> sector leadership, where price-momentum has ~0). PIT & WALK-FORWARD-HONEST: the earlier
regime-leadership was IN-SAMPLE (which sectors led historically = look-ahead). Here 'regime-favored' sectors are
recomputed at each month from PAST months only (expanding window) in the SAME regime state, then applied forward.
Regime axes (PIT, from candles): RATES rising = TLT 3mo ret<0; INFLATION rising = (TIP/TLT) 3mo change>0;
MARKET risk-on = SPY>200d MA. Variants layer regime onto accel top-10 (corrected pb, guard, low-debt, $5M, div_2x):
  base        = accel top-10
  regime_gate = accel top-10 INTERSECT regime-favored (keep >=MIN)
  regime_union= accel top-10 UNION current-regime top-favored sectors
  regime_tilt = accel top-10, extra 1.5x weight on regime-favored picks (on top of div_2x)
-> BacktestResult[regime_conditioning]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/regime_conditioning_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings, price_basis
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

TOP_N = 10
CONVICTION_MULT = 2.0
REGIME_TILT = 1.5
MIN_DVOL = 5e6
MIN_SECTORS = 4
FAVORED_K = 15          # top-K sectors by past regime-conditional relative return
MIN_STATE_OBS = 5       # need >=this many past same-state months to trust the leader set
WARMUP = 15             # let regime history accumulate


def _perf(port_rets, spy_rets):
    r = np.asarray(port_rets, float); n = len(r)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100 if n else 0.0
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eqc = np.cumprod(1 + r); dd = float(((eqc / np.maximum.accumulate(eqc)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), annual=round(ann, 1),
                sharpe=round(sh, 2), dd=round(dd, 1), periods=n, spy_total=round(sp, 1))


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

    # ---- regime axes (PIT) ----
    tlt = etf_daily.get("TLT"); tip = etf_daily.get("TIP")
    tlt_m = tlt["Close"].resample("ME").last().reindex(midx) if tlt is not None else None
    tip_m = tip["Close"].resample("ME").last().reindex(midx) if tip is not None else None
    spy200 = (etf_daily[BENCH]["Close"] / etf_daily[BENCH]["Close"].rolling(200).mean() - 1).resample("ME").last().reindex(midx)
    rates_rising = (tlt_m.pct_change(3) < 0) if tlt_m is not None else pd.Series(False, index=midx)
    infl_rising = ((tip_m / tlt_m).pct_change(3) > 0) if (tlt_m is not None and tip_m is not None) else pd.Series(False, index=midx)
    risk_on = spy200 > 0
    state = pd.Series([(bool(rates_rising.iloc[i]), bool(infl_rising.iloc[i]), bool(risk_on.iloc[i])) for i in range(len(midx))], index=midx)

    etf_fwd_rel = etf_m.pct_change().shift(-1).sub(spy_m.pct_change().shift(-1), axis=0)  # sector fwd 1mo minus SPY

    def favored_at(i):
        """expanding-window: sectors with best mean forward-rel return in PAST months sharing month i's regime."""
        s_now = state.iloc[i]
        past = [j for j in range(len(midx)) if j < i and j + 1 <= i and state.iloc[j] == s_now]
        if len(past) < MIN_STATE_OBS:
            return None
        means = etf_fwd_rel.iloc[past].mean().dropna()
        if means.empty:
            return None
        return set(means.sort_values(ascending=False).head(FAVORED_K).index)

    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    px_at = price_basis.as_traded_close(px, price_basis.refresh_splits(list(common)))
    pb = (px_at * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    adl_m, dvol = {}, {}
    for t in common:
        d = stock_daily.get(t)
        if d is None or "Volume" not in d or len(d) < 90:
            continue
        v = d["Volume"]
        dvol[t] = (d["Close"] * v).rolling(20).mean().resample("ME").last().reindex(midx)
        if {"High", "Low", "Close"}.issubset(d.columns):
            rng = (d["High"] - d["Low"]).replace(0, np.nan)
            mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng
            adl_m[t] = (mfm.fillna(0) * v).cumsum().resample("ME").last().reindex(midx)
    dvol = pd.DataFrame(dvol).reindex(index=midx, columns=common)
    adl = pd.DataFrame(adl_m).reindex(index=midx, columns=common)
    ad_slope3 = adl - adl.shift(3); px_ret3 = px.pct_change(3)
    print(f"months {len(midx)} | stocks {len(common)} | regime states seen: {state.iloc[WARMUP:].nunique()}", flush=True)

    def pick_for(etf, date):
        _, holds = sector_map.get(etf, (etf, []))
        c = [h for h in holds if h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h])
             and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])
             and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= MIN_DVOL]
        g = [x for x in c if bool(low.loc[date, x])] or c
        if not g:
            return None
        pick = min(g, key=lambda h: pb.loc[date, h])
        adsl, pr3 = ad_slope3.loc[date], px_ret3.loc[date]
        is_div = bool(pd.notna(adsl.get(pick)) and pd.notna(pr3.get(pick)) and adsl.get(pick) > 0 and pr3.get(pick) < 0)
        return pick, (CONVICTION_MULT if is_div else 1.0)

    def run(mode):
        port_rets, spy_rets = [], []
        for i in range(WARMUP, len(midx) - 1):
            date, ndate = midx[i], midx[i + 1]
            spx = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
            if not np.isfinite(spx):
                continue
            accel_top = list(accel.iloc[i].dropna().sort_values(ascending=False).head(TOP_N).index)
            fav = favored_at(i)
            if mode == "base" or fav is None:
                sectors = accel_top
            elif mode == "regime_gate":
                inter = [e for e in accel_top if e in fav]
                sectors = inter if len(inter) >= MIN_SECTORS else accel_top
            elif mode == "regime_union":
                extra = [e for e in fav if e not in accel_top][:5]
                sectors = accel_top + extra
            else:
                sectors = accel_top
            picks = []
            for etf in sectors:
                pr = pick_for(etf, date)
                if pr is None:
                    continue
                pick, w = pr
                if mode == "regime_tilt" and fav is not None and etf in fav:
                    w *= REGIME_TILT
                r = _ret_delist(px[pick], date, ndate)
                if r is None or not np.isfinite(r):
                    continue
                picks.append((w, float(r)))
            if picks:
                tw = sum(w for w, _ in picks)
                port_rets.append(sum(w * r for w, r in picks) / tw); spy_rets.append(float(spx))
        return _perf(port_rets, spy_rets)

    out = {m: run(m) for m in ("base", "regime_gate", "regime_union", "regime_tilt")}
    print("\n=== REGIME CONDITIONING (honest engine, expanding-window PIT regime leaders) ===", flush=True)
    print(f"  {'variant':14} {'total':>8} {'vsSPY':>8} {'Sharpe':>7} {'DD':>8} {'periods':>8}", flush=True)
    for k, s in out.items():
        print(f"  {k:14} {s['total']:>7}% {s['vs_spy']:>8} {s['sharpe']:>7} {s['dd']:>7}% {s['periods']:>8}", flush=True)
    base = out["base"]; best = max(out, key=lambda k: out[k]["sharpe"])
    helps = out[best]["sharpe"] > base["sharpe"] + 0.05 and best != "base"
    verdict = (f"Honest base (post-warmup) +{base['total']}%/Sh{base['sharpe']}. "
               f"gate {out['regime_gate']['total']}%/Sh{out['regime_gate']['sharpe']}; union {out['regime_union']['total']}%/Sh{out['regime_union']['sharpe']}; "
               f"tilt {out['regime_tilt']['total']}%/Sh{out['regime_tilt']['sharpe']}. " + (
               f"REGIME HELPS — '{best}' beats base out-of-sample (Sh {out[best]['sharpe']} vs {base['sharpe']})."
               if helps else
               "Regime conditioning does NOT beat pure acceleration out-of-sample (expanding-window leaders don't "
               "persist forward); the in-sample regime-leadership lift does not survive honest PIT. Accel already prices regime."))
    return {"computed_at": pd.Timestamp.utcnow().isoformat(),
            "params": {"top_n": TOP_N, "favored_k": FAVORED_K, "min_state_obs": MIN_STATE_OBS, "warmup": WARMUP,
                       "pb_basis": "as-traded (split-corrected)"},
            "results": out, "best": best, "verdict": verdict,
            "caveat": "Expanding-window regime leaders (past-only, no look-ahead). WARMUP=15 so base reads lower than "
                      "the full 313% (shorter window). Split-corrected pb, div_2x, $5M, PIT, survivorship. In-sample regime = one 5y run."}


def main():
    p = build()
    from pathlib import Path
    Path("/app/.data/studies/regime_conditioning.json").write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="regime_conditioning", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[regime_conditioning]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
