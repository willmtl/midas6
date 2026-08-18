"""
FINVIZ VERSION — rotation engine (SEPARATE from survivorship_smallcap_study.py).

Same core idea as the flagship (rank momentum, buy the cheap small-cap value stock inside the hot
groups, monthly), but the rotation UNIT is the Finviz INDUSTRY (149 of them, full membership) instead
of 91 hand-picked ETFs. Because an industry has no single tradable price, its momentum is measured on
an EQUAL-WEIGHT index built from its own constituents' candles.

Independence: this file shares only the stable DATA/PIT LIBRARY layer that both versions draw on
(seq_fundamental_study / trend_stock_studies / backtest_lowpb / price_basis) — NOT the ETF engine's
logic. The two can be run side by side and compared. Nothing here imports survivorship_smallcap_study
or mutates config.py / sector_holdings.py.

  MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/finviz_rotation_study.py
"""
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
django.setup()

import numpy as np
import pandas as pd

from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel
from backtest_lowpb import _monthly_close, _tstat_from_returns, BENCH
import price_basis
import finviz_config
from core.models import BacktestResult
from django.utils import timezone

MIN_PB = 0.1          # P/B sanity floor (same as flagship): sub-0.1 = corrupt fundamentals
SMALL = 2e9           # small-cap ceiling ($2B)
MIN_MKTCAP = 5e7      # liquidity/quality floor ($50M) — v1 proxy (no volume filter yet)
MIN_PRICE = 1.0       # no sub-$1 names
IND_MIN_MEMBERS = 3   # need >=3 members with candles for a stable equal-weight industry index
WARMUP = 6            # months before first pick (accel needs pct_change(3).shift(3))


def _pit_ttm_sum(reps, field, midx):
    """Point-in-time trailing-12-month SUM of a quarterly FLOW field (e.g. net income), forward-filled by
    avail_date to the monthly index. Copied (not imported) to keep the Finviz engine independent of the ETF
    engine — same logic the flagship uses for TTM figures."""
    out = {}
    for tk, r in reps.items():
        if field not in r.columns:
            continue
        d = r[["period_end", "avail_date", field]].dropna(subset=[field]).copy()
        if len(d) < 4:
            continue
        d = d.sort_values("period_end")
        d["ttm"] = d[field].rolling(4).sum()
        s = pd.Series(d["ttm"].values, index=pd.to_datetime(d["avail_date"])).dropna()
        if s.empty:
            continue
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
    return pd.DataFrame(out)


def _analyst_upside_panel(midx, price_panel, colset, path="/app/.data/analyst_ratings.jsonl", stale_days=120):
    """PIT implied-upside = latest analyst price-target / as-traded price − 1, per ticker per month-end.
    Reads the Benzinga archive jsonl; forward-fills the last target but NULLS it once it's > stale_days old
    (so upside decays like the ETF flagship's panel). Returns a midx × colset DataFrame (NaN where uncovered)."""
    import json as _j
    from collections import defaultdict
    rows = defaultdict(dict)   # ticker -> {date: target}
    try:
        with open(path) as f:
            for line in f:
                try:
                    r = _j.loads(line)
                except Exception:
                    continue
                t = r.get("ticker")
                pt = r.get("price_target") or r.get("adjusted_price_target")
                d = r.get("date")
                if t in colset and pt and d:
                    rows[t][d] = float(pt)
    except Exception:
        return pd.DataFrame()
    tgt = {}
    for t, dv in rows.items():
        s = pd.Series(dv)
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        asof = s.reindex(s.index.union(midx)).ffill().reindex(midx)
        # staleness: date of the last target as-of each month; null if older than stale_days
        last_dt = pd.Series(s.index, index=s.index).reindex(s.index.union(midx)).ffill().reindex(midx)
        age = (pd.Series(midx, index=midx) - pd.to_datetime(pd.Series(last_dt.values, index=midx))).dt.days
        asof[(age > stale_days).values] = np.nan
        tgt[t] = asof
    if not tgt:
        return pd.DataFrame()
    tp = pd.DataFrame(tgt).reindex(index=midx, columns=list(colset))
    return tp / price_panel.reindex(index=midx, columns=list(colset)) - 1.0


def _max_drawdown(rets):
    """Max drawdown of the compounded monthly-return series (as a negative %)."""
    if not rets:
        return 0.0
    eq = np.cumprod([1 + r for r in rets])
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min() * 100)


def _aggregate_trace(tm):
    """Build the same tearsheet sections the ETF flagship has (leaderboard / industries / best-worst /
    calendar / equity curve) from the detailed per-month trace."""
    from collections import defaultdict
    stk = defaultdict(lambda: {"contrib": 0.0, "n": 0, "rets": [], "industry": "", "company": ""})
    ind = defaultdict(lambda: {"in_top": 0, "picked": 0, "skipped": 0, "rets": []})
    allp, curve = [], []
    eqf = eqs = 1.0
    yr = defaultdict(lambda: {"s": [], "sp": []})
    for m in tm:
        ws = len(m["picks"]) or 1
        for it in m["top_industries"]:
            ind[it["industry"]]["in_top"] += 1
        for sk in m["skipped"]:
            ind[sk["industry"]]["skipped"] += 1
        for p in m["picks"]:
            v = stk[p["ticker"]]
            v["contrib"] += (1 / ws) * p["ret"]; v["n"] += 1; v["rets"].append(p["ret"])
            v["industry"] = p["industry"]; v["company"] = p["company"]
            ind[p["industry"]]["picked"] += 1; ind[p["industry"]]["rets"].append(p["ret"])
            allp.append({**p, "date": m["date"]})
        eqf *= (1 + m["basket_ret"])
        if m["spy_ret"] is not None:
            eqs *= (1 + m["spy_ret"])
        curve.append({"d": m["date"], "f": round(eqf * 100000), "s": round(eqs * 100000)})
        y = m["date"][:4]; yr[y]["s"].append(m["basket_ret"])
        if m["spy_ret"] is not None:
            yr[y]["sp"].append(m["spy_ret"])
    leaderboard = sorted(({"ticker": t, "company": v["company"], "industry": v["industry"], "held": v["n"],
                           "contrib": round(v["contrib"] * 100, 1), "avg": round(float(np.mean(v["rets"])) * 100, 1),
                           "win": round(100 * sum(1 for x in v["rets"] if x > 0) / len(v["rets"])) if v["rets"] else 0}
                          for t, v in stk.items()), key=lambda x: -x["contrib"])
    industries = sorted(({"industry": n, "in_top": v["in_top"], "picked": v["picked"], "skipped": v["skipped"],
                          "avg": round(float(np.mean(v["rets"])) * 100, 1) if v["rets"] else None}
                         for n, v in ind.items()), key=lambda x: -x["picked"])
    cal = []
    for y in sorted(yr):
        s = float(np.prod([1 + x for x in yr[y]["s"]]) - 1) * 100
        sp = float(np.prod([1 + x for x in yr[y]["sp"]]) - 1) * 100 if yr[y]["sp"] else 0.0
        cal.append({"year": y, "strategy": round(s, 1), "spy": round(sp, 1),
                    "excess": round(s - sp, 1), "months": len(yr[y]["s"])})
    return {"leaderboard": leaderboard, "industries": industries,
            "best": sorted(allp, key=lambda x: -x["ret"])[:20], "worst": sorted(allp, key=lambda x: x["ret"])[:20],
            "calendar": cal, "curve": curve, "n_stocks": len(stk),
            "final_100k_flagship": round(eqf * 100000), "final_100k_spy": round(eqs * 100000)}


def _annualized_sharpe(rets):
    r = pd.Series(rets).dropna()
    if r.std() == 0 or len(r) < 3:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(12))


def build():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")

    _usca = os.environ.get("FINVIZ_USCA", "1") != "0"          # FINVIZ_USCA=0 -> include ALL countries (US-listed ADRs, USD)
    members = finviz_config.industry_members(usca_only=_usca)   # {'Sector > Industry': [tickers]}
    all_t = sorted({t for ts in members.values() for t in ts})
    print(f"finviz universe: {len(members)} industries, {len(all_t)} tickers "
          f"({'US/CA only' if _usca else 'ALL countries'})", flush=True)

    daily = load_candles(all_t + [BENCH])
    stock_m = _monthly_close({t: d for t, d in daily.items() if t in all_t})
    midx = stock_m.index
    spy_m = daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    spy_ret = spy_m.pct_change()

    # ── industry EQUAL-WEIGHT momentum index -> acceleration (same 6mo 2nd-derivative as the flagship) ──
    mret = stock_m.pct_change()
    ind_ret = {}
    for name, ts in members.items():
        cols = [t for t in ts if t in stock_m.columns]
        if len(cols) >= IND_MIN_MEMBERS:
            ind_ret[name] = mret[cols].mean(axis=1, skipna=True)     # equal-weight constituent return
    ind_ret = pd.DataFrame(ind_ret)
    ind_px = (1 + ind_ret.fillna(0)).cumprod()
    accel = ind_px.pct_change(3) - ind_px.pct_change(3).shift(3)     # rotation signal
    print(f"industries with a stable index: {ind_ret.shape[1]}", flush=True)

    # ── PIT P/B panel (identical construction to the flagship: mktcap / total_equity, avail-dated) ──
    reps = load_financial_reports(all_t)
    sh = _pit_monthly_panel(reps, "shares_outstanding", midx)
    eq = _pit_monthly_panel(reps, "total_equity", midx)
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    px = stock_m[common]
    sh = sh.reindex(index=midx, columns=common)
    eq = eq.reindex(index=midx, columns=common)
    as_traded = price_basis.as_traded_close(px)                     # split-adjusted 'as traded' close
    mktcap = as_traded * sh
    pb = mktcap / eq.where(eq != 0)

    # ── QUALITY / LIQUIDITY panels — the filters the ETF gives us for free (index inclusion), rebuilt
    # explicitly for the Finviz raw universe. ROE = TTM net income / equity (profit gate); dvol = trailing
    # 63-day mean dollar-volume (liquidity gate). ──
    ttm_ni = _pit_ttm_sum(reps, "net_income", midx).reindex(index=midx, columns=common)
    roe = ttm_ni / eq.where(eq != 0)
    dvol_cols = {}
    for t in common:
        dfr = daily.get(t)
        if dfr is None or "Volume" not in dfr.columns:
            continue
        dv = (dfr["Close"] * dfr["Volume"]).rolling(63, min_periods=20).mean().resample("ME").last()
        dvol_cols[t] = dv.reindex(midx)
    dvol = pd.DataFrame(dvol_cols).reindex(index=midx, columns=common)

    # ── analyst IMPLIED-UPSIDE panel (PIT): latest Benzinga price-target / price − 1, forward-filled to each
    # month-end, nulled if the last target is older than STALE days. The ETF flagship's single biggest lever. ──
    upside = _analyst_upside_panel(midx, as_traded, set(common))
    _cov = int(upside.notna().any().sum()) if not upside.empty else 0
    print(f"PIT panels: {len(common)} tickers | P/B, ROE, dollar-volume | analyst-upside covers {_cov}", flush=True)

    ind_cols = {name: [t for t in ts if t in common] for name, ts in members.items()}

    meta = finviz_config.ticker_meta()

    def run(top_k=10, start=None, end=None, min_mktcap=3e8, require_profit=False,
            min_dvol=0.0, pb_floor=MIN_PB, value_key="pb", trace=False):   # DEFAULT floor $300M; value_key "pb"/"blend"
        rets = []
        picks_log = []
        tmonths = []          # detailed per-month record (only when trace=True)
        idx = [d for d in midx[WARMUP:-1]]
        for d in idx:
            if start and d < pd.Timestamp(start):
                continue
            if end and d > pd.Timestamp(end):
                continue
            i = midx.get_loc(d)
            ndate = midx[i + 1]
            row = accel.loc[d].dropna()
            if row.empty:
                continue
            top = list(row.sort_values(ascending=False).head(top_k).index)
            held = set()
            month_rets = []
            month_names = []
            t_top, t_picks, t_skips = [], [], []
            for name in top:
                if trace:
                    _ir = ind_ret.loc[ndate, name] if name in ind_ret.columns else np.nan
                    t_top.append({"industry": name, "accel": float(accel.loc[d, name]) if name in accel.columns else None,
                                  "ind_ret": (float(_ir) if pd.notna(_ir) else None)})
                cands = []
                for t in ind_cols.get(name, []):
                    if t in held:
                        continue
                    p_pb = pb.loc[d, t] if t in pb.columns else np.nan
                    p_mc = mktcap.loc[d, t] if t in mktcap.columns else np.nan
                    p_px = as_traded.loc[d, t] if t in as_traded.columns else np.nan
                    if not (pd.notna(p_pb) and p_pb > pb_floor):
                        continue
                    if not (pd.notna(p_mc) and min_mktcap < p_mc < SMALL):     # small-cap
                        continue
                    if not (pd.notna(p_px) and p_px >= MIN_PRICE):
                        continue
                    if require_profit:                                        # profit gate (ROE>0) — the ETF's free lunch
                        p_roe = roe.loc[d, t] if t in roe.columns else np.nan
                        if not (pd.notna(p_roe) and p_roe > 0):
                            continue
                    if min_dvol > 0:                                          # liquidity gate (dollar-volume)
                        p_dv = dvol.loc[d, t] if t in dvol.columns else np.nan
                        if not (pd.notna(p_dv) and p_dv >= min_dvol):
                            continue
                    p_up = upside.loc[d, t] if (not upside.empty and t in upside.columns) else np.nan
                    cands.append((t, float(p_pb), (float(p_up) if pd.notna(p_up) else None)))
                if not cands:
                    if trace:
                        t_skips.append({"industry": name, "reason": "no qualifying small-cap value name (P/B, $300M size, price)"})
                    continue                                                   # SKIP (same as flagship)
                if value_key == "blend":     # 60% analyst-upside + 40% cheap-P/B rank blend; fallback cheapest-P/B
                    q = [(ct, cpb, cup) for ct, cpb, cup in cands if cup is not None]
                    if len(q) >= 3:
                        pr = pd.Series({ct: cpb for ct, cpb, cup in q}).rank(pct=True)                 # low P/B good
                        ur = pd.Series({ct: cup for ct, cpb, cup in q}).rank(pct=True, ascending=False)  # high upside good
                        t = (0.6 * ur + 0.4 * pr).idxmin()
                    else:
                        t = min(cands, key=lambda x: x[1])[0]
                else:
                    t = min(cands, key=lambda x: x[1])[0]                       # cheapest P/B
                held.add(t)
                p0 = as_traded.loc[d, t]
                p1 = as_traded.iloc[i + 1][t] if t in as_traded.columns else np.nan
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    _r = p1 / p0 - 1.0
                    month_rets.append(_r)
                    month_names.append(t)
                    if trace:
                        _mc = mktcap.loc[d, t] if t in mktcap.columns else np.nan
                        _up = upside.loc[d, t] if (not upside.empty and t in upside.columns) else np.nan
                        t_picks.append({"ticker": t, "industry": name,
                                        "company": meta.get(t, {}).get("company", ""),
                                        "pb": round(float(pb.loc[d, t]), 2) if pd.notna(pb.loc[d, t]) else None,
                                        "mktcap": (float(_mc) if pd.notna(_mc) else None),
                                        "roe": round(float(roe.loc[d, t]), 3) if (t in roe.columns and pd.notna(roe.loc[d, t])) else None,
                                        "upside": round(float(_up), 3) if pd.notna(_up) else None,
                                        "ret": round(_r, 4)})
            if month_rets:
                br = float(np.mean(month_rets))
                rets.append(br)
                _spy = float(spy_ret.loc[ndate]) if pd.notna(spy_ret.loc[ndate]) else None
                picks_log.append({"date": str(d.date()), "n": len(month_names),
                                  "picks": month_names, "ret": round(br * 100, 2),
                                  "spy": round(_spy * 100, 2) if _spy is not None else None})
                if trace:
                    tmonths.append({"date": str(d.date()), "ndate": str(ndate.date()),
                                    "basket_ret": br, "spy_ret": _spy,
                                    "top_industries": t_top, "picks": t_picks, "skipped": t_skips})
        total = (np.prod([1 + r for r in rets]) - 1) * 100 if rets else 0.0
        # matched SPY over the same months
        sp = [picks_log[k]["spy"] / 100 for k in range(len(picks_log)) if picks_log[k]["spy"] is not None]
        spy_total = (np.prod([1 + r for r in sp]) - 1) * 100 if sp else 0.0
        return {"total": round(total, 1), "spy_total": round(spy_total, 1),
                "sharpe": round(_annualized_sharpe(rets), 2), "dd": round(_max_drawdown(rets), 1),
                "months": len(rets), "picks_log": picks_log, "tmonths": tmonths}

    wins = [("FULL", None, None), ("ex-2020", "2021-01-31", None),
            ("H1 19-22", None, "2022-12-31"), ("H2 23-26", "2023-01-31", None)]
    print("\n=== FINVIZ INDUSTRY ROTATION (equal-weight accel -> cheapest-P/B small-cap, monthly) ===", flush=True)
    print(f"  {'window':12}{'finviz%':>12}{'SPY%':>10}{'Sharpe':>8}{'months':>8}", flush=True)
    full = None
    for lab, sd, ed in wins:
        r = run(top_k=10, start=sd, end=ed, value_key="blend", trace=(lab == "FULL"))  # v2 default = $300M floor + blend
        if lab == "FULL":
            full = r
        print(f"  {lab:12}{r['total']:>11.0f}%{r['spy_total']:>9.0f}%{r['sharpe']:>8.2f}{r['dd']:>8.1f}%{r['months']:>7}", flush=True)

    # top_k sensitivity on FULL
    print("\n  top_k sensitivity (FULL window):", flush=True)
    for k in (5, 8, 10, 15):
        r = run(top_k=k)
        print(f"    top_{k:<3} {r['total']:>8.0f}%  Sharpe {r['sharpe']:.2f}", flush=True)

    # ── RULE-TIGHTENING sweep: rebuild the quality/liquidity filter the ETF provides implicitly. Each row
    # is walked forward (FULL + the two halves) so a lever has to hold up, not just win in-sample. ──
    variants = [
        ("naive v1 (raw P/B, $50M)", dict(min_mktcap=5e7)),
        ("size floor $300M (v2 default)", dict()),
        ("$300M + profit gate (ROE>0)", dict(require_profit=True)),
        ("$300M + liquidity $5M/day", dict(min_dvol=5e6)),
        ("$300M + top-5 concentration", dict(top_k=5)),
        ("$300M + analyst-upside blend", dict(value_key="blend")),
        ("$300M + blend + top-5", dict(value_key="blend", top_k=5)),
    ]
    print("\n  RULE-TIGHTENING sweep (the ETF's implicit filter + ETF levers) — FULL / H1 / H2 / DD:", flush=True)
    print(f"  {'variant':34}{'FULL':>11}{'H1':>10}{'H2':>10}{'Shrp':>7}{'MaxDD':>8}", flush=True)
    sweep = []
    best = None
    for lab, kw in variants:
        tk = kw.pop("top_k", 10)
        rF = run(top_k=tk, **kw)
        rH1 = run(top_k=tk, end="2022-12-31", **kw)
        rH2 = run(top_k=tk, start="2023-01-31", **kw)
        print(f"  {lab:34}{rF['total']:>10.0f}%{rH1['total']:>9.0f}%{rH2['total']:>9.0f}%{rF['sharpe']:>7.2f}{rF['dd']:>7.1f}%", flush=True)
        row = {"variant": lab, "params": {**{k: v for k, v in kw.items()}, "top_k": tk},
               "full": rF["total"], "h1": rH1["total"], "h2": rH2["total"], "sharpe": rF["sharpe"], "dd": rF["dd"]}
        sweep.append(row)
        # "best" = highest FULL that is ALSO positive-vs-nothing in BOTH halves (walk-forward-honest)
        if rH1["total"] > 0 and rH2["total"] > 0 and (best is None or rF["total"] > best[1]["full"]):
            best = (lab, row, kw)

    if best:
        print(f"\n  BEST walk-forward-honest variant: {best[0]} — FULL {best[1]['full']:.0f}% Sharpe {best[1]['sharpe']}", flush=True)

    BacktestResult.objects.update_or_create(kind="finviz_rotation", defaults=dict(computed_at=timezone.now(), payload={
        "engine": "finviz_industry_rotation_v2",
        "note": ("SEPARATE Finviz-taxonomy engine: rotate top-accel Finviz INDUSTRIES (equal-weight "
                 "constituent index), buy cheapest-P/B small-cap inside each, monthly. v2 adds the quality/"
                 "liquidity filters the ETF gives for free (profit gate ROE>0, size floor, dollar-volume "
                 "liquidity, P/B distress floor) — see tightening_sweep. Shares only the data/PIT library."),
        "universe": {"industries": ind_ret.shape[1], "tickers": len(all_t), "with_fundamentals": len(common)},
        "full": {k: v for k, v in full.items() if k not in ("picks_log", "tmonths")},
        "tightening_sweep": sweep,
        "best_variant": (best[1] if best else None),
        "picks_log": full["picks_log"],
        "sections": _aggregate_trace(full.get("tmonths", [])),
        "tmonths": full.get("tmonths", []),
    }))
    print("\nsaved BacktestResult[finviz_rotation]", flush=True)


if __name__ == "__main__":
    build()
