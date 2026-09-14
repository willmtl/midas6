#!/usr/bin/env python3
"""EARNINGS-PREDICT v1 — stacked 'predict results' model (user: 'we need to find a way to predict results').

Two walk-forward models, numpy-only (container has no sklearn), all features STRICTLY point-in-time as of
the last close BEFORE the print (index f = t-1, t = first trading day >= report_date):
  MODEL A (fundamental surprise): logistic  P(EPS beat)   target = 1[eps_surprise_pct > 0]
  MODEL B (reaction, the money):  ridge      through-print SPY-EXCESS return, target = close(t-1)->close(t+1)
                                             demeaned vs SPY same window, net COST_BPS round-trip.
STACKED: after both are fit out-of-sample, bucket events by predicted-beat-prob x predicted-reaction and
show the ACTUAL reaction in each cell -> where fundamentals and price-setup DISAGREE = the mispricing.

Features (all PIT at f): prior-surprise persistence (last/avg4/streak/slope, only events with report_date <
current), pre-earnings run-up (5/10/21d, SPY-relative), technicals (RSI10, %vs SMA20/50, 63d mom, vol ramp),
PIT valuation/quality (pb, pe, profit_margin, op_margin, earnings_growth, revenue_growth, debt_to_equity,
pct_52w, beta) from prepare_pit_metrics, plus cap bucket / sector / calendar quarter context.

VALIDATION = the gate: expanding-window WALK-FORWARD, train on years < Y, predict Y, roll. OUT-OF-SAMPLE ONLY.
Reported: A -> OOS accuracy / AUC / Brier vs base rate; B -> OOS Spearman rank-IC per year + top-decile
hold-into-print basket actual mean (net, SPY-excess) BY YEAR and segmented CAP x SECTOR. Univariate
walk-forward decile sorts per feature show which signals carry any OOS power at all. Saved to
BacktestResult[earnings_predict] (+JSON). If it only works in-sample it is reported REFUTED.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_predict.py"""
import os, sys, json
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR
from pit_fundamentals import prepare_pit_metrics

COST_BPS = 10
RIDGE_LAMBDA = 10.0
START_YEAR = 2015                 # eps_surprise_pct fully populated from here
PIT_FEATS = ["pb", "pe", "profit_margin", "operating_margin", "earnings_growth",
             "revenue_growth", "debt_to_equity", "pct_52w", "beta"]


def cap_bucket(dvol):
    if dvol is None or not np.isfinite(dvol) or dvol <= 0:
        return "cap?"
    m = dvol / 1e6
    return "ADV>500M" if m >= 500 else "ADV100-500M" if m >= 100 else "ADV20-100M" if m >= 20 else "ADV5-20M"


def rsi(close, n=10):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    rs = np.divide(ru, rd, out=np.full_like(ru, np.nan), where=rd > 0)
    return 100 - 100 / (1 + rs)


def spearman_ic(pred, act):
    m = np.isfinite(pred) & np.isfinite(act)
    if m.sum() < 20:
        return np.nan
    rp = pd.Series(pred[m]).rank().to_numpy(dtype=float, copy=True)
    ra = pd.Series(act[m]).rank().to_numpy(dtype=float, copy=True)
    rp -= rp.mean(); ra -= ra.mean()
    d = np.sqrt((rp ** 2).sum() * (ra ** 2).sum())
    return float((rp * ra).sum() / d) if d > 0 else np.nan


def auc(score, y):
    m = np.isfinite(score)
    score, y = score[m], y[m]
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(score).rank().values
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle, Fundamental
    ev = list(EarningsEvent.objects.exclude(eps_surprise_pct=None)
              .filter(report_date__year__gte=START_YEAR)
              .values_list("ticker", "report_date", "eps_surprise_pct"))
    have = set(t for t, *_ in ev)
    names = [t for t in universe if t in have]
    sect = dict(Fundamental.objects.exclude(sector="").values_list("ticker", "sector")) \
        if any(f.name == "sector" for f in Fundamental._meta.get_fields()) else {}
    print(f"universe {len(universe)} | surprise-labelled tickers {len(have)} | tradable ∩ {len(names)} | events {len(ev)}", flush=True)

    per_tkr = defaultdict(list)
    for tk, rd, es in ev:
        per_tkr[tk].append((pd.Timestamp(rd), float(es)))
    for tk in per_tkr:
        per_tkr[tk].sort()

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    rows = []
    cost = COST_BPS / 1e4
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values.astype(float); vol = sdf["Volume"].values.astype(float)
            n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            sma20 = pd.Series(close).rolling(20, min_periods=10).mean().values
            sma50 = pd.Series(close).rolling(50, min_periods=25).mean().values
            r10 = rsi(close, 10)
            spy_al = spy_c.reindex(dates).ffill().values
            di = dates.values
            try:
                pit = prepare_pit_metrics(sdf, reps.get(tk), None, spy_c)
            except Exception:
                pit = None
            evs = per_tkr.get(tk, [])
            for i, (rd, es) in enumerate(evs):
                pos = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if pos >= n:
                    continue
                t = pos; f = t - 1
                if f - 21 < 0 or t + 1 >= n:
                    continue
                pf = close[f]
                if not (pf > PRICE_FLOOR and np.isfinite(pf) and dvol20[f] >= DVOL_FLOOR):
                    continue
                # ---- reaction target (SPY-excess, net cost) ----
                pa, pb_ = close[t - 1], close[t + 1]
                if not (pa > 0 and np.isfinite(pa) and np.isfinite(pb_)):
                    continue
                sa, sb = spy_al[t - 1], spy_al[t + 1]
                sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                react = (pb_ - pa) / pa - cost - sr
                # ---- prior-surprise persistence (strictly earlier events) ----
                prior = [e for (d0, e) in evs[:i]]
                last = prior[-1] if prior else np.nan
                p4 = prior[-4:]
                avg4 = float(np.mean(p4)) if p4 else np.nan
                streak = 0
                for e in reversed(prior):
                    if e > 0:
                        streak += 1
                    else:
                        break
                slope = np.nan
                if len(p4) >= 2:
                    slope = float(np.polyfit(range(len(p4)), p4, 1)[0])
                # ---- run-up (SPY-relative) ----
                def ret(k):
                    p0 = close[f - k]
                    if not (p0 > 0 and np.isfinite(p0)):
                        return np.nan
                    s0 = spy_al[f - k]; ssr = (spy_al[f] - s0) / s0 if (np.isfinite(s0) and s0 > 0) else 0.0
                    return (close[f] - p0) / p0 - ssr
                ru5, ru10, ru21 = ret(5), ret(10), ret(21)
                # ---- technicals ----
                rsi10 = r10[f]
                vsma20 = (close[f] / sma20[f] - 1) if (np.isfinite(sma20[f]) and sma20[f] > 0) else np.nan
                vsma50 = (close[f] / sma50[f] - 1) if (np.isfinite(sma50[f]) and sma50[f] > 0) else np.nan
                mom63 = ret(63) if f - 63 >= 0 else np.nan
                vr = np.nan
                if f - 26 >= 0:
                    a = vol[f - 5:f].mean(); b = vol[f - 26:f - 6].mean()
                    vr = a / b if b > 0 else np.nan
                row = dict(ticker=tk, year=int(pd.Timestamp(dates[t]).year), react=react,
                           beat=1 if es > 0 else 0, cap=cap_bucket(float(dvol20[f])), sector=sect.get(tk) or "?",
                           quarter=(pd.Timestamp(dates[t]).month - 1) // 3 + 1,
                           f_last=last, f_avg4=avg4, f_streak=streak, f_slope=slope,
                           f_ru5=ru5, f_ru10=ru10, f_ru21=ru21, f_rsi10=rsi10,
                           f_vsma20=vsma20, f_vsma50=vsma50, f_mom63=mom63, f_volramp=vr)
                if pit is not None:
                    for c in PIT_FEATS:
                        row[f"f_{c}"] = float(pit[c].iloc[f]) if (c in pit.columns and np.isfinite(pit[c].iloc[f])) else np.nan
                else:
                    for c in PIT_FEATS:
                        row[f"f_{c}"] = np.nan
                rows.append(row)
        done += len(ch)
        print(f"  scanned {done}/{len(names)}  events {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    FEATS = [c for c in df.columns if c.startswith("f_")]
    print(f"\nassembled {len(df)} events | {len(FEATS)} features | years {df.year.min()}-{df.year.max()}", flush=True)

    # ---------- univariate walk-forward decile sorts (raw OOS predictive power) ----------
    print("\n=== UNIVARIATE feature -> reaction (top-decile minus bottom-decile, %/trade, SPY-excess) ===", flush=True)
    print(f"  {'feature':12}{'n':>7}{'D10':>8}{'D1':>8}{'spread':>8}{'IC':>7}{'yr+':>6}", flush=True)
    uni = {}
    for c in FEATS:
        x = df[c].values.astype(float); y = df["react"].values * 100
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 500:
            continue
        xv, yv = x[m], y[m]
        q = pd.qcut(pd.Series(xv).rank(method="first"), 10, labels=False)
        d10 = yv[q == 9].mean(); d1 = yv[q == 0].mean()
        ic = spearman_ic(xv, yv)
        # by-year sign consistency of the spread
        yrs = df["year"].values[m]; pos = 0; tot = 0
        for Y in np.unique(yrs):
            mm = yrs == Y
            if mm.sum() < 50:
                continue
            qq = pd.qcut(pd.Series(xv[mm]).rank(method="first"), 10, labels=False)
            tot += 1; pos += 1 if (yv[mm][qq == 9].mean() - yv[mm][qq == 0].mean()) > 0 else 0
        uni[c] = dict(n=int(m.sum()), d10=round(d10, 3), d1=round(d1, 3), spread=round(d10 - d1, 3),
                      ic=round(ic, 4), yr_pos=f"{pos}/{tot}")
        print(f"  {c:12}{int(m.sum()):>7}{d10:>8.3f}{d1:>8.3f}{d10 - d1:>8.3f}{ic:>7.3f}{pos:>4}/{tot}", flush=True)

    # ---------- walk-forward multivariate: ridge(react) + logistic(beat) ----------
    Xall = df[FEATS].values.astype(float)
    med_all = np.nanmedian(Xall, axis=0)
    years = sorted(df["year"].unique())
    oos_pred_react = np.full(len(df), np.nan)
    oos_pred_beat = np.full(len(df), np.nan)
    coefs = []
    for Y in years:
        tr = df["year"].values < Y
        te = df["year"].values == Y
        if tr.sum() < 1000 or te.sum() < 30:
            continue
        Xtr = Xall[tr].copy(); Xte = Xall[te].copy()
        # impute train-median, winsorize train 1/99, standardize by train moments
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, med_all)
        for j in range(Xtr.shape[1]):
            Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]; Xte[~np.isfinite(Xte[:, j]), j] = med[j]
        lo = np.percentile(Xtr, 1, axis=0); hi = np.percentile(Xtr, 99, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
        mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
        Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
        Ztr1 = np.column_stack([np.ones(len(Ztr)), Ztr]); Zte1 = np.column_stack([np.ones(len(Zte)), Zte])
        # ridge (closed form) on reaction (do not penalize intercept)
        yr_ = df["react"].values[tr]
        P = np.eye(Ztr1.shape[1]) * RIDGE_LAMBDA; P[0, 0] = 0
        w = np.linalg.solve(Ztr1.T @ Ztr1 + P, Ztr1.T @ yr_)
        oos_pred_react[te] = Zte1 @ w
        coefs.append(w[1:])
        # logistic (GD) on beat
        yb = df["beat"].values[tr].astype(float)
        wl = np.zeros(Ztr1.shape[1]); lr = 0.3
        for _ in range(400):
            p = 1 / (1 + np.exp(-Ztr1 @ wl))
            g = Ztr1.T @ (p - yb) / len(yb) + np.r_[0, RIDGE_LAMBDA / len(yb) * wl[1:]]
            wl -= lr * g
        oos_pred_beat[te] = 1 / (1 + np.exp(-Zte1 @ wl))

    df["pred_react"] = oos_pred_react; df["pred_beat"] = oos_pred_beat
    scored = df[np.isfinite(df["pred_react"])].copy()

    # ---------- Model A (beat) OOS ----------
    yb = scored["beat"].values; pbp = scored["pred_beat"].values
    base = yb.mean(); acc = ((pbp > 0.5).astype(int) == yb).mean()
    print(f"\n=== MODEL A (beat) OOS: n={len(scored)} base_rate={base:.3f} acc={acc:.3f} "
          f"AUC={auc(pbp, yb):.3f} Brier={np.mean((pbp - yb) ** 2):.4f} ===", flush=True)

    # ---------- Model B (reaction) OOS ----------
    print("\n=== MODEL B (reaction) OOS rank-IC + top-decile hold-into-print basket (net, SPY-excess, %/trade) ===", flush=True)
    print(f"  {'year':>6}{'n':>7}{'IC':>8}{'D10_mean':>10}{'D10_win%':>10}{'all_mean':>10}", flush=True)
    byyear = {}
    for Y in years:
        s = scored[scored["year"] == Y]
        if len(s) < 50:
            continue
        ic = spearman_ic(s["pred_react"].values, s["react"].values)
        thr = s["pred_react"].quantile(0.9)
        d10 = s[s["pred_react"] >= thr]["react"].values * 100
        byyear[int(Y)] = dict(n=len(s), ic=round(ic, 4), d10_mean=round(float(d10.mean()), 3),
                              d10_win=round(float((d10 > 0).mean() * 100), 1),
                              all_mean=round(float(s["react"].mean() * 100), 3))
        print(f"  {Y:>6}{len(s):>7}{ic:>8.3f}{d10.mean():>10.3f}{(d10 > 0).mean() * 100:>10.1f}{s['react'].mean() * 100:>10.3f}", flush=True)
    ics = [v["ic"] for v in byyear.values() if np.isfinite(v["ic"])]
    d10s = [v["d10_mean"] for v in byyear.values()]
    print(f"  MEAN OOS IC={np.mean(ics):+.4f}  ({sum(1 for x in ics if x > 0)}/{len(ics)} yrs +)  |  "
          f"D10 basket mean={np.mean(d10s):+.3f}%  ({sum(1 for x in d10s if x > 0)}/{len(d10s)} yrs +)", flush=True)

    # ---------- top-decile basket segmented cap x sector ----------
    thr_all = scored.groupby("year")["pred_react"].transform(lambda s: s.quantile(0.9))
    top = scored[scored["pred_react"] >= thr_all]
    print(f"\n=== D10 basket by CAP x SECTOR (n>=40 cells, %/trade net SPY-excess) ===", flush=True)
    seg = {}
    for (cp, sc), g in top.groupby(["cap", "sector"]):
        if len(g) < 40:
            continue
        seg[f"{cp}|{sc}"] = dict(n=len(g), mean=round(float(g["react"].mean() * 100), 3),
                                 win=round(float((g["react"] > 0).mean() * 100), 1))
    for k in sorted(seg, key=lambda k: -seg[k]["mean"]):
        print(f"  {k:34}{seg[k]['n']:>6}{seg[k]['mean']:>9.3f}{seg[k]['win']:>8.1f}", flush=True)

    # ---------- STACKED: predicted-beat x predicted-reaction -> actual reaction ----------
    print(f"\n=== STACKED: actual reaction by pred_beat quartile x pred_react quartile (%/trade) ===", flush=True)
    scored["qb"] = pd.qcut(scored["pred_beat"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    scored["qr"] = pd.qcut(scored["pred_react"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    stack = {}
    print(f"  {'':10}" + "".join([f"predR_Q{q:>1}".rjust(11) for q in [1, 2, 3, 4]]), flush=True)
    for qb in [1, 2, 3, 4]:
        cells = []
        for qr in [1, 2, 3, 4]:
            g = scored[(scored["qb"] == qb) & (scored["qr"] == qr)]
            v = round(float(g["react"].mean() * 100), 3) if len(g) >= 30 else None
            stack[f"b{qb}r{qr}"] = dict(n=len(g), mean=v)
            cells.append(f"{v:+.3f}".rjust(11) if v is not None else "·".rjust(11))
        print(f"  predB_Q{qb:<2}" + "".join(cells), flush=True)

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), cost_bps=COST_BPS, start_year=START_YEAR,
                   n_events=len(df), n_scored=len(scored), features=FEATS,
                   ridge_lambda=RIDGE_LAMBDA,
                   univariate=uni,
                   model_a=dict(base_rate=round(float(base), 4), acc=round(float(acc), 4),
                                auc=round(float(auc(pbp, yb)), 4), brier=round(float(np.mean((pbp - yb) ** 2)), 5)),
                   model_b_by_year=byyear,
                   model_b_mean_ic=round(float(np.mean(ics)), 4),
                   model_b_ic_pos=f"{sum(1 for x in ics if x > 0)}/{len(ics)}",
                   model_b_d10_pos=f"{sum(1 for x in d10s if x > 0)}/{len(d10s)}",
                   feat_coef=dict(zip(FEATS, [round(float(x), 4) for x in np.mean(coefs, axis=0)])) if coefs else {},
                   d10_by_cap_sector=seg, stacked=stack,
                   caveat="Walk-forward OOS (train years<Y, predict Y). Target A=1[eps_surprise>0]; B=through-print "
                   "close(t-1)->close(t+1) SPY-excess net cost. Features strictly PIT at t-1. v1 predicts the SURPRISE "
                   "(fully-historical); guidance-RAISE prediction deferred (only ~13mo LLM-labelled). numpy-only "
                   "ridge+logistic; no GBM/SI/revision-momentum in v1. Success gate = year-stable positive OOS IC and "
                   "D10 basket beating SPY most years; else REFUTED.")
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/earnings_predict.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="earnings_predict",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[earnings_predict]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
