#!/usr/bin/env python3
"""What do WINNING vs LOSING pre-earnings run-up trades have in common? Profile every K=10 trade (ADV>=100M,
net 5bps) on features known AT ENTRY (t-K, PIT) plus the one contemporaneous driver (what SPY did during the
hold). Two outcome definitions:
  RAW win  = trade return > 0
  EXCESS win = trade return - SPY over the identical window > 0   (strips beta -> the actionable part)

Features at entry a=t-K (all PIT / no look-ahead):
  mom20   : close[a]/close[a-20]-1                 (trailing 1m momentum)
  rsi10   : Wilder RSI(10) at a
  vol20   : annualized stdev of daily ret over [a-19..a]
  beta60  : cov(stock,SPY)/var(SPY) over trailing 60d
  dvol    : 20d avg $volume at a (liquidity)
  pb      : shares*close[a]/total_equity (PIT fundamentals)
  prof    : TTM net_income>0 (PIT)
  prev_up : sign of prior earnings reaction (close[tp-1]->[tp+1])
  regime  : SPY above its 50d MA at entry (ex-ante market state)
  spy_win : SPY return over hold a->t-1 (CONTEMPORANEOUS beta capture -- explanatory, not a filter)

Reports: mean feature for RAW winners vs losers and EXCESS winners vs losers; corr(feature, raw ret) and
corr(feature, excess ret); win-rate by regime and by prev_up. Persists BacktestResult[preearn_winloss_profile].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/preearn_winloss_profile.py"""
import os, sys, json, math, bisect
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, PRICE_FLOOR

K = 10
LIQ = 100e6
COST = 5 / 1e4


def wilder_rsi(close, period=10):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().values
    rd = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean().values
    rs = np.divide(ru, rd, out=np.full_like(ru, np.nan), where=rd > 0)
    return 100 - 100 / (1 + rs)


def main():
    universe, _ = _universe()
    from core.models import EarningsEvent, Candle
    have = set(EarningsEvent.objects.values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have]
    edates = defaultdict(list)
    for tk, rd in EarningsEvent.objects.values_list("ticker", "report_date"):
        if tk in have:
            edates[tk].append(pd.Timestamp(rd))
    for tk in edates:
        edates[tk] = sorted(set(edates[tk]))

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()
    spy_ret_s = spy_c.pct_change()
    spy_ma50 = spy_c.rolling(50).mean()

    FEATS = ["mom20", "rsi10", "vol20", "beta60", "dvol", "pb", "prof", "prev_up", "regime", "spy_win"]
    rows = []            # dict per trade
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            dret = pd.Series(close, index=dates).pct_change().values
            rsi = wilder_rsi(close, 10)
            vol20 = pd.Series(dret).rolling(20).std().values * math.sqrt(252)
            di = dates.values
            spy_al = spy_c.reindex(dates).ffill().values
            spy_r_al = spy_ret_s.reindex(dates).fillna(0.0).values
            spy_ma_al = spy_ma50.reindex(dates).ffill().values
            # trailing 60d beta
            sr = pd.Series(spy_r_al); rr = pd.Series(dret).fillna(0.0)
            cov = rr.rolling(60).cov(sr); var = sr.rolling(60).var()
            beta = (cov / var).values
            # PIT fundamentals
            r = reps.get(tk); av = ni = te = shs = None
            if r is not None and len(r):
                rr2 = r.dropna(subset=["avail_date"]).sort_values("avail_date")
                av = pd.to_datetime(rr2["avail_date"]).values
                ni = rr2["net_income"].values if "net_income" in rr2 else None
                te = rr2["total_equity"].values if "total_equity" in rr2 else None
                shs = rr2["shares_outstanding"].values if "shares_outstanding" in rr2 else None
            evs = edates.get(tk, [])
            tpos = [int(np.searchsorted(di, np.datetime64(rd), side="left")) for rd in evs]
            for i, t in enumerate(tpos):
                if t >= n or t - 1 < 0:
                    continue
                a = t - K
                if a < 20 or close[a] <= PRICE_FLOOR or not np.isfinite(close[a]):
                    continue
                if not (np.isfinite(dvol20[a]) and dvol20[a] >= LIQ):
                    continue
                pa, pb_ = float(close[a]), float(close[t - 1])
                if not np.isfinite(pb_):
                    continue
                ret = pb_ / pa - 1.0 - COST
                sw = spy_al[t - 1] / spy_al[a] - 1.0 if spy_al[a] > 0 else 0.0
                exc = ret - sw
                # prev earnings reaction sign
                prev_up = np.nan
                if i > 0:
                    tp = tpos[i - 1]
                    if 0 <= tp - 1 and tp + 1 < n and np.isfinite(close[tp - 1]) and np.isfinite(close[tp + 1]) and close[tp - 1] > 0:
                        prev_up = 1.0 if (close[tp + 1] / close[tp - 1] - 1.0) > 0 else 0.0
                # pb / prof at entry
                pbv = prof = np.nan
                if av is not None and len(av):
                    j = bisect.bisect_right(av, np.datetime64(dates[a])) - 1
                    if j >= 0:
                        if ni is not None and j >= 3:
                            prof = 1.0 if np.nansum(ni[j - 3:j + 1]) > 0 else 0.0
                        if te is not None and shs is not None and np.isfinite(te[j]) and te[j] > 0 and np.isfinite(shs[j]):
                            pbv = (shs[j] * pa) / te[j]
                rows.append(dict(ret=ret, exc=exc,
                                 mom20=close[a] / close[a - 20] - 1.0 if close[a - 20] > 0 else np.nan,
                                 rsi10=rsi[a], vol20=vol20[a], beta60=beta[a], dvol=dvol20[a],
                                 pb=pbv, prof=prof, prev_up=prev_up,
                                 regime=1.0 if (np.isfinite(spy_ma_al[a]) and spy_al[a] > spy_ma_al[a]) else 0.0,
                                 spy_win=sw))
        done += len(ch)
        if done % 200 < 40:
            print(f"  scanned {done}/{len(names)}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\ntrades: {len(df)}   raw win {100*(df.ret>0).mean():.1f}%   excess win {100*(df.exc>0).mean():.1f}%", flush=True)

    def prof_table(mask_win, label):
        w = df[mask_win]; l = df[~mask_win]
        print(f"\n=== {label}: mean feature — WINNERS (n={len(w)}) vs LOSERS (n={len(l)}) ===", flush=True)
        print(f"  {'feature':10}{'winners':>12}{'losers':>12}{'diff':>12}", flush=True)
        tab = {}
        for f in FEATS:
            wv = float(np.nanmean(w[f])); lv = float(np.nanmean(l[f]))
            tab[f] = dict(win=round(wv, 4), lose=round(lv, 4), diff=round(wv - lv, 4))
            print(f"  {f:10}{wv:>12.4f}{lv:>12.4f}{wv-lv:>12.4f}", flush=True)
        return tab

    out = {"n": len(df), "raw_win_pct": round(100 * (df.ret > 0).mean(), 1),
           "excess_win_pct": round(100 * (df.exc > 0).mean(), 1)}
    out["raw"] = prof_table(df.ret > 0, "RAW win/loss")
    out["excess"] = prof_table(df.exc > 0, "EXCESS-vs-SPY win/loss (beta stripped)")

    # correlations
    print(f"\n=== corr(feature, return) ===", flush=True)
    print(f"  {'feature':10}{'corr_raw':>12}{'corr_excess':>14}", flush=True)
    cor = {}
    for f in FEATS:
        v = df[f].values.astype(float)
        m = np.isfinite(v) & np.isfinite(df.ret.values)
        cr = float(np.corrcoef(v[m], df.ret.values[m])[0, 1]) if m.sum() > 50 else np.nan
        me = np.isfinite(v) & np.isfinite(df.exc.values)
        ce = float(np.corrcoef(v[me], df.exc.values[me])[0, 1]) if me.sum() > 50 else np.nan
        cor[f] = dict(raw=round(cr, 4), excess=round(ce, 4))
        print(f"  {f:10}{cr:>12.4f}{ce:>14.4f}", flush=True)
    out["corr"] = cor

    # conditional win/return tables for the categorical/ex-ante levers
    print(f"\n=== win-rate & mean by REGIME (SPY>50dMA at entry) and PREV_UP ===", flush=True)
    cond = {}
    for f in ("regime", "prev_up"):
        print(f"  -- {f} --", flush=True)
        for val in (1.0, 0.0):
            s = df[df[f] == val]
            if len(s) < 50:
                continue
            cond[f"{f}={int(val)}"] = dict(n=len(s), raw_win=round(100 * (s.ret > 0).mean(), 1),
                                           mean_ret=round(100 * s.ret.mean(), 3),
                                           excess_win=round(100 * (s.exc > 0).mean(), 1),
                                           mean_exc=round(100 * s.exc.mean(), 3))
            print(f"     {f}={int(val)}: n={len(s)} raw_win {100*(s.ret>0).mean():.1f}% mean {100*s.ret.mean():.3f}% "
                  f"| excess_win {100*(s.exc>0).mean():.1f}% mean_exc {100*s.exc.mean():.3f}%", flush=True)
    out["conditional"] = cond

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), liq=LIQ, k=K, cost_bps=5, results=out,
                   caveat="Winner/loser profile of K=10 pre-earnings run-up trades (ADV>=100M). RAW win vs "
                   "EXCESS-vs-SPY win. Features PIT at entry t-K except spy_win (contemporaneous beta capture). "
                   "regime=SPY>50dMA at entry. Full daily history.")
    Path("/app/.data/studies/preearn_winloss_profile.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="preearn_winloss_profile",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[preearn_winloss_profile]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
