#!/usr/bin/env python3
"""EARNINGS-BEAT HOLD-EXTENSION study (user hypothesis, 2026-08-19): the flagship rotates monthly, but if a
name we're HOLDING reports a STRONG earnings beat during the hold month, maybe we should KEEP it a few more
months to ride the post-earnings drift (PEAD) instead of rotating out.

This is a DIAGNOSTIC (not a re-simulation): for every (month, held pick), we check whether an earnings report
landed inside the hold window (date, ndate], classify it by the GROUNDED score (so a 'beat-guided-down' is not
counted as a strong beat — see ground_earnings.py / grounded-earnings memory), and then measure the
OPPORTUNITY-COST of keeping vs rotating:

    keep_ret_N   = the SAME name's compounded return over the NEXT N months (starting at ndate = decision time)
    rotate_ret_N = the flagship BASKET's compounded return over those same N months (what we actually earn)
    delta_N      = keep_ret_N - rotate_ret_N      (>0 => keeping the beater beats rotating)

Point-in-time: the beat is known at ndate (the rotation decision), and the extended hold runs ndate -> ndate+N,
so there is no look-ahead. We bucket by grounded verdict (strong beat >=2 / mild beat ==1 / inline / miss <=-1 /
no earnings this month = control) and compare forward continuation across buckets. If strong-beat delta is
clearly positive AND beats the no-earnings control, the bet holds and a hold-extension overlay is worth wiring.

Reads /app/.data/studies/flagship_history.json (the 112,950% adaptive flagship trace).
-> BacktestResult[earnings_hold] + prints. Run: docker exec -w /app rotation-backend-1 python -u earnings_hold_study.py
"""
import os, sys, json, warnings
sys.path.insert(0, "/app")
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles

TRACE = Path("/app/.data/studies/flagship_history.json")
FWD = [1, 2, 3, 4, 6]                 # forward hold-extension horizons (months) to test
STRONG = 2.0                          # grounded_score >= STRONG = "strong beat" (beat + guided up, not guided-down)


def _prod(xs):
    """Compound a list of monthly simple returns (skip Nones); None if nothing usable."""
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.prod([1 + x for x in xs]) - 1) if xs else None


def main():
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")   # /dev/shm=64MB -> avoid parallel Candle DiskFull

    D = json.load(open(TRACE))
    months = D["months"]
    dates = [m["date"] for m in months]                          # buy dates (month-ends), sorted
    didx = {d: i for i, d in enumerate(dates)}
    basket = [m.get("basket_ret") for m in months]               # basket return over [date_i, ndate_i]
    tickers = sorted({p["ticker"] for m in months for p in m.get("picks", []) if p.get("ticker")})
    print(f"trace: {len(months)} months ({dates[0]}..{dates[-1]}), {len(tickers)} unique pick names, "
          f"flagship total {D['perf']['total']}%", flush=True)

    # ── per-name monthly return series, aligned to the trace's month-end dates ──
    cand = load_candles(tickers)
    dt_index = pd.to_datetime(dates)
    name_ret = {}                                                # ticker -> list aligned to `dates`: ret[i]=[date_i,date_{i+1}]
    for t in tickers:
        df = cand.get(t)
        if df is None or "Close" not in df or not len(df):
            name_ret[t] = [None] * len(dates); continue
        mc = df["Close"].resample("ME").last().reindex(dt_index)
        r = mc.pct_change().shift(-1)                            # r[i] = close[i+1]/close[i]-1  (return of month i)
        name_ret[t] = [float(x) if pd.notna(x) else None for x in r.values]

    # ── earnings events for the pick names (full history, grounded) ──
    from core.models import EarningsEvent
    ev = defaultdict(list)
    for e in (EarningsEvent.objects.filter(ticker__in=tickers)
              .values("ticker", "report_date", "grounded_score", "grounded_label",
                      "eps_surprise_pct", "guidance_eps_pct")):
        ev[e["ticker"]].append(e)

    def _beat_in_window(t, d0, d1):
        """Return the grounded event with the max grounded_score whose report_date is in (d0, d1], or None."""
        best = None
        for e in ev.get(t, []):
            rd = pd.Timestamp(e["report_date"])
            if d0 < rd <= d1:
                if best is None or (e.get("grounded_score") or -9) > (best.get("grounded_score") or -9):
                    best = e
        return best

    # ── classify every held pick-month and record forward keep/rotate returns ──
    buckets = {"huge_beat (surp>=50%)": [], "big_beat (surp 25-50%)": [], "beat (surp 0-25%)": [],
               "miss/inline (surp<=0)": [], "beat/miss (no surp%)": [], "no_earnings (control)": []}
    label_bkt = defaultdict(list)                                # grounded_label -> forward deltas (N=3)
    Nmax = max(FWD)
    rows = []                                                    # (bucket, i, ticker, {N: (keep,rotate,delta)})
    for i, m in enumerate(months):
        if i + 1 >= len(dates):
            continue
        d0 = pd.Timestamp(m["date"]); d1 = pd.Timestamp(m["ndate"]) if m.get("ndate") else None
        if d1 is None:
            continue
        for p in m.get("picks", []):
            t = p.get("ticker")
            if not t:
                continue
            e = _beat_in_window(t, d0, d1)
            # "STRONGLY beat" = large EPS surprise magnitude (grounded_score>=2 = beat-AND-guided-up is too rare,
            # ~0.4% of events -> empty sample). Surprise percentiles among beats: p75=29% p90=73%.
            sp = (e.get("eps_surprise_pct") if e else None)
            if e is None:
                bk = "no_earnings (control)"
            elif sp is None:
                bk = "beat/miss (no surp%)"
            elif sp >= 50:
                bk = "huge_beat (surp>=50%)"
            elif sp >= 25:
                bk = "big_beat (surp 25-50%)"
            elif sp > 0:
                bk = "beat (surp 0-25%)"
            else:
                bk = "miss/inline (surp<=0)"
            perN = {}
            for N in FWD:
                if i + N >= len(dates):
                    perN[N] = None; continue
                kr = [name_ret[t][i + k] for k in range(1, N + 1)]               # hold name over next N months
                br = [basket[i + k] for k in range(1, N + 1)]                    # rotate normally over same span
                # MATCHED SPAN: require the FULL N months present for BOTH legs (a delisted name whose candles end
                # mid-horizon would otherwise make keep span fewer months than rotate + drop its terminal move).
                if any(x is None or not np.isfinite(x) for x in kr) or any(x is None or not np.isfinite(x) for x in br):
                    perN[N] = None; continue
                keep = float(np.prod([1 + x for x in kr]) - 1)
                rot = float(np.prod([1 + x for x in br]) - 1)
                perN[N] = (keep, rot, keep - rot)
            buckets[bk].append(perN)
            if e is not None and e.get("grounded_label"):
                d3 = perN.get(3)
                if d3 and d3[2] is not None:
                    label_bkt[e["grounded_label"]].append(d3[2])
            rows.append((bk, m["date"], t, perN))

    # ── aggregate ──
    def _agg(lst, N):
        vals = [x[N] for x in lst if x.get(N) is not None]      # each is a full (keep,rot,delta) over matched span
        if not vals:
            return None
        keeps = [v[0] for v in vals]; rots = [v[1] for v in vals]; dels = [v[2] for v in vals]
        return dict(n=len(vals),
                    keep=float(np.mean(keeps)) * 100, rotate=float(np.mean(rots)) * 100,
                    delta=float(np.mean(dels)) * 100, delta_med=float(np.median(dels)) * 100,
                    winrate=float(np.mean([d > 0 for d in dels])) * 100)

    print("\n=== EARNINGS-BEAT HOLD-EXTENSION (keep the beater N more months vs rotate the basket) ===", flush=True)
    print("    keep = same name's fwd N-mo return | rotate = flagship basket fwd N-mo | delta = keep-rotate (>0 keep wins)\n", flush=True)
    hdr = f"  {'bucket':22}{'n':>5}"
    for N in FWD:
        hdr += f"  |  +{N}mo keep/rot/Δ  win%"
    print(hdr, flush=True)
    out = {}
    for bk, lst in buckets.items():
        line = f"  {bk:22}{len(lst):>5}"
        out[bk] = {}
        for N in FWD:
            a = _agg(lst, N)
            out[bk][N] = a
            if a:
                line += f"  | {a['keep']:>6.1f}/{a['rotate']:>5.1f}/{a['delta']:>+6.1f} {a['winrate']:>4.0f}"
            else:
                line += f"  |        n/a         "
        print(line, flush=True)

    print("\n  by grounded_label (mean +3mo delta, keep-minus-rotate; n):", flush=True)
    for lab, ds in sorted(label_bkt.items(), key=lambda kv: -np.mean(kv[1]) if kv[1] else 0):
        if len(ds) >= 3:
            print(f"    {lab:22} {np.mean(ds)*100:>+7.1f}%   (n={len(ds)})", flush=True)

    # verdict — the user's bet is on the STRONGEST beats (huge/big surprise)
    ctrl = out.get("no_earnings (control)", {})
    print("\n  VERDICT (does a STRONG beat identify names worth KEEPING vs rotating?):", flush=True)
    for lab in ("huge_beat (surp>=50%)", "big_beat (surp 25-50%)"):
        sb = out.get(lab, {})
        for N in FWD:
            a, c = sb.get(N), ctrl.get(N)
            if a and c:
                edge = a["delta"] - c["delta"]
                verdict = "KEEP beats rotate AND control" if (a["delta"] > 0 and edge > 0) else \
                          ("keep>rotate but not vs control" if a["delta"] > 0 else "rotate wins (keep hurts)")
                print(f"    {lab:22} +{N}mo: Δ {a['delta']:>+6.1f}% (n={a['n']}) vs control Δ {c['delta']:>+6.1f}% "
                      f"(edge {edge:>+6.1f}pp) -> {verdict}", flush=True)

    try:
        from core.models import BacktestResult
        from django.utils import timezone
        payload = {"buckets": {k: {str(n): v for n, v in d.items()} for k, d in out.items()},
                   "by_label": {k: {"mean_delta3_pct": float(np.mean(v) * 100), "n": len(v)} for k, v in label_bkt.items()},
                   "fwd_horizons": FWD, "strong_threshold": STRONG, "flagship_total": D["perf"]["total"],
                   "n_months": len(months), "n_pick_names": len(tickers)}
        BacktestResult.objects.update_or_create(
            kind="earnings_hold", defaults={"payload": json.loads(json.dumps(payload, default=str)),
                                            "computed_at": timezone.now()})
        print("\nSaved BacktestResult[earnings_hold]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
