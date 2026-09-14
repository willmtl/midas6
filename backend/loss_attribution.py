#!/usr/bin/env python3
"""LOSS ATTRIBUTION (user): were the flagship's LOSING picks bought (a) at a HIGH trailing-earnings ratio (high
P/E TTM) or (b) CLOSE TO / ABOVE analyst consensus (low implied upside)? If losers cluster in high-P/E or
low-upside, a ceiling/floor removes losers = return-additive. Uses the real blotter flagship_history.json
(per-pick PIT `pe` + outcome `ret`) and joins analyst implied-upside = median target in (date-180,date] / buy-close.
Saves BacktestResult[loss_attribution] + JSON."""
import os, json, math, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from seq_fundamental_study import load_candles

TRACE = "/app/.data/studies/flagship_history.json"
ANALYST = "/app/.data/analyst_ratings.jsonl"
OUT = "/app/.data/studies/loss_attribution.json"


def main():
    d = json.load(open(TRACE))
    picks = []
    for m in d["months"]:
        if not m.get("ndate"):
            continue
        for p in m.get("picks") or []:
            if p.get("ticker") and p.get("ret") is not None:
                picks.append({"tk": p["ticker"], "date": m["date"], "ret": float(p["ret"]),
                              "pe": p.get("pe"), "w": float(p.get("weight") or 1.0),
                              "mc": p.get("mktcap_usd"), "sector": p.get("sector")})
    print(f"{len(picks)} picks from {len(d['months'])} months", flush=True)

    # analyst implied-upside at buy date = median(target in (date-180,date]) / buy-close - 1
    from collections import defaultdict
    byt = defaultdict(list)
    if Path(ANALYST).exists():
        for line in Path(ANALYST).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("price_target") and r.get("date") and r.get("ticker"):
                byt[r["ticker"]].append((pd.Timestamp(r["date"]), float(r["price_target"])))
    for tk in byt:
        byt[tk].sort()
    tickers = sorted({p["tk"] for p in picks})
    cand = {}
    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    for i in range(0, len(tickers), 40):
        cand.update(load_candles(tickers[i:i + 40]))
    px = {}
    for tk, df in cand.items():
        if df is not None and not df.empty:
            s = df["Close"].copy(); s.index = pd.to_datetime(s.index).normalize()
            px[tk] = s[~s.index.duplicated(keep="last")].sort_index()

    def buy_close(tk, dte):
        s = px.get(tk)
        if s is None:
            return None
        dte = pd.Timestamp(dte).normalize()
        idx = s.index[s.index <= dte]
        return float(s.loc[idx[-1]]) if len(idx) else None

    for p in picks:
        p["upside"] = None
        pts = byt.get(p["tk"])
        c = buy_close(p["tk"], p["date"])
        if pts and c and c > 0:
            d0 = pd.Timestamp(p["date"]); lo = d0 - pd.Timedelta(days=180)
            tv = [t for dd, t in pts if lo < dd <= d0]
            if tv:
                p["upside"] = float(np.median(tv)) / c - 1.0

    df = pd.DataFrame(picks)
    wins, losses = df[df.ret > 0], df[df.ret < 0]
    print(f"\nwin rate {len(wins)/len(df)*100:.1f}%  ({len(wins)} win / {len(losses)} loss)  avg ret {df.ret.mean()*100:+.2f}%", flush=True)

    def med(x):
        x = pd.to_numeric(x, errors="coerce").dropna()
        return (float(x.median()), float(x.mean()), len(x))

    for col in ["pe", "upside"]:
        wm, wa, wn = med(wins[col]); lm, la, ln = med(losses[col])
        print(f"\n{col:>7s}:  WINS  median={wm:8.2f} mean={wa:8.2f} (n={wn})", flush=True)
        print(f"{'':7s}   LOSS  median={lm:8.2f} mean={la:8.2f} (n={ln})", flush=True)

    def bucket_stats(col, edges, names):
        out = {}
        v = pd.to_numeric(df[col], errors="coerce")
        print(f"\n=== loss-rate & avg-return by {col} bucket ===", flush=True)
        print(f"{'bucket':>14s} {'n':>5s} {'loss%':>7s} {'avg_ret':>9s} {'wt_ret':>9s}", flush=True)
        cats = pd.cut(v, bins=edges, labels=names)
        for nm in names:
            sub = df[cats == nm]
            if len(sub) == 0:
                continue
            lr = (sub.ret < 0).mean() * 100
            wr = (sub.ret * sub.w).sum() / sub.w.sum()
            print(f"{nm:>14s} {len(sub):>5d} {lr:>6.1f}% {sub.ret.mean()*100:>+8.2f}% {wr*100:>+8.2f}%", flush=True)
            out[nm] = {"n": int(len(sub)), "loss_pct": float(lr), "avg_ret": float(sub.ret.mean()),
                       "wt_ret": float(wr)}
        # names with NO coverage of this column
        nano = df[v.isna()]
        if len(nano):
            print(f"{'(no '+col+')':>14s} {len(nano):>5d} {(nano.ret<0).mean()*100:>6.1f}% {nano.ret.mean()*100:>+8.2f}%", flush=True)
            out["no_data"] = {"n": int(len(nano)), "loss_pct": float((nano.ret < 0).mean() * 100),
                              "avg_ret": float(nano.ret.mean())}
        return out

    pe_b = bucket_stats("pe", [-1e9, 0, 15, 30, 50, 1e9], ["neg (no earn)", "0-15", "15-30", "30-50", "50+"])
    up_b = bucket_stats("upside", [-1e9, 0, 0.25, 0.5, 1.0, 1e9], ["<=0 (>=target)", "0-25%", "25-50%", "50-100%", "100%+"])

    # big losses specifically
    big = df[df.ret < -0.20]
    print(f"\nBIG LOSSES (ret < -20%): n={len(big)}  median pe={pd.to_numeric(big.pe,errors='coerce').median():.1f}  "
          f"median upside={pd.to_numeric(big.upside,errors='coerce').median()}", flush=True)

    payload = {"computed_at": dt.datetime.now().isoformat(), "n_picks": len(df),
               "win_rate": float(len(wins) / len(df)), "avg_ret": float(df.ret.mean()),
               "pe_wins_median": med(wins.pe)[0], "pe_loss_median": med(losses.pe)[0],
               "upside_wins_median": med(wins.upside)[0], "upside_loss_median": med(losses.upside)[0],
               "pe_buckets": pe_b, "upside_buckets": up_b,
               "upside_coverage": float(df.upside.notna().mean())}
    Path(OUT).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nupside coverage of picks: {df.upside.notna().mean()*100:.0f}%   wrote {OUT}", flush=True)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="loss_attribution",
            defaults={"payload": payload, "computed_at": timezone.now()})
        print("Saved BacktestResult[loss_attribution]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
