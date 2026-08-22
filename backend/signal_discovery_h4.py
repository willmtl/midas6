#!/usr/bin/env python3
"""SIGNAL DISCOVERY LAB — H4 (intraday) arm. Same standalone-per-signal, cross-sectionally-demeaned, winsorized,
regime-split, long/short harness as the daily arm — but on 4h bars, sub-daily horizons. SINGLE-PROCESS (562 names
is small; avoids the MP fragility of the daily arm).

⚠️ SURVIVOR-BIASED: 4h parquet exists for only the ~562 liquid large/mid-cap H4-candidate names, and NO delisted
names — so this arm cannot be survivorship-corrected the way the daily arm is. Treat as directional/large-cap-only.
Horizons are in 4h BARS (~2 bars/trading-day): 1,2,3,6,12,24 bars ≈ 0.5,1,1.5,3,6,12 trading days.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/signal_discovery_h4.py [--db]"""
import os, sys, json, math, glob
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

from studies import SIGNALS, MARKET_SIGNAL_KEYS
from core.models import Candle

HBARS = [1, 2, 3, 6, 12, 24]        # forward horizons in 4h bars
PRICE_FLOOR = 3.0
WINS = (-100.0, 150.0)
DEDUP_BARS = 6                      # collapse clustered fires within ~3 trading days into one episode
SKIP = set(MARKET_SIGNAL_KEYS) | {"rsi_x_pos_updn"}
PARQ = "/app/.data/intraday/4h"


def _wins(x):
    return float(min(max(x, WINS[0]), WINS[1]))


def _load_4h():
    out = {}
    for f in sorted(glob.glob(f"{PARQ}/*.parquet")):
        tk = f.split("/")[-1][:-8]
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if len(df) < 200 or "Close" not in df.columns:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[tk] = df
    return out


def main():
    frames = _load_4h()
    print(f"H4 universe: {len(frames)} names with 4h data (survivor-biased, large/mid-cap)", flush=True)
    # 4h EW-universe benchmark: mean 4h-bar return across names per timestamp, cumulated
    rets = {}
    for tk, df in frames.items():
        r = df["Close"].pct_change()
        rets[tk] = r[(r > -0.5) & (r < 0.5)]
    ew_ret = pd.DataFrame(rets).mean(axis=1)
    ew_level = (1.0 + ew_ret.fillna(0.0)).cumprod()
    # regime: SPY daily above/below 200d MA -> map each bar's calendar date
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                       columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"]); spy_c = spy.set_index("date")["close"].astype(float).sort_index()
    bull = (spy_c > spy_c.rolling(200).mean())
    bull_by_date = {d.date(): bool(v) for d, v in bull.items()}

    sig_keys = [k for k in SIGNALS if k not in SKIP]
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])       # (sk,h,reg,ym) -> [n,sum_dem,sumsq,wins,sum_raw]
    for ni, (tk, df) in enumerate(frames.items()):
        close = df["Close"].values; n = len(close)
        bench = ew_level.reindex(df.index).values
        dts = df.index
        for sk in sig_keys:
            try:
                sig = SIGNALS[sk][1](df).fillna(False)
            except Exception:
                continue
            fires = np.where(sig.values)[0]
            if len(fires) == 0:
                continue
            last = -10 ** 9
            for i in fires:                                # episode dedup
                if i - last < DEDUP_BARS:
                    continue
                last = i
                ep = float(close[i])
                if ep <= PRICE_FLOOR or not np.isfinite(bench[i]) or bench[i] <= 0:
                    continue
                d = dts[i]
                reg = "bull" if bull_by_date.get(d.date(), False) else "bear"
                ym = f"{d.year}-{d.month:02d}"
                for h in HBARS:
                    j = i + h
                    if j >= n or not np.isfinite(bench[j]) or bench[j] <= 0:
                        continue
                    raw = (float(close[j]) - ep) / ep * 100.0
                    dem = _wins(raw - (bench[j] / bench[i] - 1.0) * 100.0)
                    a = acc[(sk, h, reg, ym)]
                    a[0] += 1; a[1] += dem; a[2] += dem * dem; a[3] += 1 if dem > 0 else 0; a[4] += _wins(raw)
        if (ni + 1) % 100 == 0:
            print(f"  {ni+1}/{len(frames)} names processed (keys {len(acc)})", flush=True)

    # finalize -> rows (long/short) with honest monthly-panel t
    grp = defaultdict(dict)
    for (sk, h, reg, ym), v in acc.items():
        grp[(sk, h, reg)][ym] = v
    rows = []
    for (sk, h, reg), months in grp.items():
        tot_n = sum(v[0] for v in months.values())
        if tot_n < 100:
            continue
        tot_sum = sum(v[1] for v in months.values()); tot_w = sum(v[3] for v in months.values()); tot_raw = sum(v[4] for v in months.values())
        mmeans = [v[1] / v[0] for v in months.values() if v[0] > 0]
        mean = tot_sum / tot_n
        t = None
        if len(mmeans) >= 6:
            mu = np.mean(mmeans); sd = np.std(mmeans, ddof=1)
            t = mu / (sd / math.sqrt(len(mmeans))) if sd > 0 else None
        for direction in ("long", "short"):
            s = 1.0 if direction == "long" else -1.0
            rows.append(dict(signal=sk, signal_name=SIGNALS[sk][0], bars=h, regime=reg, direction=direction,
                             n=tot_n, months=len(mmeans), demeaned=round(s * mean, 3),
                             abs_ret=round(s * tot_raw / tot_n, 3),
                             win=round((tot_w if direction == "long" else tot_n - tot_w) / tot_n * 100, 1),
                             t=None if t is None else round(s * t, 2)))
    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), arm="h4", universe=len(frames),
                   horizons_bars=HBARS, rows=rows,
                   caveat="H4/4h intraday arm. SURVIVOR-BIASED: only ~562 liquid large/mid-cap names have 4h data, "
                          "NO delisted -> NOT survivorship-corrected (unlike daily arm). Cross-sectionally demeaned "
                          "vs 4h EW-universe index, winsorized, episode-deduped, monthly-panel t, regime-split, "
                          "LONG+SHORT. Horizons in 4h bars (~2/trading-day). Candidate generator only.")
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/signal_discovery_h4.json").write_text(json.dumps(payload, default=str))
    # leaderboards: POSITIVE-long tradeable (bull) + strongest overall by |demeaned|
    def show(title, subset):
        print(f"\n=== {title} ===", flush=True)
        print(f"  {'signal':26}{'bars':>5}{'reg':>5}{'dir':>6}{'n':>7}{'mo':>4}{'dem%':>7}{'abs%':>7}{'win%':>6}{'t':>7}", flush=True)
        for r in subset:
            print(f"  {r['signal'][:25]:26}{r['bars']:>5}{r['regime']:>5}{r['direction']:>6}{r['n']:>7}{r['months']:>4}"
                  f"{r['demeaned']:>7.2f}{r['abs_ret']:>7.2f}{r['win']:>6.1f}{r['t']:>7.2f}", flush=True)
    poslong = sorted([r for r in rows if r["direction"] == "long" and r["t"] and r["t"] >= 3
                      and r["months"] >= 12 and r["demeaned"] > 0 and r["abs_ret"] > 0], key=lambda r: -r["demeaned"])[:20]
    show("TOP POSITIVE-LONG (tradeable: alpha>0 AND abs>0, t>=3, >=12mo)", poslong)
    strong = sorted([r for r in rows if r["t"] and abs(r["t"]) >= 3 and r["months"] >= 12], key=lambda r: -abs(r["demeaned"]))[:15]
    show("STRONGEST overall by |alpha|", strong)
    if "--db" in sys.argv:
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(kind="signal_discovery_h4",
                defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
            print("\nSaved BacktestResult[signal_discovery_h4]", flush=True)
        except Exception as e:
            print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
