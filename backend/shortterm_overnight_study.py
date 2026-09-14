#!/usr/bin/env python3
"""SHORT-TERM TRADE HUNT (3 of 4 angles; H4 is a separate run on 4h data).
User wants a short-term trade. The ONE short-term effect we've PROVEN real in our data is the
overnight premium (exec-timing: names earn ~+37bps/night, intraday ~0). Question = does it beat
SPREADS on tradeable names? Test on FULL-history daily OHLC (cached), tiered by PIT liquidity,
cost modeled explicitly. Discipline from this session: no survivorship shortcuts, regime split,
distrust anything that needs recent-only data.

Angles:
 (1) OVERNIGHT HARVEST  — buy close[t], sell open[t+1]. Mean overnight return by LIQUIDITY tier
     (trailing 20d median $vol = PIT) & regime, vs the intraday leg (close[t]/open[t]-1) which
     should be ~0. NET at round-trip costs {5,10,20,40 bps}. Edge survives only where gross > cost.
 (2) OVERNIGHT + SELECTION — overnight return CONDITIONED on a same-day trigger (big down day,
     oversold RSI, gap down): does the pop concentrate? Best liquid tier only.
 (3) DOWN-DAY BOUNCE — after ret[t] < -k*vol, forward close->close 1/3/5-day return by tier/regime.

-> BacktestResult[shortterm_overnight] + JSON.  Cached candles /tmp/beat_candles.pkl.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-celery-worker-1 python -u /app/shortterm_overnight_study.py
"""
import os, json, warnings, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, ta
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from seq_fundamental_study import load_candles

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "shortterm_overnight.json"
CACHE = Path("/tmp/beat_candles.pkl")
# liquidity tiers by trailing 20d median $-volume (typical round-trip spread in bps, rough)
TIERS = [("A >$100M", 100e6, 1e18, 4), ("B $20-100M", 20e6, 100e6, 10),
         ("C $5-20M", 5e6, 20e6, 25), ("D $1-5M", 1e6, 5e6, 60), ("E <$1M", 0, 1e6, 150)]
COSTS_BPS = [5, 10, 20, 40]


def robust(fn, tries=6):
    from django.db import connection
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception:
            connection.close(); time.sleep(4 * (a + 1))
    return None


class Acc:
    __slots__ = ("s", "ss", "n", "npos")
    def __init__(self): self.s = self.ss = 0.0; self.n = 0; self.npos = 0
    def add(self, x):
        self.s += x; self.ss += x * x; self.n += 1; self.npos += (x > 0)
    def stat(self):
        if self.n == 0: return {"n": 0}
        m = self.s / self.n; var = max(self.ss / self.n - m * m, 0.0)
        t = m / (np.sqrt(var / self.n)) if var > 0 and self.n > 1 else None
        return {"n": self.n, "mean_bps": round(m * 1e4, 2), "hit%": round(self.npos / self.n * 100, 1),
                "t": round(float(t), 1) if t is not None else None}


def main():
    print("loading cache...", flush=True)
    candles = pickle.loads(CACHE.read_bytes())
    spy = robust(lambda: load_candles(["SPY"]))["SPY"]["Close"]
    spy_bull = (spy > spy.rolling(200).mean())
    print(f"  {len(candles)} tickers", flush=True)

    def tier_of(dv):
        for name, lo, hi, _ in TIERS:
            if lo <= dv < hi:
                return name
        return None

    # accumulators
    ov = {t[0]: {"bull": Acc(), "bear": Acc(), "all": Acc()} for t in TIERS}     # overnight
    intr = {t[0]: Acc() for t in TIERS}                                          # intraday leg
    cc = {t[0]: Acc() for t in TIERS}                                            # full close-close (context)
    # (2) overnight conditioned on same-day trigger, best liquid tiers (A+B)
    trig = {k: Acc() for k in ("downday_1v", "downday_2v", "oversold35", "gap_down2", "up_day")}
    trig_uncond = Acc()
    # (3) down-day bounce forward close-close
    bounce = {h: {"A+B": Acc(), "C": Acc()} for h in (1, 3, 5)}

    for tk, df in candles.items():
        if df is None or len(df) < 60:
            continue
        o = df["Open"].values.astype(float); c = df["Close"].values.astype(float)
        v = df["Volume"].values.astype(float); idx = df.index
        n = len(c)
        dvol = (df["Close"] * df["Volume"]).rolling(20).median().values
        rsi = ta.momentum.rsi(df["Close"], 14).values
        ret = np.empty(n); ret[0] = 0.0; ret[1:] = c[1:] / c[:-1] - 1
        vol20 = pd.Series(ret).rolling(20).std().values
        for t in range(20, n - 5):
            if not (np.isfinite(dvol[t]) and dvol[t] > 0 and c[t] > 5 and o[t + 1] > 0 and o[t] > 0):
                continue
            tier = tier_of(dvol[t])
            if tier is None:
                continue
            overnight = o[t + 1] / c[t] - 1.0
            intraday = c[t] / o[t] - 1.0
            if abs(overnight) > 0.5 or abs(intraday) > 0.5:      # split/bad-data guard
                continue
            reg = "bull" if bool(spy_bull.asof(idx[t])) else "bear"
            ov[tier]["all"].add(overnight); ov[tier][reg].add(overnight)
            intr[tier].add(intraday)
            cc[tier].add(c[t + 1] / c[t] - 1.0)
            # (2) triggers — measured on the liquid tiers where a trade is possible
            if tier in ("A >$100M", "B $20-100M"):
                trig_uncond.add(overnight)
                z = ret[t] / vol20[t] if np.isfinite(vol20[t]) and vol20[t] > 0 else 0.0
                if z < -1: trig["downday_1v"].add(overnight)
                if z < -2: trig["downday_2v"].add(overnight)
                if np.isfinite(rsi[t]) and rsi[t] < 35: trig["oversold35"].add(overnight)
                if o[t] / c[t - 1] - 1 < -0.02: trig["gap_down2"].add(overnight)
                if z > 1: trig["up_day"].add(overnight)
            # (3) down-day bounce: big down day -> forward close-close
            z = ret[t] / vol20[t] if np.isfinite(vol20[t]) and vol20[t] > 0 else 0.0
            if z < -2:
                grp = "A+B" if tier in ("A >$100M", "B $20-100M") else ("C" if tier == "C $5-20M" else None)
                if grp:
                    for h in (1, 3, 5):
                        if t + h < n:
                            bounce[h][grp].add(c[t + h] / c[t] - 1.0)

    # ---- report ----
    print("\n=== (1) OVERNIGHT vs INTRADAY by liquidity tier (full history) ===", flush=True)
    print(f"{'tier':12} {'spread~':>7} | {'overnight n':>11} {'mean_bps':>9} {'hit%':>6} {'t':>6} | "
          f"{'intraday_bps':>12} | net@cost(bps): " + " ".join(f"{c}c" for c in COSTS_BPS), flush=True)
    tier_res = {}
    for name, lo, hi, spread in TIERS:
        og = ov[name]["all"].stat(); ig = intr[name].stat()
        tier_res[name] = {"overnight": og, "intraday": ig, "assumed_spread_bps": spread,
                          "overnight_bull": ov[name]["bull"].stat(), "overnight_bear": ov[name]["bear"].stat()}
        if og.get("n", 0) == 0:
            continue
        nets = " ".join(f"{round(og['mean_bps'] - c, 1):>6}" for c in COSTS_BPS)
        print(f"{name:12} {spread:>6}b | {og['n']:>11} {og['mean_bps']:>9} {og['hit%']:>6} {str(og['t']):>6} | "
              f"{ig['mean_bps']:>12} | {nets}", flush=True)

    print("\n=== (2) OVERNIGHT conditioned on same-day trigger (liquid A+B only) ===", flush=True)
    base = trig_uncond.stat()
    print(f"  unconditioned    n{base['n']:>7}  {base['mean_bps']:>7} bps  hit {base['hit%']}%  t{base['t']}", flush=True)
    trig_res = {"unconditioned": base}
    for k, a in trig.items():
        s = a.stat(); trig_res[k] = s
        if s.get("n", 0):
            print(f"  {k:16} n{s['n']:>7}  {s['mean_bps']:>7} bps  hit {s['hit%']}%  t{s['t']}"
                  f"   (lift {round(s['mean_bps']-base['mean_bps'],1)} bps)", flush=True)

    print("\n=== (3) DOWN-DAY BOUNCE (ret< -2*vol) fwd close-close ===", flush=True)
    bounce_res = {}
    for h in (1, 3, 5):
        for grp in ("A+B", "C"):
            s = bounce[h][grp].stat(); bounce_res[f"{grp}_{h}d"] = s
            if s.get("n", 0):
                print(f"  {grp:4} +{h}d  n{s['n']:>6}  {s['mean_bps']:>8} bps  hit {s['hit%']}%  t{s['t']}", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "tiers": tier_res, "triggers": trig_res, "bounce": bounce_res,
               "costs_bps": COSTS_BPS,
               "caveat": "Full-history daily OHLC (cached). Overnight = open[t+1]/close[t]-1; adjacent-day so "
                         "adj-factor cancels; |ret|>50% filtered. Liquidity tier = trailing 20d median $vol (PIT). "
                         "Assumed spreads are rough; NET columns show gross-minus-flat-cost. Universe = currently-"
                         "listed tickers (residual delisted survivorship). Per-night means; a real book holds a "
                         "diversified basket each night so the tier mean ~ the book's gross overnight return."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="shortterm_overnight",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[shortterm_overnight]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
