#!/usr/bin/env python3
"""Vol-normalized SHOCK continuation study -> .data/studies/vol_shock_study.json + Postgres.

Question: after a day whose return is large RELATIVE to the stock's own volatility (z = daily
return / trailing-20d realized vol, vol shifted so today doesn't deflate its own z), does the
stock CONTINUE in that direction, or REVERSE? Entry at the shock close; forward returns vs the
stock's own baseline drift. No lookahead. Last-5y DB candles.

Three payload blocks (rendered on the "Vol-Shock" dashboard tab):
  continuation - episode-deduped (independent-episode) mean forward return + continuation rate +
                 edge-vs-baseline + honest t, per shock threshold x horizon, for up and down shocks.
  slices       - 2-sigma shocks bucketed by market-cap, volume-confirmation, SPY 50/200 regime,
                 and sector (where does continuation vs reversal actually hold?).
  backtest     - the registered vol_shock_* signals x a representative EXIT ladder (episode-deduped)
                 -> which hold/risk rule the move actually pays. NO fees; survivorship-caveated.

Run in the backend container:
  docker exec rotation-backend-1 python -u /app/vol_shock_study.py
Options: --limit N (cap universe for a light test).
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
from studies import SIGNALS, EXITS, _episode_starts, _tstat_from_returns
from seq_fundamental_study import load_candles, _compute_trades

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "vol_shock_study.json"

WIN = 20
THRS = [1.5, 2.0, 3.0]
HZ = [1, 3, 5, 10, 20]
MIN_BARS = WIN + 40
SLICE_THR = 2.0
SLICE_HZ = [5, 10]
BT_SIGNALS = ["vol_shock_up", "vol_shock_dn", "vol_shock_dn3",
              "vol_shock_up_hivol", "vol_shock_dn_hivol", "vol_shock_dn3_hivol"]
EXIT_KEYS = ["1d", "3d", "1w", "2w", "4w", "8w", "12w",
             "trail_5", "trail_10", "sl_5", "sl_10", "tp_5", "tp_10", "tp_15"]


def cap_bucket(mc):
    if mc is None or mc <= 0: return "unknown"
    if mc < 500e6: return "micro (<500M)"
    if mc < 2e9:   return "small (0.5-2B)"
    if mc < 10e9:  return "mid (2-10B)"
    return "large (>10B)"


class Acc:
    __slots__ = ("n", "s", "ss", "w")
    def __init__(self): self.n = self.s = self.ss = self.w = 0
    def add(self, arr, wins):
        arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
        self.n += arr.size; self.s += float(arr.sum()); self.ss += float((arr * arr).sum())
        self.w += int(np.asarray(wins).sum())
    def add1(self, v, win):
        if v is None or not np.isfinite(v): return
        self.n += 1; self.s += v; self.ss += v * v; self.w += int(win)
    def stats(self):
        if not self.n: return dict(n=0, mean=None, wr=None, t=None)
        mean = self.s / self.n; wr = 100 * self.w / self.n; t = None
        if self.n >= 3:
            var = (self.ss - self.s * self.s / self.n) / (self.n - 1)
            if var > 0: t = round(float(mean / (np.sqrt(var) / np.sqrt(self.n))), 1)
        return dict(n=self.n, mean=round(100 * mean, 3), wr=round(wr, 1), t=t)


def build(limit=None):
    from core.models import Candle, Sector, Fundamental
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    universe = sorted(set(Candle.objects.values_list("ticker", flat=True).distinct()) - etfs)
    if limit:
        universe = universe[:limit]

    fmap = {}
    for tk, mc, sec in Fundamental.objects.values_list("ticker", "market_cap", "sector"):
        if tk not in fmap: fmap[tk] = (mc, sec or "?")

    spy = load_candles(["SPY"]).get("SPY")
    bull_map = {}
    if spy is not None:
        spc = spy["Close"]
        bull_map = {d: bool(v) for d, v in (spc.rolling(50).mean() > spc.rolling(200).mean()).items()}

    candles = load_candles(universe)   # load once; reused by the backtest below
    print(f"Universe {len(universe)}, loaded {len(candles)} with candles, win={WIN}d", flush=True)

    base = {H: Acc() for H in HZ}
    cond = {(thr, H, d): Acc() for thr in THRS for H in HZ for d in ("up", "dn")}
    cond_hv = {(thr, H, d): Acc() for thr in THRS for H in HZ for d in ("up", "dn")}  # volume-confirmed
    slc = {}
    def sadd(kind, d, H, bucket, v):
        slc.setdefault((kind, d, H, bucket), Acc()).add1(v, (v > 0) if d == "up" else (v < 0))

    n_names = 0
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS: continue
        n_names += 1
        close = sdf["Close"].values.astype(float)
        volu = sdf["Volume"].values.astype(float)
        dates = sdf.index
        ret = pd.Series(close).pct_change()
        vstd = ret.rolling(WIN).std().shift(1)
        z = (ret / vstd).values; vv = vstd.values
        avgvol = pd.Series(volu).rolling(WIN).mean().shift(1).values
        hv_arr = (avgvol > 0) & (volu > 1.5 * avgvol)   # volume-confirmed (>1.5x trailing avg)
        mc, sec = fmap.get(tk, (None, "?")); capb = cap_bucket(mc)
        for H in HZ:
            fwd = np.full(len(close), np.nan); fwd[:-H] = close[H:] / close[:-H] - 1.0
            ok = np.isfinite(z) & np.isfinite(fwd) & (vv > 0)
            base[H].add(fwd[ok], fwd[ok] > 0)
            for thr in THRS:
                for d, mask in (("up", z >= thr), ("dn", z <= -thr)):
                    idxs = np.where(mask & ok)[0]
                    if idxs.size == 0: continue
                    ep = sorted(_episode_starts(idxs.tolist(), gap=H))
                    f = fwd[ep]
                    cond[(thr, H, d)].add(f, f > 0 if d == "up" else f < 0)
                    # volume-confirmed continuation: episodes among hi-vol shocks only
                    idxs_hv = np.where(mask & ok & hv_arr)[0]
                    if idxs_hv.size:
                        ep_hv = sorted(_episode_starts(idxs_hv.tolist(), gap=H))
                        fhv = fwd[ep_hv]
                        cond_hv[(thr, H, d)].add(fhv, fhv > 0 if d == "up" else fhv < 0)
                    if thr == SLICE_THR and H in SLICE_HZ:
                        for j in ep:
                            fv = fwd[j]
                            sadd("cap", d, H, capb, fv)
                            sadd("sector", d, H, sec, fv)
                            vc = "hi-vol (>1.5x)" if (avgvol[j] > 0 and volu[j] > 1.5 * avgvol[j]) else "normal-vol"
                            sadd("vol", d, H, vc, fv)
                            sadd("regime", d, H, ("SPY bull" if bull_map.get(dates[j], False) else "SPY bear"), fv)

    baseline = {H: base[H].stats()["mean"] for H in HZ}

    # continuation rows (shared builder for the all-shocks and volume-confirmed matrices)
    def _cont_rows(condmap):
        out = {"up": [], "dn": []}
        for d in ("up", "dn"):
            for thr in THRS:
                for H in HZ:
                    st = condmap[(thr, H, d)].stats()
                    if not st["n"]: continue
                    b = baseline[H]
                    edge = round((st["mean"] - b) if d == "up" else (b - st["mean"]), 3)
                    cont = st["wr"] if d == "up" else round(100 - st["wr"], 1)
                    out[d].append({"thr": thr, "H": H, "episodes": st["n"],
                                   "mean_pct": st["mean"], "cont_pct": cont,
                                   "edge_pct": edge, "t": st["t"]})
        return out
    continuation = _cont_rows(cond)
    continuation_hivol = _cont_rows(cond_hv)

    # slices
    slices = {}
    for kind in ("cap", "vol", "regime", "sector"):
        slices[kind] = {"up": {}, "dn": {}}
        for d in ("up", "dn"):
            for H in SLICE_HZ:
                items = [{"bucket": b, **a.stats()}
                         for (k, dd, HH, b), a in slc.items() if k == kind and dd == d and HH == H]
                items = [x for x in items if x["n"] >= 30]
                items.sort(key=lambda x: -(x["mean"] if x["mean"] is not None else -1e9))
                slices[kind][d][str(H)] = items

    # backtest ladder
    backtest = {}
    for sk in BT_SIGNALS:
        _, sig_fn = SIGNALS[sk]
        rows = []
        for ek in EXIT_KEYS:
            _, exit_fn = EXITS[ek]
            tr = _compute_trades(candles, sig_fn, exit_fn)
            by_tk = {}
            for tk, idx, r in tr: by_tk.setdefault(tk, []).append((idx, r))
            rets = []
            for tk, lst in by_tk.items():
                lst.sort(); keep = _episode_starts([i for i, _ in lst], gap=5)
                rets += [r for i, r in lst if i in keep]
            if len(rets) < 20: continue
            a = np.array(rets)
            rows.append({"exit": ek, "name": EXITS[ek][0], "trades": len(rets),
                         "avg_pct": round(float(a.mean()), 3), "win_pct": round(float((a > 0).mean() * 100), 1),
                         "t": _tstat_from_returns(list(rets))})
        rows.sort(key=lambda x: -x["avg_pct"])
        backtest[sk] = {"name": SIGNALS[sk][0], "rows": rows}
        print(f"  backtested {sk}: {len(rows)} exits", flush=True)

    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"vol_window": WIN, "thresholds": THRS, "horizons": HZ,
                   "slice_thr": SLICE_THR, "slice_hz": SLICE_HZ,
                   "universe": {"names": n_names}},
        "baseline": baseline,
        "continuation": continuation,
        "continuation_hivol": continuation_hivol,
        "slices": slices,
        "backtest": backtest,
        "note": ("Entry at the shock close (no lookahead). Episode-deduped (independent episodes, "
                 "gap=horizon) so t-stats aren't inflated by clustered fires. Backtest is NO-fee and "
                 "the universe is today's listed names (survivorship — inflates the down-shock bounce, "
                 "esp. micro-caps). Read as directional."),
    }


def _save_db(payload):
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="vol_shock_study",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[vol_shock_study] to DB", flush=True)
    except Exception as e:
        print(f"DB save failed for vol_shock_study: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap universe for a light test")
    args = ap.parse_args()
    payload = build(limit=args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _save_db(payload)
    print("\n=== VOL-SHOCK CONTINUATION (episode-deduped) ===", flush=True)
    print("baseline fwd:", {f"{H}d": payload["baseline"][H] for H in HZ}, flush=True)
    for d in ("up", "dn"):
        print(f"\n{d.upper()}-shock:")
        for r in payload["continuation"][d]:
            print(f"  thr{r['thr']} H{r['H']:>2}  n={r['episodes']:>6}  mean {r['mean_pct']:>+7.2f}%  "
                  f"cont {r['cont_pct']:>5}%  edge {r['edge_pct']:>+6.2f}%  t={r['t']}", flush=True)
    print("\n=== BACKTEST (top exit per signal) ===", flush=True)
    for sk, blk in payload["backtest"].items():
        top = blk["rows"][0] if blk["rows"] else None
        if top:
            print(f"  {sk:22} best={top['exit']:>7} avg {top['avg_pct']:>+6.2f}% "
                  f"win {top['win_pct']:>5}% t={top['t']}", flush=True)
    print("\nSaved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
