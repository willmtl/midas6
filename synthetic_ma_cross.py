#!/usr/bin/env python3
"""MOVING-AVERAGE CROSSOVER ON EVERY SYNTHETIC RS CANDLE.

For each of the 93 sectors, build the synthetic relative-strength candle (Close = etf/spy daily) and run
classic MA crossovers ON THAT BAR. A golden cross on the RS bar means the sector is starting to
OUTPERFORM SPY (not just go up). We measure, pooled across all 93 synthetic candles:
  - forward RELATIVE return after each cross-up / cross-down (21/63/126 trading days) — the signal's edge,
  - hit rate (% of events where the sector then out/under-performed) and a t-stat,
  - the state test: avg daily relative return WHILE in a golden cross vs while in a death cross,
  - a live snapshot: which sectors are in a golden cross right now + days since the last cross.

RS forward return r[t+h]/r[t]-1 == the ETF's return MINUS SPY's over the window, so >0 = beat SPY.
Directional signal-predictiveness study (uses forward windows by construction); no fees/slippage.
-> BacktestResult[synthetic_ma_cross] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/synthetic_ma_cross.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np, pandas as pd
import config
from seq_fundamental_study import load_candles

BENCH = getattr(config, "BENCHMARK", "SPY")
PAIRS = [(20, 50), (50, 200), (10, 30)]        # fast/slow SMA windows (trading days)
HORIZONS = [21, 63, 126]                        # ~1 / 3 / 6 months forward


def _tstat(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) < 3 or a.std(ddof=1) == 0:
        return None
    return round(float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))), 2)


def _agg(vals):
    a = np.asarray(vals, float); a = a[~np.isnan(a)]
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean_pct": round(float(a.mean()) * 100, 2),
            "pos_pct": round(float((a > 0).mean()) * 100, 1), "t": _tstat(a)}


def build():
    etfs = [e for e in config.SECTOR_ETFS.values() if e]
    data = load_candles(etfs + [BENCH])
    spy = data.get(BENCH)
    if spy is None:
        raise RuntimeError(f"no benchmark {BENCH}")
    spy_close = spy["Close"]
    name_by_etf = {e: n for n, e in config.SECTOR_ETFS.items()}

    # pooled event buckets: pool[pair][dir][horizon] -> list of forward RS returns
    pool = {p: {"up": {h: [] for h in HORIZONS}, "dn": {h: [] for h in HORIZONS}} for p in PAIRS}
    pool_abs = {p: {"up": {h: [] for h in HORIZONS}} for p in PAIRS}   # forward ABSOLUTE etf return after golden cross
    state_ret = {p: {"bull": [], "bear": []} for p in PAIRS}           # daily RS log-ret split by MA state
    snapshot = {p: [] for p in PAIRS}                                  # current per-sector state
    n_candles = 0

    for etf in etfs:
        df = data.get(etf)
        if df is None or len(df) < 260:
            continue
        rs = (df["Close"] / spy_close.reindex(df.index)).dropna()
        if len(rs) < 260:
            continue
        n_candles += 1
        rs_logret = np.log(rs / rs.shift(1))
        etf_close = df["Close"].reindex(rs.index)
        for (fast, slow) in PAIRS:
            f, s = rs.rolling(fast).mean(), rs.rolling(slow).mean()
            bull = f > s
            cross_up = bull & ~bull.shift(1).fillna(False)
            cross_dn = (~bull) & bull.shift(1).fillna(False)
            for h in HORIZONS:
                fwd_rs = rs.shift(-h) / rs - 1                 # relative fwd return (etf minus spy)
                fwd_abs = etf_close.shift(-h) / etf_close - 1  # absolute etf fwd return
                pool[(fast, slow)]["up"][h] += list(fwd_rs[cross_up].dropna().values)
                pool[(fast, slow)]["dn"][h] += list(fwd_rs[cross_dn].dropna().values)
                pool_abs[(fast, slow)]["up"][h] += list(fwd_abs[cross_up].dropna().values)
            state_ret[(fast, slow)]["bull"] += list(rs_logret[bull.shift(1).fillna(False)].dropna().values)
            state_ret[(fast, slow)]["bear"] += list(rs_logret[(~bull).shift(1).fillna(False)].dropna().values)
            # live snapshot
            cur_bull = bool(bull.iloc[-1])
            flip = bull != bull.shift(1)
            last_cross_idx = flip[flip].index[-1] if flip.any() else None
            days_since = int((rs.index[-1] - last_cross_idx).days) if last_cross_idx is not None else None
            snapshot[(fast, slow)].append({
                "sector": name_by_etf.get(etf, etf), "etf": etf,
                "state": "golden" if cur_bull else "death",
                "days_since_cross": days_since,
                "rs_vs_50d_high": round(float(rs.iloc[-1] / rs.rolling(50).max().iloc[-1]), 3),
            })

    pairs_out = []
    for (fast, slow) in PAIRS:
        row = {"pair": f"{fast}/{slow}",
               "cross_up": {f"{h}d": _agg(pool[(fast, slow)]["up"][h]) for h in HORIZONS},
               "cross_dn": {f"{h}d": _agg(pool[(fast, slow)]["dn"][h]) for h in HORIZONS},
               "abs_after_golden": {f"{h}d": _agg(pool_abs[(fast, slow)]["up"][h]) for h in HORIZONS}}
        b = np.asarray(state_ret[(fast, slow)]["bull"], float); b = b[~np.isnan(b)]
        e = np.asarray(state_ret[(fast, slow)]["bear"], float); e = e[~np.isnan(e)]
        # annualized mean relative drift while in each state (252 trading days)
        row["state_test"] = {
            "golden_ann_rs_pct": round(float(b.mean()) * 252 * 100, 2) if len(b) else None,
            "death_ann_rs_pct": round(float(e.mean()) * 252 * 100, 2) if len(e) else None,
            "golden_days": int(len(b)), "death_days": int(len(e))}
        snap = snapshot[(fast, slow)]
        row["now_golden"] = sorted([s for s in snap if s["state"] == "golden"],
                                   key=lambda x: (x["days_since_cross"] is None, x["days_since_cross"] or 0))
        row["now_death_n"] = sum(1 for s in snap if s["state"] == "death")
        pairs_out.append(row)

    return {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"n_synthetic_candles": n_candles, "benchmark": BENCH,
                   "ma_pairs": [f"{f}/{s}" for f, s in PAIRS], "horizons_days": HORIZONS,
                   "signal": "SMA(fast) crosses SMA(slow) on the synthetic etf/spy (relative-strength) candle"},
        "pairs": pairs_out,
        "note": ("Forward relative return r[t+h]/r[t]-1 = ETF return minus SPY over the window (>0 = beat SPY). "
                 "Signal-predictiveness study across all synthetic candles; forward windows by construction, "
                 "directional, no fees. A golden cross on the RS bar = sector starting to OUTPERFORM SPY."),
    }


def main():
    from pathlib import Path
    payload = build()
    out = Path(__file__).resolve().parent / ".data" / "studies" / "synthetic_ma_cross.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="synthetic_ma_cross",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[synthetic_ma_cross]", flush=True)
    except Exception as ex:
        print("DB save failed:", ex, flush=True)

    print(f"\n=== MA CROSSOVER ON {payload['params']['n_synthetic_candles']} SYNTHETIC RS CANDLES ===", flush=True)
    for row in payload["pairs"]:
        print(f"\n--- SMA {row['pair']} on the RS bar ---", flush=True)
        st = row["state_test"]
        print(f"  state: while GOLDEN, sector out-drifts SPY {st['golden_ann_rs_pct']:+}%/yr "
              f"({st['golden_days']}d);  while DEATH {st['death_ann_rs_pct']:+}%/yr ({st['death_days']}d)", flush=True)
        for h in HORIZONS:
            u = row["cross_up"][f"{h}d"]; d = row["cross_dn"][f"{h}d"]; a = row["abs_after_golden"][f"{h}d"]
            print(f"  +{h:>3}d  golden-cross: rel {u.get('mean_pct'):>+6}%  {u.get('pos_pct')}%pos  t{u.get('t')}  "
                  f"(n{u.get('n')}) | abs-etf {a.get('mean_pct'):>+6}%  ||  death-cross rel {d.get('mean_pct'):>+6}%  "
                  f"{d.get('pos_pct')}%pos  t{d.get('t')} (n{d.get('n')})", flush=True)
        gc = row["now_golden"]
        print(f"  NOW: {len(gc)} sectors in a golden cross, {row['now_death_n']} in a death cross. "
              f"Freshest golden: " + ", ".join(f"{g['sector']}({g['days_since_cross']}d)" for g in gc[:8]), flush=True)


if __name__ == "__main__":
    main()
