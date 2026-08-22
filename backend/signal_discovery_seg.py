#!/usr/bin/env python3
"""Daily signal discovery SEGMENTED by market-cap bucket AND sector (user HARD RULE — never average across the
universe). Reuses signal_discovery's universe / EW-index / loaders; the worker attaches a cap bucket (PIT
shares_outstanding x price) and a Finviz sector to each event and accumulates per segment. Focused exit set
(2w/4w/12w/6m — the long horizons where a durable edge would live). Finds POSITIVE-LONG edges hidden inside a
specific (cap x sector) cell that the universe average erased. Survivorship-aware, cross-sectionally demeaned,
winsorized, monthly-panel t, bull/bear. LONG+SHORT.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/signal_discovery_seg.py --jobs 8 --db"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

from studies import SIGNALS, EXITS, _episode_starts
from all_on_all_study import _prepare_indicators, _prepare_alt
from seq_fundamental_study import load_candles, load_insider, load_filings, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, _ew_index, _wins, DVOL_FLOOR, PRICE_FLOOR, SKIP

EXIT_SET = ["2w", "4w", "12w", "6m"]


def _sector_map():
    bt = json.load(open("/app/.data/finviz_universe.json"))["by_ticker"]
    return {t: (v.get("sector") or "sec?") for t, v in bt.items()}


def _cap_bucket(mc):
    if mc is None or not np.isfinite(mc) or mc <= 0:
        return "cap?"
    b = mc / 1e9
    return "mega>200" if b >= 200 else "large10-200" if b >= 10 else "mid2-10" if b >= 2 else "small<2"


def _worker(payload):
    signal_keys, exit_keys, tickers, ew_level, spy_c, secmap = payload
    import django as _dj
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        _dj.setup()
    except Exception:
        pass
    candles = load_candles(tickers); insider = load_insider(tickers); filings = load_filings(tickers)
    reps = load_financial_reports(tickers)
    spy_ma = spy_c.rolling(200).mean()
    exit_fns = {ek: EXITS[ek][1] for ek in exit_keys}
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])          # (sk,ek,reg,segtype,segval,ym) -> [n,sumdem,sumsq,wins,sumraw]
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
        dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
        if not np.any((dvol20 >= DVOL_FLOOR) & (close > PRICE_FLOOR)):
            continue
        _prepare_indicators(sdf); _prepare_alt(sdf, insider.get(tk), filings.get(tk))
        dates = sdf.index
        bench_al = ew_level.reindex(dates).values
        bull_al = (spy_c.reindex(dates) > spy_ma.reindex(dates)).values
        # PIT shares -> cap bucket per bar
        shares_al = None
        r = reps.get(tk)
        if r is not None and "shares_outstanding" in r.columns:
            d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
            if not d.empty:
                s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
                s = s[~s.index.duplicated(keep="last")].sort_index()
                shares_al = s.reindex(s.index.union(dates)).ffill().reindex(dates).values
        sector = secmap.get(tk, "sec?")
        exit_cache = {}; _MISS = object()
        for sk in signal_keys:
            try:
                sig = SIGNALS[sk][1](sdf).fillna(False)
            except Exception:
                continue
            idxs = [sdf.index.get_loc(d) for d in sig[sig].index]
            if not idxs:
                continue
            episode = _episode_starts(idxs)
            for idx in idxs:
                if idx not in episode:
                    continue
                ep = float(close[idx])
                if ep <= PRICE_FLOOR or not (dvol20[idx] >= DVOL_FLOOR):
                    continue
                b0 = bench_al[idx]
                if not np.isfinite(b0) or b0 <= 0:
                    continue
                reg = "bull" if bool(bull_al[idx]) else "bear"
                ym = str(dates[idx].year)                          # YEAR granularity: ~12x fewer keys + by-year robustness
                mc = (shares_al[idx] * ep) if (shares_al is not None and np.isfinite(shares_al[idx])) else None
                cap = _cap_bucket(mc)
                for ek, fn in exit_fns.items():
                    ck = (ek, idx); xi = exit_cache.get(ck, _MISS)
                    if xi is _MISS:
                        try:
                            xi = fn(sdf, idx)
                        except Exception:
                            xi = None
                        if xi is None or xi >= n:
                            xi = n - 1
                        exit_cache[ck] = xi
                    if xi <= idx:
                        continue
                    b1 = bench_al[xi]
                    if not np.isfinite(b1) or b1 <= 0:
                        continue
                    raw = (float(close[xi]) - ep) / ep * 100.0
                    dem = _wins(raw - (b1 / b0 - 1.0) * 100.0); rw = _wins(raw)
                    for st, sv in (("cap", cap), ("sec", sector)):
                        a = acc[(sk, ek, reg, st, sv, ym)]
                        a[0] += 1; a[1] += dem; a[2] += dem * dem; a[3] += 1 if dem > 0 else 0; a[4] += rw
    return dict(acc)


def _finalize(acc):
    grp = defaultdict(dict)
    for (sk, ek, reg, st, sv, ym), v in acc.items():
        grp[(sk, ek, reg, st, sv)][ym] = v
    rows = []
    for (sk, ek, reg, st, sv), months in grp.items():
        tot_n = sum(v[0] for v in months.values())
        if tot_n < 100:
            continue
        tot_sum = sum(v[1] for v in months.values()); tot_w = sum(v[3] for v in months.values()); tot_raw = sum(v[4] for v in months.values())
        mm = [v[1] / v[0] for v in months.values() if v[0] > 0]   # per-YEAR means
        mean = tot_sum / tot_n
        t = (np.mean(mm) / (np.std(mm, ddof=1) / math.sqrt(len(mm)))) if len(mm) >= 4 and np.std(mm, ddof=1) > 0 else None
        for direction in ("long", "short"):
            s = 1.0 if direction == "long" else -1.0
            rows.append(dict(signal=sk, exit=ek, regime=reg, seg_type=st, seg=sv, direction=direction,
                             n=tot_n, months=len(mm), demeaned=round(s * mean, 3), abs_ret=round(s * tot_raw / tot_n, 3),
                             win=round((tot_w if direction == "long" else tot_n - tot_w) / tot_n * 100, 1),
                             t=None if t is None else round(s * t, 2)))
    return rows


def run(jobs, limit=None, save_db=True):
    import concurrent.futures as cf, multiprocessing as mp
    universe, delisted = _universe()
    if limit:
        universe = universe[:limit]
    sig_keys = [k for k in SIGNALS if k not in SKIP]
    secmap = _sector_map()
    print(f"universe {len(universe)} | signals {len(sig_keys)} | exits {len(EXIT_SET)} | jobs {jobs} | "
          f"sector-classified {sum(1 for t in universe if t in secmap and secmap[t]!='sec?')}", flush=True)
    ew_level, spy_c = _ew_index(universe)
    print(f"  EW index {len(ew_level)} days", flush=True)
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
    def _mrg(part):
        for k, v in part.items():
            a = acc[k]; a[0]+=v[0]; a[1]+=v[1]; a[2]+=v[2]; a[3]+=v[3]; a[4]+=v[4]
    chunks = _chunk(universe, jobs * 8)
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, (sig_keys, EXIT_SET, ch, ew_level, spy_c, secmap)) for ch in chunks]
        for i, f in enumerate(cf.as_completed(futs)):
            _mrg(f.result())
            print(f"  chunk {i+1}/{len(chunks)} merged (keys {len(acc)})", flush=True)
    rows = _finalize(acc)
    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), arm="daily_segmented", universe=len(universe),
                   exits=EXIT_SET, rows=rows, caveat="Segmented by cap-bucket (PIT shares x price) AND Finviz sector. "
                   "Survivorship-aware, cross-sectionally demeaned vs EW-universe, winsorized, monthly-panel t, bull/bear.")
    Path("/app/.data/studies/signal_discovery_seg.json").write_text(json.dumps(payload, default=str))
    # LEADERBOARD: top POSITIVE-LONG cells (the hidden edges the average erased)
    pos = [r for r in rows if r["direction"] == "long" and r["t"] and r["t"] >= 3 and r["months"] >= 4
           and r["demeaned"] > 0 and r["abs_ret"] > 0 and r["seg"] not in ("cap?", "sec?")]
    pos.sort(key=lambda r: -r["demeaned"])
    print(f"\n=== TOP POSITIVE-LONG CELLS (alpha>0 AND abs>0, yearly-t>=3, >=4yr) — the hidden edges ===", flush=True)
    print(f"  {'signal':22}{'exit':5}{'reg':5}{'seg':16}{'n':>7}{'yr':>4}{'dem%':>7}{'abs%':>7}{'win%':>6}{'t':>6}", flush=True)
    for r in pos[:35]:
        print(f"  {r['signal'][:21]:22}{r['exit']:5}{r['regime']:5}{(r['seg_type']+':'+r['seg'])[:15]:16}"
              f"{r['n']:>7}{r['months']:>4}{r['demeaned']:>7.2f}{r['abs_ret']:>7.2f}{r['win']:>6.1f}{r['t']:>6.2f}", flush=True)
    print(f"\n  total positive-long robust cells: {len(pos)}", flush=True)
    if save_db and rows:
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(kind="signal_discovery_seg",
                defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
            print("Saved BacktestResult[signal_discovery_seg]", flush=True)
        except Exception as e:
            print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    run(jobs, limit=lim, save_db=("--no-db-save" not in sys.argv))
