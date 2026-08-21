#!/usr/bin/env python3
"""SIGNAL DISCOVERY LAB — every one of the 381 studies.SIGNALS tested as its OWN standalone strategy
(buy names firing it; no flagship gating), across timeframes/exits, LONG and SHORT, under the integrity
harness that killed this session's mirages: survivorship-aware, split-clean, market-DEMEANED, winsorized,
episode-deduped, honest MONTHLY-panel t-stats, bull/bear regime split. Ranks by absolute demeaned return.

Daily arm (this module): full survivorship-aware universe (covered ∪ delisted-with-candles), exits spanning
1d→6m plus trailing/TP/rule exits. Fixes all_on_all's survivorship leak (it drops trades whose exit falls
past a delisted name's last bar; here they exit AT the last bar = the real delisting outcome).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/signal_discovery.py --jobs 16 --db
Smoke: ... --signals rsi_oversold30,gap_down_large,dd25_cap --limit 400 --no-db-save"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

import studies
from studies import SIGNALS, EXITS, MARKET_SIGNAL_KEYS, _episode_starts
from all_on_all_study import _prepare_indicators, _prepare_alt
from seq_fundamental_study import load_candles, load_insider, load_filings, MIN_BARS, _chunk

DVOL_FLOOR = 5e6
PRICE_FLOOR = 5.0
WINS = (-100.0, 150.0)                    # winsorize demeaned % return
SKIP = set(MARKET_SIGNAL_KEYS) | {"rsi_x_pos_updn"}
EXIT_SET = ["1d", "3d", "1w", "2w", "4w", "8w", "12w", "6m",     # fixed horizons ~1d→6mo
            "trail_5", "trail_10", "tp_10", "tp_20", "rsi_x_dn"]  # rule-based


def _universe():
    """Survivorship-aware: analyst-covered US names ∪ delisted-with-candles, minus ETFs. Avoids the
    Candle-hypertable DISTINCT trap (built from the ratings file + DelistedCompany)."""
    from core.models import Sector, DelistedCompany
    etfs = set(Sector.objects.values_list("etf", flat=True)) | {"SPY", "QQQ"}
    covered = set()
    for line in Path("/app/.data/analyst_ratings.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            tk = json.loads(line).get("ticker")
        except Exception:
            tk = None
        if tk and "." not in tk:
            covered.add(tk)
    delisted = set(DelistedCompany.objects.exclude(delisted_date=None).values_list("ticker", flat=True))
    return sorted((covered | delisted) - etfs), delisted


def _wins(x):
    return float(min(max(x, WINS[0]), WINS[1]))


def _ew_index(universe):
    """Equal-weight UNIVERSE daily total-return index (the cross-sectional benchmark): mean simple daily
    return across all names each date, cumulated. Demeaning events by THIS (not SPY) isolates the signal's
    edge vs the average stock, stripping the small/mid-cap-vs-SPY size-beta drag. Streaming per-date aggregate
    (memory-light). Also returns SPY (for the regime gate)."""
    from core.models import Candle
    sums = defaultdict(float); cnts = defaultdict(int)
    for ch in _chunk(universe, max(1, len(universe) // 400 + 1)):
        rows = Candle.objects.filter(ticker__in=ch, date__gte="2014-06-01").values_list("ticker", "date", "close")
        df = pd.DataFrame(list(rows), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for _tk, g in df.groupby("ticker", sort=False):
            r = g.sort_values("date").set_index("date")["close"].pct_change()
            r = r[(r > -0.9) & (r < 2.0)]                          # drop garbage-bar daily returns
            for d, val in r.items():
                sums[d] += val; cnts[d] += 1
    idx = sorted(sums)
    ew_ret = pd.Series([sums[d] / cnts[d] if cnts[d] >= 20 else 0.0 for d in idx], index=pd.DatetimeIndex(idx))
    ew_level = (1.0 + ew_ret).cumprod()
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY", date__gte="2014-06-01").values_list("date", "close")),
                       columns=["date", "close"])
    spy["date"] = pd.to_datetime(spy["date"]); spy_c = spy.set_index("date")["close"].astype(float).sort_index()
    return ew_level, spy_c


def _worker(payload):
    signal_keys, exit_keys, tickers, ew_level, spy_c = payload
    import django as _dj
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        _dj.setup()
    except Exception:
        pass
    candles = load_candles(tickers)
    insider = load_insider(tickers)
    filings = load_filings(tickers)
    spy_ma = spy_c.rolling(200).mean()
    exit_fns = {ek: EXITS[ek][1] for ek in exit_keys}
    # acc[(sk, ek, regime, ym)] = [n, sum_dem, sumsq_dem, wins_dem, sum_raw]
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_indicators(sdf); _prepare_alt(sdf, insider.get(tk), filings.get(tk))
        close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
        dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
        dates = sdf.index
        bench_al = ew_level.reindex(dates).values                 # EW-universe index aligned to this ticker's dates
        bull_al = (spy_c.reindex(dates) > spy_ma.reindex(dates)).values
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
                if idx not in episode:                            # overlap-dedup: independent episodes only
                    continue
                ep = float(close[idx])
                if ep <= PRICE_FLOOR or not (dvol20[idx] >= DVOL_FLOOR):
                    continue
                b0 = bench_al[idx]
                if not np.isfinite(b0) or b0 <= 0:
                    continue
                reg = "bull" if bool(bull_al[idx]) else "bear"
                ym = f"{dates[idx].year}-{dates[idx].month:02d}"
                for ek, fn in exit_fns.items():
                    try:
                        xi = fn(sdf, idx)
                    except Exception:
                        xi = None
                    if xi is None or xi >= n:
                        xi = n - 1                                # SURVIVORSHIP: exit at last bar (delisting outcome)
                    if xi <= idx:
                        continue
                    b1 = bench_al[xi]
                    if not np.isfinite(b1) or b1 <= 0:
                        continue
                    raw = (float(close[xi]) - ep) / ep * 100.0
                    dem = _wins(raw - (b1 / b0 - 1.0) * 100.0)    # cross-sectionally demeaned (vs EW universe), winsorized
                    a = acc[(sk, ek, reg, ym)]
                    a[0] += 1; a[1] += dem; a[2] += dem * dem; a[3] += 1 if dem > 0 else 0; a[4] += _wins(raw)
    return dict(acc)


def _finalize(acc):
    """acc[(sk,ek,reg,ym)] -> per (sk,ek,reg): pooled mean/win + honest monthly-panel t. LONG & SHORT rows."""
    grp = defaultdict(dict)                                        # (sk,ek,reg) -> {ym: [n,sum,sumsq,wins,sum_raw]}
    for (sk, ek, reg, ym), v in acc.items():
        grp[(sk, ek, reg)][ym] = v
    rows = []
    for (sk, ek, reg), months in grp.items():
        tot_n = sum(v[0] for v in months.values())
        if tot_n < 100:                                           # min events for a real cell
            continue
        tot_sum = sum(v[1] for v in months.values()); tot_w = sum(v[3] for v in months.values())
        tot_raw = sum(v[4] for v in months.values())
        mmeans = [v[1] / v[0] for v in months.values() if v[0] > 0]
        mean = tot_sum / tot_n
        t = None
        if len(mmeans) >= 6:
            mu = np.mean(mmeans); sd = np.std(mmeans, ddof=1)
            t = mu / (sd / math.sqrt(len(mmeans))) if sd > 0 else None
        for direction in ("long", "short"):
            s = 1.0 if direction == "long" else -1.0
            rows.append(dict(signal=sk, signal_name=SIGNALS[sk][0], exit=ek, regime=reg, direction=direction,
                             n=tot_n, months=len(mmeans),
                             demeaned=round(s * mean, 3),               # vs EW-universe (cross-sectional alpha)
                             abs_ret=round(s * tot_raw / tot_n, 3),     # raw absolute return (tradeable?)
                             win=round((tot_w if direction == "long" else tot_n - tot_w) / tot_n * 100, 1),
                             t=None if t is None else round(s * t, 2)))
    return rows


def run(jobs, limit=None, signal_keys=None, save_db=True):
    import concurrent.futures as cf, multiprocessing as mp
    universe, delisted = _universe()
    if limit:
        universe = universe[:limit]
    sig_keys = [k for k in (signal_keys or list(SIGNALS.keys())) if k not in SKIP]
    print(f"universe {len(universe)} ({len(delisted)} delisted) | signals {len(sig_keys)} | exits {len(EXIT_SET)} "
          f"| combos {len(sig_keys)*len(EXIT_SET)} | jobs {jobs}", flush=True)
    print("building EW-universe benchmark index...", flush=True)
    ew_level, spy_c = _ew_index(universe)
    print(f"  EW index {len(ew_level)} days {ew_level.index[0].date()}..{ew_level.index[-1].date()}", flush=True)
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])

    def _mrg(part):
        for k, v in part.items():
            a = acc[k]; a[0]+=v[0]; a[1]+=v[1]; a[2]+=v[2]; a[3]+=v[3]; a[4]+=v[4]
    if jobs <= 1:
        _mrg(_worker((sig_keys, EXIT_SET, universe, ew_level, spy_c)))
    else:
        chunks = _chunk(universe, jobs * 3)
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, (sig_keys, EXIT_SET, ch, ew_level, spy_c)) for ch in chunks]
            for i, f in enumerate(cf.as_completed(futs)):
                _mrg(f.result())
                print(f"  chunk {i+1}/{len(chunks)} merged (keys {len(acc)})", flush=True)
    rows = _finalize(acc)
    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), arm="daily", universe=len(universe),
                   delisted=len(delisted), n_signals=len(sig_keys), exits=EXIT_SET, rows=rows,
                   caveat="Standalone per-signal, survivorship-aware (exit-at-last-bar through delisting), "
                          "split-adjusted, market-demeaned, winsorized [-100,150], episode-deduped, monthly-panel t, "
                          "regime-split, LONG+SHORT. CANDIDATE GENERATOR — multiple-testing across 381x13x2; every "
                          "top hit needs its own validation before trust.")
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/signal_discovery.json").write_text(json.dumps(payload, default=str))
    # leaderboard preview: bull-regime long+short, ranked by |demeaned| among robust cells
    lb = sorted([r for r in rows if r["regime"] == "bull" and r["t"] is not None and abs(r["t"]) >= 3
                 and r["months"] >= 12], key=lambda r: -abs(r["demeaned"]))[:30]
    print(f"\n=== TOP standalone cells (bull, |t|>=3, >=12mo), ranked by |cross-sectional demeaned return| ===", flush=True)
    print(f"  {'signal':26}{'exit':7}{'dir':6}{'n':>7}{'mo':>4}{'dem%':>8}{'abs%':>8}{'win%':>7}{'t':>7}", flush=True)
    for r in lb:
        print(f"  {r['signal'][:25]:26}{r['exit']:7}{r['direction']:6}{r['n']:>7}{r['months']:>4}"
              f"{r['demeaned']:>8.2f}{r['abs_ret']:>8.2f}{r['win']:>7.1f}{r['t']:>7.2f}", flush=True)
    if save_db:
        try:
            from core.models import BacktestResult
            from django.utils import timezone
            BacktestResult.objects.update_or_create(kind="signal_discovery",
                defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
            print("\nSaved BacktestResult[signal_discovery]", flush=True)
        except Exception as e:
            print("DB save failed:", e, flush=True)
    return payload


def _opt(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    jobs = int(_opt("--jobs", "16"))
    limit = _opt("--limit"); limit = int(limit) if limit else None
    sk = _opt("--signals"); sk = sk.split(",") if sk else None
    run(jobs, limit=limit, signal_keys=sk, save_db=("--no-db-save" not in sys.argv))
