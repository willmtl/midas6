#!/usr/bin/env python3
"""BURST scanner — two dashboard sections in one candle pass.

SHORT-TERM: stocks showing a short-term BURST right now — a burst trigger fired within the last
`recent` bars — tagged MOMENTUM (thrusting up) or REVERSAL (snapping back off a vol-shock-down /
oversold). Joined with the trigger's SHORT-horizon historical edge, fundamentals, sector, smart
money. -> ShortTermSignal.

GLOBAL: the CONFLUENCE overlay. A live burst is REQUIRED, so the base set is the burst names; each
is scored 0-100 across our validated layers — burst strength, short-term edge, A/D accumulation,
dark-pool accumulation, insider / 13D-13G, fundamental amplifier profile, favorable regime — with a
per-component breakdown, ranked by score. -> GlobalSignal.

Signals-only per ticker (no exit loop) -> fast; parallel spawn pool like live_firing_scan.
Run:  docker exec rotation-backend-1 python -u /app/burst_scan.py --db --jobs 4
Opts: --jobs N  --recent N (bars, default 2)  --no-db-save
"""
import warnings
warnings.filterwarnings("ignore")

import json
import os
import sys
from pathlib import Path
import numpy as np

from studies import SIGNALS, _vol_shock_z
from seq_fundamental_study import (
    build_universe, load_fundamentals, load_candles, DIMENSIONS, DEFAULT_JOBS, _chunk, MIN_BARS,
)
from all_on_all_study import _prepare_indicators
from pit_fundamentals import _ad_state

STUDIES_DIR = Path(__file__).parent / ".data" / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

# Burst trigger set, tagged. Momentum = upward thrust; reversal = short-term snap-back.
MOMENTUM = ["vol_shock_up_hivol", "vol_shock_up", "new_20high", "gap_up_large", "rsi_x_sma_below50"]
REVERSAL = ["vol_shock_dn_hivol", "vol_shock_dn", "rsi_oversold20", "new_52low", "rsi_sup10_x_dd70"]
BURST = [(k, "momentum") for k in MOMENTUM] + [(k, "reversal") for k in REVERSAL]
SHORT_EXITS = {"1d", "3d", "1w", "2w", "4w"}   # short-horizon edge for the short-term view
AD_STATE_NAME = {2: "accum divergence", 1: "accum trend-up", 0: "neutral", -1: "distribution"}

# Confluence component weights (sum = 100). global_score = Σ w_k · component_k(0..1).
# `news`     = a recent GROUNDED good-news crash the burst is bouncing off (overreaction reversal).
# `intraday` = an OVERSOLD intraday RSI cross-up (EODHD 1h -> 8h/12h) timing the entry — the study's
#              edge lives in oversold (<35) crossovers, so this rewards well-timed reversal entries.
W = {"burst": 12, "edge": 12, "ad": 12, "darkpool": 10, "smart_money": 8,
     "fundamentals": 8, "regime": 8, "news": 15, "intraday": 15}


def _fetch_recent_1h(sym, days=60):
    """Recent EODHD 1h bars for a live candidate (one request; ~60d is enough for 8h/12h RSI(14))."""
    import time
    import requests
    import pandas as pd
    eod = os.environ.get("EODHD_API_KEY", "")
    if not eod:
        return None
    end = int(time.time()); frm = end - days * 86400
    try:
        r = requests.get(f"https://eodhd.com/api/intraday/{sym}.US?interval=1h&from={frm}&to={end}"
                         f"&api_token={eod}&fmt=json", timeout=25)
        j = r.json()
    except Exception:
        return None
    if not isinstance(j, list) or not j:
        return None
    df = pd.DataFrame(j)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return (df.set_index("dt").sort_index()[["open", "high", "low", "close", "volume"]]
            .apply(pd.to_numeric, errors="coerce").dropna())


def _tf_signal(df1h, hours):
    import ta
    d = df1h.resample(f"{hours}h").agg({"open": "first", "high": "max", "low": "min",
                                        "close": "last", "volume": "sum"}).dropna()
    if len(d) < 30:
        return None
    rsi = ta.momentum.rsi(d["close"], window=14)
    sma = rsi.rolling(14).mean()
    up = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    cur = float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else None
    for k in (1, 2):                          # crossed up in the last 1-2 bars?
        if len(up) > k and bool(up.iloc[-k]):
            rc = float(rsi.iloc[-k])
            comp = 1.0 if rc < 25 else 0.7 if rc < 35 else 0.4 if rc < 45 else 0.2
            return comp, f"{hours}h RSI↑ from {rc:.0f}", cur
    if cur is not None and cur < 30:          # oversold, primed (no cross yet)
        return 0.3, f"{hours}h oversold RSI {cur:.0f}", cur
    return 0.1, "", cur


def intraday_timing(sym):
    """Best oversold intraday RSI-cross signal across 8h & 12h. Returns (component, signal_str, rsi)."""
    df1h = _fetch_recent_1h(sym)
    if df1h is None or len(df1h) < 60:
        return 0.1, "", None
    best = (0.1, "", None)
    for h in (8, 12):
        r = _tf_signal(df1h, h)
        if r and r[0] > best[0]:
            best = r
    return best


def _worker(payload):
    burst, recent, tickers = payload
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    try:
        django.setup()
    except Exception:
        pass
    candles = load_candles(tickers)
    out = []  # (ticker, sk, btype, days_ago, last_close, day1_move, z_shock, ad_code)
    for tk, sdf in candles.items():
        if len(sdf) < MIN_BARS:
            continue
        _prepare_indicators(sdf)
        close = sdf["Close"].values
        last_close = float(close[-1])
        ret = sdf["Close"].pct_change().values
        z = _vol_shock_z(sdf).values
        tk_hits = []
        for sk, btype in burst:
            fn = SIGNALS[sk][1]
            try:
                sig = fn(sdf).fillna(False)
            except Exception:
                continue
            recent_vals = sig.iloc[-recent:].tolist()
            if not any(recent_vals):
                continue
            days_ago = next(i for i, v in enumerate(reversed(recent_vals)) if v)
            idx = len(close) - 1 - days_ago
            d1 = round(float(ret[idx]) * 100, 2) if 0 <= idx < len(ret) and np.isfinite(ret[idx]) else None
            zz = round(float(z[idx]), 2) if 0 <= idx < len(z) and np.isfinite(z[idx]) else None
            tk_hits.append((sk, btype, days_ago, d1, zz))
        if not tk_hits:
            continue
        try:
            adv = _ad_state(sdf).iloc[-1]
            adc = int(adv) if np.isfinite(adv) else None
        except Exception:
            adc = None
        for sk, btype, days_ago, d1, zz in tk_hits:
            out.append((tk, sk, btype, days_ago, round(last_close, 2), d1, zz, adc))
    return out


def _short_edges(signal_keys, min_trades=200):
    """Best SHORT-horizon exit edge per burst signal from StockStudy (robust-first, then avg)."""
    from core.models import StockStudy
    best = {}
    for sk, ek, avg, wr, tr, ts in (
            StockStudy.objects.filter(signal_key__in=signal_keys, exit_key__in=list(SHORT_EXITS),
                                      total_trades__gte=min_trades)
            .values_list("signal_key", "exit_key", "avg_return", "win_rate", "total_trades", "t_stat")):
        robust = ts is not None and abs(ts) >= 2
        rk = (1 if robust else 0, avg if avg is not None else -1e9)
        cur = best.get(sk)
        if cur is None or rk > cur[0]:
            best[sk] = (rk, {"best_exit_key": ek, "hist_avg_return": avg,
                             "hist_win_rate": wr, "hist_trades": tr})
    return {k: v[1] for k, v in best.items()}


def _fav_fund_score(f):
    """0..1 favorable-amplifier score from raw fundamentals (the study's edge amplifiers:
    small/micro cap, cheap P/E, low float, profitable, growing)."""
    checks = []
    mc, pe, fl = f.get("market_cap"), f.get("pe_ratio"), f.get("float_shares")
    pm, rg = f.get("profit_margin"), f.get("revenue_growth")
    if mc is not None: checks.append(mc < 2e9)
    if pe is not None: checks.append(0 < pe < 15)
    if fl is not None: checks.append(fl < 100e6)
    if pm is not None: checks.append(pm > 0)
    if rg is not None: checks.append(rg > 0.2)
    if not checks:
        return 0.3
    return round(sum(1 for c in checks if c) / len(checks), 3)


def run(jobs, recent=2, save_db=True):
    import sector_holdings
    burst = [(k, t) for k, t in BURST if k in SIGNALS]
    signal_keys = [k for k, _ in burst]
    tickers = build_universe()
    funds = load_fundamentals(tickers)
    edges = _short_edges(signal_keys)
    print(f"Universe {len(tickers)}, burst signals {len(signal_keys)}, recent {recent} bars, jobs {jobs}",
          flush=True)

    hits = []
    if jobs <= 1:
        hits = _worker((burst, recent, tickers))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp
        try:
            from django.db import connections
            connections.close_all()
        except Exception:
            pass
        payloads = [(burst, recent, c) for c in _chunk(tickers, jobs * 3)]
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            for h in ex.map(_worker, payloads):
                hits.extend(h)

    def bkts(tk):
        f = funds.get(tk, {})
        return {dim: bfn(f.get(field)) for (dim, field, bfn, _o, pit) in DIMENSIONS if not pit}

    firing = list({h[0] for h in hits})
    from core.models import InsiderBuy, SecFiling, DarkPoolWeek
    from django.db.models import Sum, Count
    from datetime import date, timedelta
    today = date.today()
    ins = dict(InsiderBuy.objects.filter(ticker__in=firing, filed_date__gte=today - timedelta(days=180))
               .values_list("ticker").annotate(s=Sum("buy_value")))
    sec = {}
    for r in (SecFiling.objects.filter(ticker__in=firing, filed_date__gte=today - timedelta(days=180))
              .values("ticker", "form_group").annotate(n=Count("id"))):
        sec.setdefault(r["ticker"], {})[r["form_group"]] = r["n"]
    dp = {}   # ticker -> [off_pct newest-first]
    for tk, ov in (DarkPoolWeek.objects.filter(ticker__in=firing, off_pct__isnull=False)
                   .order_by("ticker", "-week_start").values_list("ticker", "off_pct")):
        dp.setdefault(tk, []).append(ov)
    spy = load_candles(["SPY"]).get("SPY")
    regime_bull = bool(spy is not None and len(spy) > 200
                       and spy["Close"].iloc[-1] > spy["Close"].rolling(200).mean().iloc[-1])

    # ---- intraday entry timing for the live candidates (EODHD 1h -> 8h/12h oversold RSI cross) ----
    # Sequential + gentle sleep (rate-limit friendly); ~firing tickers once/night. Failures -> 0.1.
    import time as _t
    intraday = {}
    for _i, _tk in enumerate(firing):
        intraday[_tk] = intraday_timing(_tk)
        _t.sleep(0.1)
        if (_i + 1) % 100 == 0:
            print(f"  intraday {_i + 1}/{len(firing)}", flush=True)
    n_itim = sum(1 for v in intraday.values() if v[0] >= 0.7)
    print(f"intraday timing: {len(firing)} candidates, {n_itim} with an oversold cross", flush=True)

    # ---- ShortTermSignal rows (one per ticker×signal) ----
    st_rows = []
    for tk, sk, btype, days_ago, last_close, d1, zz, adc in hits:
        f = funds.get(tk, {})
        e = edges.get(sk, {})
        st_rows.append({
            "ticker": tk, "signal_key": sk, "signal_name": SIGNALS[sk][0], "burst_type": btype,
            "days_ago": days_ago, "last_close": last_close, "day1_move": d1, "z_shock": zz,
            "best_exit_key": e.get("best_exit_key", ""), "hist_avg_return": e.get("hist_avg_return"),
            "hist_win_rate": e.get("hist_win_rate"), "hist_trades": e.get("hist_trades"),
            "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
            "forward_pe": f.get("forward_pe"), "fund_buckets": bkts(tk),
            "sectors": sector_holdings.get_sectors_for_ticker(tk),
            "insider_buy_90d": ins.get(tk),
            "recent_13d": sec.get(tk, {}).get("13D", 0), "recent_13g": sec.get(tk, {}).get("13G", 0),
            "intraday_signal": intraday.get(tk, (0.1, "", None))[1],
            "intraday_rsi": intraday.get(tk, (0.1, "", None))[2],
        })
    st_rows.sort(key=lambda r: (r["days_ago"], -(r["hist_avg_return"] or 0)))

    # ---- news-overreaction signal per firing ticker (grounded verdict where available) ----
    # A recent good-news crash the burst is now bouncing off = the PODD-type reversal. Grounded so a
    # beat-that-guided-down is NOT treated as good news.
    from core.models import NewsItem, EarningsEvent
    import datetime as _dt
    recent_cut = _dt.date.today() - _dt.timedelta(days=20)
    gvmap = {}
    for etk, erd, gs in (EarningsEvent.objects.filter(ticker__in=firing, grounded_score__isnull=False,
                                                      report_date__gte=recent_cut)
                         .values_list("ticker", "report_date", "grounded_score")):
        gvmap[(etk, erd)] = gs
    news_by_tk = {}
    for tk_, dt_, abn_, rat_ in (NewsItem.objects.filter(
            ticker__in=firing, local_impact__gte=2, day_abn__isnull=False, dt__date__gte=recent_cut)
            .exclude(junk=True).values_list("ticker", "dt", "day_abn", "local_rating")):
        news_by_tk.setdefault(tk_, []).append((dt_.date(), abn_, rat_ or 0))
    news_sig = {}
    for tk_, items in news_by_tk.items():
        val = 0.3
        for d_, abn_, rat_ in items:
            g = gvmap.get((tk_, d_)) or gvmap.get((tk_, d_ - _dt.timedelta(days=1)))
            sent = g if g is not None else rat_
            if sent > 0 and abn_ is not None and abn_ <= -8:   # good news that crashed -> bounce setup
                val = max(val, 1.0)
            elif sent > 0:
                val = max(val, 0.5)
            elif sent < 0:
                val = min(val, 0.2)                            # bad grounded news -> burst suspect
        news_sig[tk_] = round(val, 2)

    # ---- GlobalSignal rows (one per ticker; best burst; confluence score) ----
    by_tk = {}
    for h in hits:
        by_tk.setdefault(h[0], []).append(h)
    g_rows = []
    for tk, hl in by_tk.items():
        # pick the "lead" burst: freshest, then best short edge
        hl.sort(key=lambda h: (h[3], -(edges.get(h[1], {}).get("hist_avg_return") or 0)))
        _, sk, btype, days_ago, last_close, d1, zz, adc = hl[0]
        f = funds.get(tk, {})
        e = edges.get(sk, {})
        b = bkts(tk)
        dpl = dp.get(tk, [])
        dp_level = dpl[0] if dpl else None
        dp_rising = bool(len(dpl) >= 2 and dpl[0] > dpl[1])
        ins_v = ins.get(tk)
        s13d = sec.get(tk, {}).get("13D", 0)
        s13g = sec.get(tk, {}).get("13G", 0)
        comps = {
            "burst": 1.0 if "hivol" in sk else 0.7,
            "edge": max(0.0, min(1.0, (e.get("hist_avg_return") or 0) / 20.0)),
            "ad": {2: 1.0, 1: 0.6, 0: 0.2, -1: 0.0}.get(adc if adc is not None else 0, 0.2),
            "darkpool": (1.0 if (dp_rising and (dp_level or 0) >= 0.12) else
                         0.6 if dp_rising else 0.3 if dpl else 0.0),
            "smart_money": min(1.0, (0.5 if (ins_v or 0) > 0 else 0) + (0.3 if s13d else 0) + (0.2 if s13g else 0)),
            "fundamentals": _fav_fund_score(f),
            "regime": 1.0 if regime_bull else 0.3,
            "news": news_sig.get(tk, 0.3),
            "intraday": intraday.get(tk, (0.1, "", None))[0],
        }
        itim = intraday.get(tk, (0.1, "", None))
        score = round(sum(W[k] * comps[k] for k in W), 1)
        g_rows.append({
            "ticker": tk, "global_score": score,
            "components": {k: round(comps[k], 3) for k in comps},
            "burst_signal_key": sk, "burst_signal_name": SIGNALS[sk][0], "burst_type": btype,
            "burst_days_ago": days_ago, "last_close": last_close,
            "best_signal_key": sk, "hist_avg_return": e.get("hist_avg_return"),
            "hist_win_rate": e.get("hist_win_rate"), "hist_trades": e.get("hist_trades"),
            "ad_state": AD_STATE_NAME.get(adc, ""), "darkpool_off_pct": dp_level,
            "darkpool_rising": dp_rising,
            "market_cap": f.get("market_cap"), "pe_ratio": f.get("pe_ratio"),
            "forward_pe": f.get("forward_pe"), "fund_buckets": b,
            "sectors": sector_holdings.get_sectors_for_ticker(tk),
            "insider_buy_90d": ins_v, "recent_13d": s13d, "recent_13g": s13g,
            "regime_bull": regime_bull, "sector_state": "",
            "intraday_signal": itim[1], "intraday_rsi": itim[2],
        })
    g_rows.sort(key=lambda r: -r["global_score"])

    out = {"recent_window": recent, "regime_bull": regime_bull,
           "universe_size": len(tickers), "n_burst": len(st_rows), "n_global": len(g_rows),
           "n_momentum": sum(1 for r in st_rows if r["burst_type"] == "momentum"),
           "n_reversal": sum(1 for r in st_rows if r["burst_type"] == "reversal"),
           "short_term": st_rows, "global": g_rows}
    (STUDIES_DIR / "burst_scan.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSHORT-TERM: {len(st_rows)} burst hits "
          f"({out['n_momentum']} momentum / {out['n_reversal']} reversal)", flush=True)
    print(f"GLOBAL: {len(g_rows)} confluence candidates (regime {'BULL' if regime_bull else 'BEAR'})", flush=True)
    for r in g_rows[:15]:
        print(f"  {r['ticker']:8} score {r['global_score']:>5.1f}  {r['burst_type']:8} "
              f"{r['burst_signal_key']:20} AD={r['ad_state'] or '-':16} "
              f"edge {(r['hist_avg_return'] or 0):+.1f}%", flush=True)

    if save_db:
        _save_short_term(st_rows)
        _save_global(g_rows)
    return out


def _save_short_term(rows):
    from core.models import ShortTermSignal
    from django.utils import timezone
    from django.db import transaction
    now = timezone.now()
    with transaction.atomic():
        for r in rows:
            ShortTermSignal.objects.update_or_create(
                ticker=r["ticker"], signal_key=r["signal_key"],
                defaults={k: r[k] for k in (
                    "signal_name", "burst_type", "days_ago", "last_close", "day1_move", "z_shock",
                    "best_exit_key", "hist_avg_return", "hist_win_rate", "hist_trades",
                    "market_cap", "pe_ratio", "forward_pe", "fund_buckets", "sectors",
                    "insider_buy_90d", "recent_13d", "recent_13g",
                    "intraday_signal", "intraday_rsi")} | {"computed_at": now})
    ShortTermSignal.objects.exclude(computed_at=now).delete()
    print(f"DB: upserted {len(rows)} ShortTermSignal rows, cleared stale.", flush=True)


def _save_global(rows):
    from core.models import GlobalSignal
    from django.utils import timezone
    from django.db import transaction
    now = timezone.now()
    with transaction.atomic():
        for r in rows:
            GlobalSignal.objects.update_or_create(
                ticker=r["ticker"],
                defaults={k: r[k] for k in (
                    "global_score", "components", "burst_signal_key", "burst_signal_name",
                    "burst_type", "burst_days_ago", "last_close", "best_signal_key",
                    "hist_avg_return", "hist_win_rate", "hist_trades", "ad_state",
                    "darkpool_off_pct", "darkpool_rising", "market_cap", "pe_ratio", "forward_pe",
                    "fund_buckets", "sectors", "insider_buy_90d", "recent_13d", "recent_13g",
                    "regime_bull", "sector_state",
                    "intraday_signal", "intraday_rsi")} | {"computed_at": now})
    GlobalSignal.objects.exclude(computed_at=now).delete()
    print(f"DB: upserted {len(rows)} GlobalSignal rows, cleared stale.", flush=True)


if __name__ == "__main__":
    argv = sys.argv

    def _opt(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    jobs = int(_opt("--jobs", min(4, DEFAULT_JOBS)))
    recent = int(_opt("--recent", 2))
    if "--db" in argv:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
        django.setup()
    run(jobs, recent=recent, save_db=("--db" in argv and "--no-db-save" not in argv))
