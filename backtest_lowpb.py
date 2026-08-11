#!/usr/bin/env python3
"""Decompose WHERE the sector-rotation edge comes from -> .data/studies/backtest_lowpb.json.

Question: is the edge the SECTOR ROTATION, the STOCK PICK, or the COMBINATION? And does a
cheap-P/B (value) pick stack with a technical entry signal?

Monthly, equal-weight, point-in-time. For BOTH pick rules -- lowpb (cheapest positive P/B,
PIT) and momentum (highest trailing return) -- we compute a 3-ARM decomposition:

  ARM 1  "pick only"          each month buy the top-N picks across the WHOLE universe,
                              ignoring sectors (pure stock selection).
  ARM 2  "rotation only"      hold the top-N momentum SECTOR ETFs, no stock pick
                              (pure sector rotation; identical under both pick rules).
  ARM 3  "rotation + pick"    rotate into the top-N momentum sectors, then within each pick
                              one stock by the rule (the trend_stock_studies mixed engine).

If ARM3 ~= ARM1, the edge is the PICK. If ARM3 ~= ARM2, it's the ROTATION. If ARM3 > both,
the combination adds something.

Plus a "value INTERSECT technical" section: among the stocks FIRING a top technical signal in
a month, buy the cheapest-P/B one, vs the signal alone (equal-weight all firers) and lowpb alone.

Each arm reports: total_return, spy_total, vs_spy (alpha), annual_return, sharpe, max_drawdown,
t_stat (one-sample vs 0 over non-overlapping monthly periods), periods, and an equity curve.

PIT discipline: selection at a rebalance date uses ONLY data available at that date
(_available_at); single-name holds realize via _ret_delist (a delisting is a real exit, not a
drop -> no survivorship lookahead). Baskets use next-month realized returns (delisted -> dropped).

Run AFTER the heavy recompute finishes (loads stock candles -> avoid OOM competing for memory):
  docker exec -i rotation-backend-1 python -u /app/backtest_lowpb.py
  docker exec -i rotation-backend-1 python -u /app/backtest_lowpb.py --limit 150   # quick subset
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config
import sector_holdings
from studies import SIGNALS, _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import (_pit_monthly_panel, _available_at, _ret_delist, CRYPTO)

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "backtest_lowpb.json"
BENCH = getattr(config, "BENCHMARK", "SPY")

# --- knobs -------------------------------------------------------------------
LOOKBACK = 6          # trailing-return lookback (months) for momentum ranking (stocks & sectors)
TOP_N_STOCKS = 10     # basket size for the "pick only" arm
TOP_N_SECTORS = 10    # sectors held in the rotation arms
# technical signals to intersect with value (guarded if a key is absent from SIGNALS)
TECH_SIGNALS = ["obv_div_sort_pos", "rsi_x_above_sma"]


# --- stats (mirror backtest_concept._stats / ._curve) ------------------------
def _stats(rets, spy_rets):
    r = np.array(rets, dtype=float)
    n = len(r)
    if n == 0:
        return {"total_return": 0, "spy_total": 0, "vs_spy": 0, "annual_return": 0,
                "sharpe": 0, "max_drawdown": 0, "t_stat": None, "periods": 0}
    total = float(np.prod(1 + r) - 1) * 100
    spy_total = float(np.prod(1 + np.array(spy_rets)) - 1) * 100
    ann = (float(np.prod(1 + r)) ** (12.0 / n) - 1) * 100
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r)
    dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    t = _tstat_from_returns(list(r))
    return {"total_return": round(total, 1), "spy_total": round(spy_total, 1),
            "vs_spy": round(total - spy_total, 1), "annual_return": round(ann, 1),
            "sharpe": round(sharpe, 2), "max_drawdown": round(dd, 1),
            "t_stat": round(t, 2) if t is not None else None, "periods": n}


def _curve(rets, spy_rets, index):
    eq = np.cumprod(1 + np.array(rets)) if rets else []
    seq = np.cumprod(1 + np.array(spy_rets)) if spy_rets else []
    return [{"date": str(pd.Timestamp(d).date()), "strat": round(float(s), 4), "spy": round(float(sp), 4)}
            for d, s, sp in zip(index, eq, seq)]


def _arm(select_fn, index, spy_fwd, warmup):
    """Monthly loop: at each rebalance i, select_fn(i) returns the realized (index[i]->index[i+1])
    equal-weight portfolio return, or None to skip the period (e.g. no candidates)."""
    rets, spy_rets, idx = [], [], []
    for i in range(warmup, len(index) - 1):
        sp = spy_fwd.iloc[i]
        if pd.isna(sp):
            continue
        pr = select_fn(i)
        if pr is None:
            continue
        rets.append(float(pr)); spy_rets.append(float(sp)); idx.append(index[i + 1])
    return {"summary": _stats(rets, spy_rets), "curve": _curve(rets, spy_rets, idx)}


# --- data --------------------------------------------------------------------
def _monthly_close(daily, min_len=60):
    """{ticker: daily OHLCV df} -> month-end Close DataFrame (union calendar), tickers with history."""
    cols = {t: df["Close"].resample("ME").last() for t, df in daily.items() if len(df) > min_len}
    return pd.DataFrame(cols).sort_index()


def build():
    etfs = {name: etf for name, etf in config.SECTOR_ETFS.items() if etf not in CRYPTO}
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    # sector -> (etf, holdings); collect the stock universe
    sector_map, all_holds = {}, set()
    for name, etf in etfs.items():
        h = [t for t in sector_holdings.get_holdings(name) if t not in (etf, BENCH) and t not in CRYPTO]
        sector_map[etf] = (name, h)
        all_holds.update(h)
    all_holds = sorted(all_holds)
    if limit:
        all_holds = all_holds[:limit]
        hset = set(all_holds)
        sector_map = {e: (n, [h for h in hs if h in hset]) for e, (n, hs) in sector_map.items()}

    etf_tickers = list(etfs.values())
    print(f"Loading {len(etf_tickers)} ETFs + {len(all_holds)} stocks + {BENCH} from DB...", flush=True)

    # Full daily OHLCV once (reused for both monthly frames and technical signals).
    etf_daily = load_candles(etf_tickers + [BENCH])
    etf_monthly = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tickers})
    spy_daily = etf_daily.get(BENCH)
    midx = etf_monthly.index
    spy_m = (spy_daily["Close"].resample("ME").last().reindex(midx) if spy_daily is not None
             else pd.Series(index=midx, dtype=float))

    stock_daily = load_candles(all_holds)
    stock_monthly = _monthly_close(stock_daily).reindex(midx)

    # Trailing (ranking) and forward (realized) return matrices, all on the ETF monthly calendar.
    etf_trail = etf_monthly.pct_change(LOOKBACK)
    etf_fwd = etf_monthly.pct_change().shift(-1)
    stock_trail = stock_monthly.pct_change(LOOKBACK)
    stock_fwd = stock_monthly.pct_change().shift(-1)
    spy_fwd = spy_m.pct_change().shift(-1)

    # 200-day MA trend per sector ETF: month-end distance of daily Close above its 200d SMA
    # (>0 = above = uptrend). Selects which sectors the 200-MA rotation holds.
    etf_sma200 = pd.DataFrame({
        t: (d["Close"] / d["Close"].rolling(200).mean() - 1).resample("ME").last()
        for t, d in etf_daily.items() if t in etf_tickers and len(d) >= 200}).reindex(midx)

    # Point-in-time P/B panel (mirror trend_stock_studies.run): mktcap / book equity, ffilled by
    # the report's avail_date so only already-published fundamentals are used at each rebalance.
    reps = load_financial_reports(all_holds)
    shares_p = _pit_monthly_panel(reps, "shares_outstanding", midx)
    equity_p = _pit_monthly_panel(reps, "total_equity", midx)
    common = stock_monthly.columns.intersection(shares_p.columns).intersection(equity_p.columns)
    pb_panel = (stock_monthly[common] * shares_p[common]) / equity_p[common].where(equity_p[common] != 0)

    print(f"months {len(midx)} | ETFs {etf_monthly.shape[1]} | stocks {stock_monthly.shape[1]} | "
          f"pb panel {pb_panel.shape}", flush=True)

    warmup = max(LOOKBACK, 1)

    # ---- pick helpers -------------------------------------------------------
    def _avail_mask(date):
        row = stock_monthly.loc[date]
        return row[(row.notna()) & (row > 0)].index

    def _pick_in_sector(holds, date, panel, direction):
        cands = [h for h in holds if h in stock_monthly.columns and _available_at(stock_monthly[h], date)]
        if not cands or panel is None or date not in panel.index:
            return None
        row = panel.loc[date, [c for c in cands if c in panel.columns]].dropna()
        if direction == "min":
            row = row[row > 0]           # cheapest POSITIVE P/B (ignore negative-book)
        if not len(row):
            return None
        return row.idxmin() if direction == "min" else row.idxmax()

    def _etf_ret(etf, date, ndate):
        if etf in etf_monthly.columns:
            r = _ret_delist(etf_monthly[etf], date, ndate)
            if r is not None:
                return r
        return 0.0

    # ---- ARM 1: pick only (whole universe) ----------------------------------
    def _pick_only(kind):
        def sel(i):
            date = midx[i]
            cands = _avail_mask(date)
            if kind == "lowpb":
                row = pb_panel.loc[date, [c for c in cands if c in pb_panel.columns]].dropna()
                row = row[row > 0]
                picks = row.nsmallest(TOP_N_STOCKS).index
            else:  # momentum
                row = stock_trail.loc[date, [c for c in cands if c in stock_trail.columns]].dropna()
                picks = row.nlargest(TOP_N_STOCKS).index
            fwd = stock_fwd.loc[date, [p for p in picks if p in stock_fwd.columns]].dropna()
            return float(fwd.mean()) if len(fwd) else None
        return sel

    # ---- ARM 2: rotation only (top momentum sector ETFs) --------------------
    def _rotation_only(i):
        date = midx[i]
        ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N_SECTORS)
        fwd = etf_fwd.loc[date, [e for e in ranks.index if e in etf_fwd.columns]].dropna()
        return float(fwd.mean()) if len(fwd) else None

    # ---- ARM 3: rotation + pick ---------------------------------------------
    def _rotation_plus_pick(kind):
        panel, direction = (pb_panel, "min") if kind == "lowpb" else (stock_trail, "max")

        def sel(i):
            date, ndate = midx[i], midx[i + 1]
            ranks = etf_trail.loc[date].dropna().sort_values(ascending=False).head(TOP_N_SECTORS)
            if not len(ranks):
                return None
            rets = []
            for etf in ranks.index:
                _, holds = sector_map.get(etf, (etf, []))
                pick = _pick_in_sector(holds, date, panel, direction)
                if pick is not None:
                    r = _ret_delist(stock_monthly[pick], date, ndate)
                    rets.append(r if r is not None else _etf_ret(etf, date, ndate))
                else:                       # no qualifying stock -> hold the ETF (mirror _pick_return)
                    rets.append(_etf_ret(etf, date, ndate))
            return float(np.mean(rets)) if rets else None
        return sel

    rotation_arm = _arm(_rotation_only, midx, spy_fwd, warmup)
    decomposition = {}
    for kind in ("lowpb", "momentum"):
        decomposition[kind] = {
            "arm1_pick_only": _arm(_pick_only(kind), midx, spy_fwd, warmup),
            "arm2_rotation_only": rotation_arm,
            "arm3_rotation_plus_pick": _arm(_rotation_plus_pick(kind), midx, spy_fwd, warmup),
        }

    # ---- 200-day MA rotation: BOTH numbers (hold the ETF vs pick the stock after) ------
    # Rotate into sectors trading above their 200d MA (ranked by distance), then report the
    # ETF-level return AND the stock-pick-after return so we always see both.
    def _sma200_ranks(date):
        if date not in etf_sma200.index:
            return []
        row = etf_sma200.loc[date].dropna()
        row = row[row > 0]                       # only sectors ABOVE their 200d MA (uptrend)
        return list(row.sort_values(ascending=False).head(TOP_N_SECTORS).index)

    def _sma200_rotation_only(i):
        date = midx[i]
        ranks = _sma200_ranks(date)
        if not ranks:
            return None                           # all sectors below 200d MA -> cash (skip period)
        fwd = etf_fwd.loc[date, [e for e in ranks if e in etf_fwd.columns]].dropna()
        return float(fwd.mean()) if len(fwd) else None

    def _sma200_rotation_plus_pick(kind):
        panel, direction = (pb_panel, "min") if kind == "lowpb" else (stock_trail, "max")

        def sel(i):
            date, ndate = midx[i], midx[i + 1]
            ranks = _sma200_ranks(date)
            if not ranks:
                return None
            rets = []
            for etf in ranks:
                _, holds = sector_map.get(etf, (etf, []))
                pick = _pick_in_sector(holds, date, panel, direction)
                if pick is not None:
                    r = _ret_delist(stock_monthly[pick], date, ndate)
                    rets.append(r if r is not None else _etf_ret(etf, date, ndate))
                else:
                    rets.append(_etf_ret(etf, date, ndate))
            return float(np.mean(rets)) if rets else None
        return sel

    sma200_rotation = {
        "rotation_only": _arm(_sma200_rotation_only, midx, spy_fwd, warmup),          # ETF number
        "rotation_plus_pick": {                                                       # stock numbers
            kind: _arm(_sma200_rotation_plus_pick(kind), midx, spy_fwd, warmup)
            for kind in ("lowpb", "momentum")},
    }

    # ---- value INTERSECT technical -----------------------------------------
    lowpb_alone = decomposition["lowpb"]["arm1_pick_only"]
    vxt = {"lowpb_alone": lowpb_alone, "signals": []}
    for key in TECH_SIGNALS:
        if key not in SIGNALS:
            vxt["signals"].append({"signal_key": key, "present": False})
            continue
        name, fn = SIGNALS[key]
        # monthly "fired this month" panel (any daily fire within the month), PIT.
        fired = {}
        for tk, df in stock_daily.items():
            if len(df) < 60:
                continue
            try:
                s = fn(df).fillna(False)
            except Exception:
                continue
            fired[tk] = s.resample("ME").max().reindex(midx).fillna(False)
        if not fired:
            vxt["signals"].append({"signal_key": key, "signal_name": name, "present": True,
                                   "fires": 0})
            continue
        fpanel = pd.DataFrame(fired).reindex(
            columns=[c for c in stock_monthly.columns if c in fired]).fillna(False).astype(bool)

        def _firers(date):
            if date not in fpanel.index:
                return []
            frow = fpanel.loc[date].astype(bool)
            avail = set(_avail_mask(date))
            return [t for t in frow[frow].index if t in avail]

        def _sig_alone(i):
            date = midx[i]
            firing = [t for t in _firers(date) if t in stock_fwd.columns]
            fwd = stock_fwd.loc[date, firing].dropna() if firing else pd.Series(dtype=float)
            return float(fwd.mean()) if len(fwd) else None

        def _sig_lowpb(i):
            date, ndate = midx[i], midx[i + 1]
            firing = [t for t in _firers(date) if t in pb_panel.columns]
            if not firing:
                return None
            row = pb_panel.loc[date, firing].dropna()
            row = row[row > 0]
            if not len(row):
                return None
            pick = row.idxmin()
            r = _ret_delist(stock_monthly[pick], date, ndate)
            return r if r is not None else None

        vxt["signals"].append({
            "signal_key": key, "signal_name": name, "present": True,
            "monthly_fires": int(fpanel.values.sum()),
            "signal_alone": _arm(_sig_alone, midx, spy_fwd, warmup),
            "signal_plus_lowpb": _arm(_sig_lowpb, midx, spy_fwd, warmup),
        })

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"lookback_months": LOOKBACK, "top_n_stocks": TOP_N_STOCKS,
                   "top_n_sectors": TOP_N_SECTORS, "rebalance": "monthly", "benchmark": BENCH,
                   "limit": limit},
        "universe": {"etfs": int(etf_monthly.shape[1]), "stocks": int(stock_monthly.shape[1]),
                     "months": int(len(midx))},
        "decomposition": decomposition,
        "sma200_rotation": sma200_rotation,
        "value_x_technical": vxt,
    }
    return payload


def _line(tag, s):
    return (f"  {tag:26} total {s['total_return']:>8.1f}%  vs SPY {s['vs_spy']:>7.1f}%  "
            f"Sharpe {s['sharpe']:>5.2f}  DD {s['max_drawdown']:>7.1f}%  t={s['t_stat']}  n={s['periods']}")


def _save_db(kind, payload):
    """Persist to Postgres (BacktestResult) so the decomposition lives in the DB, not only in JSON."""
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind=kind, defaults={"payload": payload, "computed_at": timezone.now()})
        print(f"Saved BacktestResult[{kind}] to DB", flush=True)
    except Exception as e:
        print(f"DB save failed for {kind}: {e}", flush=True)


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    _save_db("decomposition", payload)

    print("\n=== 3-ARM DECOMPOSITION (monthly, equal-weight, PIT) ===", flush=True)
    for kind, arms in payload["decomposition"].items():
        print(f"\n{kind.upper()} pick:", flush=True)
        print(_line("ARM1 pick only", arms["arm1_pick_only"]["summary"]), flush=True)
        print(_line("ARM2 rotation only", arms["arm2_rotation_only"]["summary"]), flush=True)
        print(_line("ARM3 rotation + pick", arms["arm3_rotation_plus_pick"]["summary"]), flush=True)

    sm = payload["sma200_rotation"]
    print("\n=== 200-DAY MA ROTATION (both numbers: hold ETF vs pick stock after) ===", flush=True)
    print(_line("rotation only (ETF)", sm["rotation_only"]["summary"]), flush=True)
    for kind, arm in sm["rotation_plus_pick"].items():
        print(_line(f"+ {kind} pick after", arm["summary"]), flush=True)

    print("\n=== VALUE (cheapest P/B) INTERSECT TECHNICAL ===", flush=True)
    print(_line("lowpb alone (ARM1)", payload["value_x_technical"]["lowpb_alone"]["summary"]), flush=True)
    for sg in payload["value_x_technical"]["signals"]:
        if not sg.get("present"):
            print(f"  [absent from SIGNALS] {sg['signal_key']}", flush=True)
            continue
        if "signal_alone" not in sg:
            print(f"  [no fires] {sg['signal_name']}", flush=True)
            continue
        print(f"\n  {sg['signal_name']} ({sg['signal_key']}, {sg['monthly_fires']} monthly fires):", flush=True)
        print(_line("signal alone", sg["signal_alone"]["summary"]), flush=True)
        print(_line("signal + lowpb pick", sg["signal_plus_lowpb"]["summary"]), flush=True)

    print("\nSaved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
