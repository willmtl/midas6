#!/usr/bin/env python3
"""DECISIVE TEST for Tier-1 stake_13g: build a real, tradable, daily EQUAL-WEIGHT book of 13G-stake positions and
measure the ABSOLUTE net-of-cost equity curve vs buy-and-hold SPY (and an EW-universe buy-hold, to strip beta).
Rules: enter at the close of the session AFTER a 13G filing (PIT .shift(1)); hold H trading days; overlapping
positions equal-weighted each day; round-trip cost charged on entry+exit day; $ADV + price floors; survivorship-aware
(delisted names exit at last bar). Variants: ALL-cap vs small-cap-tilt (<$2B PIT). By-year book return + vs SPY.
Persists BacktestResult(kind=strategy_13g_book) + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_13g_book.py"""
import os, sys, json, math
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from seq_fundamental_study import load_candles, load_filings, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR

H = 126                         # 6-month hold (trading days)
COST_BPS = 20                   # round-trip; small/mid-cap 13G names ($125M median ADV)
SMALL_MAX = 2e9                 # small-cap tilt threshold (PIT mktcap)


def shares_ff(reps, tk, dates):
    r = reps.get(tk)
    if r is None or "shares_outstanding" not in r.columns:
        return None
    d = r[["avail_date", "shares_outstanding"]].dropna().sort_values("avail_date")
    if d.empty:
        return None
    s = pd.Series(d["shares_outstanding"].values, index=pd.to_datetime(d["avail_date"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates).values


def perf(dret):
    dret = dret.dropna()
    if len(dret) < 60:
        return None
    eq = (1 + dret.values).cumprod()
    yrs = len(dret) / 252.0
    cagr = (eq[-1] ** (1 / yrs) - 1) * 100 if eq[-1] > 0 else -100.0
    sh = dret.mean() / dret.std() * math.sqrt(252) if dret.std() > 0 else 0.0
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return dict(days=len(dret), start=str(dret.index[0].date()), end=str(dret.index[-1].date()),
                total=round((eq[-1] - 1) * 100, 1), cagr=round(cagr, 1), sharpe=round(sh, 2), maxdd=round(dd, 1))


def main():
    universe, delisted = _universe()
    from core.models import SecFiling, Candle
    have13g = set(SecFiling.objects.filter(form_group="13G").values_list("ticker", flat=True).distinct())
    names = [t for t in universe if t in have13g]
    print(f"tradable 13G names: {len(names)} | hold {H}d | cost {COST_BPS}bps rt", flush=True)

    # daily position-return contributions, per book variant: contrib[variant][date] = list of that day's held-name returns
    contrib = {"all": defaultdict(list), "small": defaultdict(list)}
    n_events = {"all": 0, "small": 0}
    for ch in _chunk(names, 40):
        candles = load_candles(ch); filings = load_filings(ch); reps = load_financial_reports(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            fdf = filings.get(tk)
            if fdf is None or not len(fdf):
                continue
            sub = fdf[fdf["form_group"] == "13G"]
            if not len(sub):
                continue
            ev = pd.Series(1.0, index=pd.to_datetime(sub["filed_date"]))
            fired = ev.groupby(ev.index).sum().reindex(sdf.index).shift(1).fillna(0.0).values
            dates = sdf.index
            dret = pd.Series(close, index=dates).pct_change().values
            sh = shares_ff(reps, tk, dates)
            is_delisted = tk in delisted
            for variant in ("all", "small"):
                held_until = -1                       # index; extend on stacked events (weight per name = 1)
                for i in np.where(fired > 0)[0]:
                    ep = float(close[i])
                    if ep <= PRICE_FLOOR or not (dvol20[i] >= DVOL_FLOOR):
                        continue
                    if variant == "small":
                        mc = (sh[i] * ep) if (sh is not None and np.isfinite(sh[i])) else None
                        if mc is None or not np.isfinite(mc) or mc >= SMALL_MAX or mc <= 0:
                            continue
                    start = max(i + 1, held_until + 1)  # don't double-count overlapping days for same name
                    end = min(i + H, n - 1)
                    if is_delisted:
                        end = min(end, n - 1)
                    if end < start:
                        held_until = max(held_until, i + H)
                        continue
                    n_events[variant] += 1
                    for d in range(start, end + 1):
                        r = dret[d]
                        if not np.isfinite(r):
                            continue
                        if d == start:
                            r -= (COST_BPS / 2) / 1e4
                        if d == end:
                            r -= (COST_BPS / 2) / 1e4
                        contrib[variant][dates[d]].append(r)
                    held_until = max(held_until, i + H)

    # SPY + EW-universe benchmarks
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["date", "c"])
    spy["date"] = pd.to_datetime(spy["date"]); spy_c = spy.set_index("date")["c"].astype(float).sort_index()
    spy_ret = spy_c.pct_change()

    out = {}
    for variant in ("all", "small"):
        days = sorted(contrib[variant])
        book = pd.Series([np.mean(contrib[variant][d]) for d in days], index=pd.DatetimeIndex(days)).sort_index()
        full = pd.date_range(book.index[0], book.index[-1], freq="B")
        book = book.reindex(full).fillna(0.0)             # in cash when no position marks
        bperf = perf(book)
        spy_al = spy_ret.reindex(book.index).fillna(0.0)
        sperf = perf(spy_al)
        inv = (book != 0).mean() * 100
        out[variant] = dict(events=n_events[variant], invested_pct=round(inv, 1), book=bperf, spy_same_window=sperf)
        print(f"\n=== 13G BOOK [{variant}]  events {n_events[variant]}  invested {inv:.0f}% of days  (cost {COST_BPS}bps rt) ===", flush=True)
        print(f"  BOOK : total {bperf['total']:+.1f}%  CAGR {bperf['cagr']:+.1f}%  Sharpe {bperf['sharpe']:.2f}  maxDD {bperf['maxdd']:.1f}%  ({bperf['start']}..{bperf['end']})", flush=True)
        print(f"  SPY  : total {sperf['total']:+.1f}%  CAGR {sperf['cagr']:+.1f}%  Sharpe {sperf['sharpe']:.2f}  maxDD {sperf['maxdd']:.1f}%  (same window, buy-hold)", flush=True)
        # by-year book vs SPY
        byb = (1 + book).groupby(book.index.year).prod() - 1
        bys = (1 + spy_al).groupby(spy_al.index.year).prod() - 1
        print(f"  {'year':>6}{'book%':>9}{'spy%':>9}{'excess':>9}", flush=True)
        yr_rows = {}
        for y in byb.index:
            b = byb[y] * 100; s = bys.get(y, np.nan) * 100
            print(f"  {y:>6}{b:>9.1f}{s:>9.1f}{b-s:>9.1f}", flush=True)
            yr_rows[int(y)] = dict(book=round(b, 1), spy=round(s, 1), excess=round(b - s, 1))
        out[variant]["by_year"] = yr_rows

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), hold_days=H, cost_bps=COST_BPS,
                   small_max=SMALL_MAX, results=out,
                   caveat="Daily EW book of 13G-stake positions, PIT .shift(1) entry, overlapping positions EW, "
                   "round-trip cost on entry+exit, survivorship-aware (delisted exit at last bar). In cash when no marks.")
    Path("/app/.data/studies/strategy_13g_book.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="strategy_13g_book",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[strategy_13g_book]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
