#!/usr/bin/env python3
"""FILL-QUALITY / slippage proxy for the flagship's month-end executions in holiday-shortened (4-day) weeks
vs normal weeks. We have no stored bid-ask history, so we proxy fill quality on the ACTUAL picks (trade log)
using daily candles on each pick's REAL execution day (last trading day <= the rebalance date):
  • intraday range pct = (high-low)/close   -> slippage risk (wider = worse fills near the close)
  • dollar volume = close*volume            -> liquidity (thinner = worse fills)
Shortened-week flag uses each TICKER'S OWN trading calendar (foreign exchanges have different holidays), so a
Brazilian/Canadian name is judged on its own week. If holiday-week executions are much wider / thinner, there is
a case to shift the rebalance a day; if comparable, the holiday-skip idea is a confirmed dead end.

Read-only. Saves /app/.data/studies/holiday_slippage.json + BacktestResult 'holiday_slippage'.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/holiday_slippage_study.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, BacktestResult

OUT = "/app/.data/studies/holiday_slippage.json"


def _stats(name, sub):
    if not len(sub):
        return dict(group=name, n=0)
    rng = sub["range_pct"].to_numpy(); dv = sub["dvol"].to_numpy()
    return dict(group=name, n=int(len(sub)),
                range_mean_pct=round(float(np.nanmean(rng)), 3), range_median_pct=round(float(np.nanmedian(rng)), 3),
                range_p90_pct=round(float(np.nanpercentile(rng, 90)), 3),
                dvol_median_musd=round(float(np.nanmedian(dv)) / 1e6, 2),
                dvol_p10_musd=round(float(np.nanpercentile(dv, 10)) / 1e6, 3))


def main():
    tl = json.load(open("/app/.data/studies/survivorship_trade_log.json"))
    trades = tl["trades"]
    tks = sorted({t["ticker"] for t in trades})
    print(f"{len(trades)} trades, {len(tks)} distinct tickers", flush=True)

    # per-ticker daily candles -> DataFrame with its own shortened-week flag
    cand = {}
    for tk in tks:
        rows = list(Candle.objects.filter(ticker=tk).values("date", "high", "low", "close", "volume"))
        if not rows:
            continue
        df = pd.DataFrame(rows); df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        iso = df.index.isocalendar()
        wk = pd.Series(list(zip(iso["year"], iso["week"])), index=df.index)
        df["wkcount"] = wk.map(wk.value_counts())
        df["short"] = df["wkcount"] < 5
        df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
        df["dvol"] = df["close"] * df["volume"]
        cand[tk] = df

    recs = []
    miss = 0
    for t in trades:
        df = cand.get(t["ticker"])
        if df is None:
            miss += 1; continue
        d = pd.Timestamp(t["date"])
        sub = df.loc[:d]
        if not len(sub):
            miss += 1; continue
        row = sub.iloc[-1]                                   # last trading day <= rebalance date = execution day
        recs.append(dict(ticker=t["ticker"], exec_day=sub.index[-1], month=d.month,
                         short=bool(row["short"]), range_pct=float(row["range_pct"]),
                         dvol=float(row["dvol"])))
    f = pd.DataFrame(recs)
    print(f"matched {len(f)} executions ({miss} unmatched)", flush=True)

    res = {"n_exec": len(f), "unmatched": miss,
           "by_week": [_stats("exec in FULL 5-day week", f[~f.short]),
                       _stats("exec in SHORTENED (<5) week", f[f.short])],
           "holiday_months": [_stats("Nov+Dec executions", f[f.month.isin([11, 12])]),
                              _stats("all other months", f[~f.month.isin([11, 12])])],
           "all": _stats("ALL executions", f)}
    # what fraction of executions even land in a shortened week?
    res["pct_exec_in_short_week"] = round(float(f.short.mean()) * 100, 1) if len(f) else None

    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="holiday_slippage",
                                                defaults=dict(payload=res, computed_at=timezone.now()))
        print("saved BacktestResult 'holiday_slippage'", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
