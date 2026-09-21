#!/usr/bin/env python3
"""Is there a case for NOT buying/selling the flagship on a holiday-shortened (4-day) week, or around
Christmas/holidays? The flagship rebalances at each month-END (last trading day): it SELLS the old basket
and BUYS the new one that day. The only holiday-adjacent monthly executions are end-of-November (Thanksgiving
week) and end-of-December (Christmas/New-Year week). We classify each of the 124 deployed monthly returns by
the holiday context of its BUY day (prior month-end) and SELL day (this month-end), using SPY's ACTUAL trading
calendar to flag shortened weeks — then test whether returns differ. Also a plain SPY-daily holiday-effect check.

No re-run of the engine: uses the deployed adaptive series (flagship_history.json monthly_net) + SPY daily candles.
Saves BacktestResult + /app/.data/studies/holiday_timing.json. Read-only otherwise.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/holiday_timing_study.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle, BacktestResult

OUT = "/app/.data/studies/holiday_timing.json"


def _t(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def _grp(name, r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return dict(group=name, n=int(len(r)), mean_pct=round(float(r.mean()) * 100, 3) if len(r) else None,
                median_pct=round(float(np.median(r)) * 100, 3) if len(r) else None,
                win_pct=round(float((r > 0).mean()) * 100, 1) if len(r) else None,
                t=round(_t(r), 2))


def main():
    # ---- SPY trading calendar ----
    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values("date", "open", "close")))
    spy["date"] = pd.to_datetime(spy["date"]); spy = spy.sort_values("date").set_index("date")
    cal = spy.index
    iso = pd.Series(cal, index=cal).dt.isocalendar()
    wk = pd.Series(list(zip(iso["year"], iso["week"])), index=cal)      # (isoyear, isoweek) per trading day
    wkcount = wk.map(wk.value_counts())                                 # trading days in that week
    shortened = (wkcount < 5)                                           # holiday-shortened week flag per trading day
    last_td_of_month = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max()  # last trading day / month

    # US market holidays (approx) to measure pre-holiday drift: any weekday gap in the trading calendar
    gaps = pd.Series(cal).diff().dt.days
    pre_holiday = pd.Series(False, index=cal)
    for i in range(1, len(cal)):
        # if the NEXT trading day is >1 calendar day later than a normal weekend would imply -> a holiday sits between
        d0, d1 = cal[i - 1], cal[i]
        span = (d1 - d0).days
        wd = d0.weekday()
        normal = 3 if wd == 4 else 1                                    # Fri->Mon = 3, else 1
        if span > normal:
            pre_holiday.iloc[i - 1] = True                             # d0 is the last session before a holiday
    spy_ret = spy["close"].pct_change()

    # ---- deployed flagship monthly returns ----
    d = json.load(open("/app/.data/studies/flagship_history.json"))
    mn = pd.DataFrame(d["monthly_net"], columns=["date", "ret"]); mn["date"] = pd.to_datetime(mn["date"])
    mn = mn.sort_values("date").reset_index(drop=True)

    def last_td(ts):                                                    # last trading day of the month of ts
        try:
            return last_td_of_month.loc[(ts.year, ts.month)]
        except KeyError:
            return pd.NaT

    rows = []
    for _, r in mn.iterrows():
        sell = last_td(r["date"])                                       # this month-end = SELL (and rebalance) day
        prev = (r["date"].replace(day=1) - pd.Timedelta(days=1))        # prior month
        buy = last_td(prev)                                             # prior month-end = BUY day
        rows.append(dict(ret=r["ret"], month=r["date"].month,
                         buy_short=bool(shortened.get(buy, False)) if buy is not pd.NaT else False,
                         sell_short=bool(shortened.get(sell, False)) if sell is not pd.NaT else False,
                         buy=buy, sell=sell))
    f = pd.DataFrame(rows)

    res = {"n_months": len(f)}
    # (1) buy in a shortened (4-day) week vs full week
    res["buy_week"] = [_grp("buy in FULL (5-day) week", f.loc[~f.buy_short, "ret"]),
                       _grp("buy in SHORTENED (<5) week", f.loc[f.buy_short, "ret"])]
    # (2) sell in a shortened week vs full week
    res["sell_week"] = [_grp("sell in FULL (5-day) week", f.loc[~f.sell_short, "ret"]),
                        _grp("sell in SHORTENED (<5) week", f.loc[f.sell_short, "ret"])]
    # (3) by calendar month (Dec-hold = bought Nov-end; Jan-hold = bought Dec-end are the holiday-adjacent buys)
    res["by_month"] = [_grp(f"month={m:02d}", f.loc[f.month == m, "ret"]) for m in range(1, 13)]
    # (4) holiday-adjacent buys: Dec + Jan holds (bought Nov-end / Dec-end) vs the rest
    hol = f.month.isin([12, 1])
    res["holiday_adjacent_hold"] = [_grp("Dec+Jan holds (bought around Thx/Xmas)", f.loc[hol, "ret"]),
                                    _grp("all other months", f.loc[~hol, "ret"])]
    # (5) SPY daily holiday effect
    res["spy_daily"] = [_grp("SPY normal day", spy_ret[~pre_holiday & ~shortened]),
                        _grp("SPY pre-holiday last session", spy_ret[pre_holiday]),
                        _grp("SPY any 4-day-week session", spy_ret[shortened])]
    res["all_months"] = _grp("ALL flagship months", f["ret"])

    print(json.dumps(res, indent=1, default=str))
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    try:
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="holiday_timing", defaults=dict(payload=res, computed_at=timezone.now()))
        print("saved BacktestResult 'holiday_timing'", flush=True)
    except Exception as e:
        print("BacktestResult save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
