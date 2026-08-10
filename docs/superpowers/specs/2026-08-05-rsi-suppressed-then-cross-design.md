# RSI Suppressed-Then-Cross Study — Design

**Date:** 2026-08-05
**Status:** Approved (design)

## Goal

Add a new entry signal to the studies engine: RSI must be *suppressed* — below 50
**and** below its own SMA(10) — for **10 or more consecutive trading days**, and then
RSI crosses **above** its SMA(10). The signal fires on the crossover bar. This captures
a "coiled spring" reversal: momentum held down and below its average for a stretch, then
turns up.

## Rule (strict consecutive)

On each bar, using RSI(10) and its SMA(10) (the codebase standard window):

- `suppressed[t]` = `RSI[t] < 50` AND `RSI[t] < SMA(10)[t]`
- `cross[t]` = `RSI[t] > SMA[t]` AND `RSI[t-1] <= SMA[t-1]` (RSI crosses above its average)
- Signal fires at `t` when `cross[t]` AND the suppression streak on the **bar before the
  cross** (`t-1`) was **>= 10 consecutive days**.

"The average" = RSI's own SMA(10), i.e. the same line the RSI crosses. The streak must be
unbroken; a single non-suppressed day resets it.

## Implementation

In `studies.py`:

```python
SUPPRESS_MIN = 10  # min consecutive days RSI stayed <50 AND <its SMA before the cross

def sig_rsi_suppressed_then_cross(df):
    """RSI(10) held <50 AND <its SMA(10) for SUPPRESS_MIN+ straight days, then
    crosses above its SMA(10) on that bar."""
    rsi = ta.momentum.rsi(df["Close"], window=10)
    sma = rsi.rolling(10).mean()
    cross = (rsi > sma) & (rsi.shift(1) <= sma.shift(1))
    suppressed = (rsi < 50) & (rsi < sma)
    grp = (~suppressed).cumsum()                 # new group each time suppression breaks
    streak = suppressed.groupby(grp).cumsum()    # consecutive suppressed-day count
    return (cross & (streak.shift(1) >= SUPPRESS_MIN)).fillna(False).astype(bool)
```

Register in `SIGNALS`:

```python
"rsi_sup10_x": ("RSI <50 & <avg 10d+ then Cross", sig_rsi_suppressed_then_cross),
```

The signal takes only `df` (per-sector, no SPY/QQQ market inputs), so it needs no special
handling in `run_study` — it flows through `generate_studies()` and crosses with all 70
EXITS automatically, adding 70 new study combinations. Daily (D) timeframe, matching all
studies.

## Testing

Standalone verification (synthetic frame): construct a Close series that drives RSI below
50 and below its SMA for >= 10 straight bars, then rallies to force the cross, and assert
the signal fires on exactly the cross bar and nowhere it shouldn't (e.g. a cross with a
< 10-day streak does not fire). Run in the backend container against real DB candles as a
sanity check on trigger counts.

## Out of scope

- Dashboard / `compute_scan` (still uses old RSI+Omega logic; unaffected).
- Weekly variant (can be added later if the daily version looks promising).
- Report columns.
