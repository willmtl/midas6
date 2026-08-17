#!/usr/bin/env python3
"""H4-on-daily-signals: gate the H4 engine by point-in-time candidate windows from daily systems
A (pure dip), B (capitulation), C (div_2x flagship). See
docs/superpowers/specs/2026-08-16-h4-on-daily-signals-design.md. Reuses h4_study + intraday_data;
modifies nothing. Pure functions import without Django; main() calls django.setup()."""
import numpy as np
import h4_study as H


def backtest_ticker_masked(df, allowed_dates=None):
    """Like h4_study.backtest_ticker but only counts entries whose bar date is in `allowed_dates`
    (a set of datetime.date). allowed_dates=None allows every bar (== h4_study.backtest_ticker)."""
    close = df["Close"].values
    n = len(close)
    dates = df.index.normalize()
    out = {}
    for sig, meta in H.SIGNALS.items():
        entry, mag = meta["fn"](df)
        buckets = meta["buckets"]
        cand = [i for i in range(n) if entry[i] and
                (allowed_dates is None or dates[i].date() in allowed_dates)]
        idxs = sorted(H._episode_starts(cand, gap=H.GAP))
        flat = H._empty_exit_pool(sig)
        by_bucket = {b[0]: H._empty_exit_pool(sig) for b in buckets}
        dn_exit = H._rsi_x_dn_exit(df) if meta.get("exit_fn") == "rsi_x_dn" else None
        for i in idxs:
            ep = float(close[i])
            if ep <= 0:
                continue
            blab = H.bucket_of(mag[i], buckets)
            for k, bars, _ in H.EXITS:
                j = i + bars
                if j < n:
                    r = (close[j] - ep) / ep * 100
                    flat[k].append(r)
                    if blab is not None:
                        by_bucket[blab][k].append(r)
            if dn_exit is not None:
                j = next((q for q in range(i + 1, n) if dn_exit[q]), None)
                if j is not None:
                    r = (close[j] - ep) / ep * 100
                    flat["rsi_x_dn"].append(r)
                    if blab is not None:
                        by_bucket[blab]["rsi_x_dn"].append(r)
        out[sig] = {"flat": flat, "by_bucket": by_bucket}
    return out
