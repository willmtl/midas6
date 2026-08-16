#!/usr/bin/env python3
"""H4 (4-hour) short-horizon studies engine. See docs/superpowers/specs/2026-08-16-h4-short-horizon-engine-design.md.
Signal registry × bar-based exit ladder over the liquid top-250 universe; magnitude-bucketed
(tail-not-average); daily-trend split; daily-candle benchmark; saved to BacktestResult[h4_study].
Pure signal/agg code imports without Django; main() calls django.setup()."""
import numpy as np
import pandas as pd

RTH_HOURS = 6.5
TF_HOURS = 4
GAP = 3                                   # episode-dedup gap in bars
FIXED_BARS = [1, 2, 3, 4, 5, 6, 8, 10]    # 0-3 day focus (~½ day → ~6 days on 4h)


def day_label(bars):
    days = bars * TF_HOURS / RTH_HOURS
    return f"~{days:.1f}d" if days >= 1 else "~½ day"
