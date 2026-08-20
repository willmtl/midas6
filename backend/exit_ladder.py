#!/usr/bin/env python3
"""Exit-horizon ladder for the validated short-horizon setups: hold 1..10 bars instead of only 3. Capitulation
reverts slowly (w15 window won) so longer holds likely pay more; momentum continuation is front-loaded.
Reports avg%/win%/t at each exit bar for A(gap-up momentum) and B(capitulation w15) x their best signals.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/exit_ladder.py"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import numpy as np
import h4_on_signals_study as S
import h4_study as H
from seq_fundamental_study import load_candles
from studies import SIGNALS as STUDY_SIGNALS
from intraday_data import get_4h

_name, seq_fn = STUDY_SIGNALS["seq_rsi20_ad_rising_rsi"]
EXITS = [(k, b) for (k, b, _x) in H.EXITS]                 # (key, bars)


def _t(a):
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def build_A(daily):
    """gap-up >=5% momentum on high-vol liquid, 5-bar candidate window."""
    allowed = {}
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        c = df["Close"]; idx = df.index
        vol = c.pct_change().rolling(20).std() * (252 ** 0.5)
        dvol = (c * df["Volume"]).rolling(20).mean()
        gap = c / c.shift(1) - 1.0
        ok = (gap >= 0.05) & vol.between(0.5, 3.0) & (dvol >= 5e6)
        s = set()
        for i in np.flatnonzero(ok.values):
            for j in range(i, min(i + 5, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def build_B(daily):
    """capitulation seq + $5M floor, 15-bar candidate window."""
    allowed = {}
    for tk, df in daily.items():
        if len(df) < 60:
            continue
        sig = seq_fn(df).fillna(False); idx = df.index
        dvol = (df["Close"] * df["Volume"]).rolling(20).mean()
        s = set()
        for i in np.flatnonzero(sig.values):
            if dvol.iloc[i] < 5e6:
                continue
            for j in range(i, min(i + 15, len(idx))):
                s.add(idx[j].date())
        if s:
            allowed[tk] = s
    return allowed


def ladder(allowed, frames, sigs, label):
    print(f"\n=== {label} — exit ladder (avg% / win% / t) ===", flush=True)
    print(f"  {'signal':12}" + "".join(f"{str(b)+'b':>16}" for _, b in EXITS), flush=True)
    for s in sigs:
        pool = {k: [] for k, _ in EXITS}
        for tk, dates in allowed.items():
            df = frames.get(tk)
            if df is None:
                continue
            res = S.backtest_ticker_masked(df, dates)
            for k, _b in EXITS:
                pool[k].extend(res[s]["flat"].get(k, []))
        row = f"  {s:12}"
        for k, _b in EXITS:
            a = np.asarray(pool[k], float); a = a[np.isfinite(a)]
            row += f"{(f'{a.mean():+.2f}/{(a>0).mean()*100:.0f}/{_t(a)}' if len(a)>=20 else '—'):>16}"
        print(row, flush=True)


def main():
    daily = load_candles(S._stock_universe())
    aw, bw = build_A(daily), build_B(daily)
    names = sorted(set(aw) | set(bw))
    frames = {}
    for tk in names:
        df = get_4h(tk, 5, False)
        if df is not None and len(df) >= 120:
            frames[tk] = df
    print(f"4h cached {len(frames)}/{len(names)}; A names {len(aw)} B names {len(bw)}", flush=True)
    ladder(aw, frames, ["mo_gap_up", "mo_burst", "mo_rsi_ob"], "A: gap-up momentum (high-vol)")
    ladder(bw, frames, ["mo_gap_up", "mr_ndown", "st_ad_div", "mr_newlow60"], "B: capitulation w15")


if __name__ == "__main__":
    main()
