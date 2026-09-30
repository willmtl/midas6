#!/usr/bin/env python
"""CLEAN (per-ticker own-calendar) backtest of every Sortino full-BUY, comparing TAIL-targets
(Ulcer>=5, 0<DD<=2.5) vs the other full BUYs. Round-trip = entry close -> exit close (Pine state machine).
Avoids the union-calendar bug that corrupts US names. $1B+ tradeable, $vol>=5M."""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
os.environ["MKT_GATE"] = "0"
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

S.ULCER_MIN = 0.0                     # full BUY = cross + dd>0 (ulcer gate handled by tail flag)
WARM = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
HZ = [5, 10, 21, 42, 63]
b = S.pickle.load(open(S.PANELS, "rb"))
uni = list(b["common"]); mcap_m = b.get("mktcap_usd")
bankrupt = set(b["bankrupt_tk"])
cd = S.load_candles(uni)
rows = []
for tk in uni:
    d = cd.get(tk)
    if d is None or "Close" not in d: continue
    c = d["Close"].dropna(); c = c[c > 0]
    if len(c) < WARM + 30: continue
    idx = c.index
    o = d["Open"].reindex(idx).ffill(); h = d["High"].reindex(idx).ffill()
    l = d["Low"].reindex(idx).ffill(); v = d["Volume"].reindex(idx).fillna(0.0)
    ohlc4 = (o + h + l + c) / 4.0
    ind = S.compute_indicators(c.to_frame(tk), ohlc4.to_frame(tk))
    srsi = ind["srsi"][tk].to_numpy(); se = ind["srsi_ema"][tk].to_numpy()
    nr = ind["nrsi"][tk].to_numpy(); ne = ind["nrsi_ema"][tk].to_numpy()
    dd = ind["dd_score"][tk].to_numpy(); ulc = ind["ulcer"][tk].to_numpy()
    cl = c.to_numpy(); valid = np.isfinite(cl)
    dvol = (c * v).rolling(63, min_periods=20).mean().to_numpy()
    mc = None
    if mcap_m is not None and tk in mcap_m.columns:
        mc = mcap_m[tk].reindex(idx, method="ffill").to_numpy()
    trips = S.run_state_machine(0, cl, srsi, se, nr, ne, dd, ulc, valid, WARM)
    isb = tk in bankrupt
    for (ei, xi, reason, bac) in trips:
        if xi < 0: xi = len(cl) - 1                      # still open -> mark to last bar
        if not (dvol[ei] >= S.MINVOL): continue
        mcv = mc[ei] if mc is not None else np.nan
        if not (np.isfinite(mcv) and mcv >= 1e9): continue
        cap = S.cap_bucket(mcv)
        ulv, ddv, srv, nrv = ulc[ei], dd[ei], srsi[ei], nr[ei]
        tail = bool(S.tuned_pass(ddv, ulv, bac, cap, srv, nrv, None))
        rt = (cl[xi] / cl[ei] - 1.0) * 100.0
        rec = dict(ticker=tk, cap=cap, ulcer=ulv, dd=ddv, is_tail=tail, rt=rt, hold=xi - ei)
        for hz in HZ:
            j = ei + hz
            rec[f"f{hz}"] = (cl[j] / cl[ei] - 1.0) * 100.0 if j < len(cl) and cl[j] > 0 else np.nan
        rows.append(rec)

t = pd.DataFrame(rows)
print(f"CLEAN per-ticker Sortino BUYs ($1B+, liquid): {len(t)} entries, {t.ticker.nunique()} names\n")

def stat(df, lab):
    rt = df.rt.values
    print(f"{lab:<26}{len(df):>6}  rt mean {np.mean(rt):>+6.2f}%  median {np.median(rt):>+6.2f}%  "
          f"win {np.mean(rt>0)*100:>4.0f}%  big(>=50%) {np.mean(rt>=50)*100:>4.1f}%  "
          f"hold {df.hold.mean():>3.0f}d")

stat(t, "ALL full BUYs")
stat(t[t["is_tail"]], "  TAIL-targets")
stat(t[~t["is_tail"]], "  non-tail BUYs")
print()
print("forward returns (mean %):   " + "  ".join(f"f{h}" for h in HZ))
for lab, df in [("ALL BUYs", t), ("TAIL", t[t["is_tail"]]), ("non-tail", t[~t["is_tail"]])]:
    print(f"  {lab:<10}" + "  ".join(f"{df[f'f{h}'].mean():>+5.1f}" for h in HZ))
# tail lift = tail vs non-tail
print()
tl, nt = t[t["is_tail"]], t[~t["is_tail"]]
print(f"TAIL vs non-tail BUY:  median rt {np.median(tl.rt):+.2f}% vs {np.median(nt.rt):+.2f}%  |  "
      f"win {np.mean(tl.rt>0)*100:.0f}% vs {np.mean(nt.rt>0)*100:.0f}%  |  "
      f"big% {np.mean(tl.rt>=50)*100:.1f} vs {np.mean(nt.rt>=50)*100:.1f}")
t.to_csv("/app/.data/studies/sortino_clean_entries.csv", index=False)
print(f"\nwrote /app/.data/studies/sortino_clean_entries.csv ({len(t)} rows)")
