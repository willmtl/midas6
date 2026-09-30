#!/usr/bin/env python
"""LIVE cross-scan: every stock whose RSI(14)-of-rolling-Sortino just crossed UP its EMA recently.
FIXED 2026-09-28: indicators are computed PER-TICKER on each name's OWN trading calendar. The previous
version computed them on the UNION calendar of all 1451 names (1991-2026, 8980 days incl. foreign-market
days), which injected NaN gaps into every US series and corrupted the rolling Sortino/RSI (e.g. GOOG's
Sortino-RSI got stuck flat ~9 and its real cross was invisible). Own-calendar == what TradingView plots.
  CROSS_WIN(10) bars back = "recent";  MIN_MCAP_B(1.0);  STALE_DAYS(5);  MKT(all|us|foreign)
"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

CROSS_WIN  = int(os.environ.get("CROSS_WIN", 10))
MIN_MCAP_B = float(os.environ.get("MIN_MCAP_B", 1.0))
STALE_DAYS = int(os.environ.get("STALE_DAYS", 5))        # ticker's last bar must be within this many calendar days of the freshest bar
TOPN       = int(os.environ.get("TOPN", 0))
MKT        = os.environ.get("MKT", "all")

b = S.pickle.load(open(S.PANELS, "rb"))
universe = list(b["common"]); mcap_m = b.get("mktcap_usd")
del_sector = b["delisted_sector"]; surv_sector = b["surv_sector"]
sec_of = {t: (surv_sector.get(t) or del_sector.get(t)) for t in universe}
cd = S.load_candles(universe)
mcap_last = {}
if mcap_m is not None:
    for t in universe:
        if t in mcap_m.columns:
            s = mcap_m[t].dropna()
            if len(s): mcap_last[t] = float(s.iloc[-1])

# freshest bar across the universe (to judge staleness)
global_last = None
for t in universe:
    d = cd.get(t)
    if d is None or "Close" not in d: continue
    s = d["Close"].dropna()
    if len(s):
        gl = s.index[-1]
        global_last = gl if global_last is None else max(global_last, gl)

rows = []
for tk in universe:
    d = cd.get(tk)
    if d is None or "Close" not in d: continue
    c = d["Close"].dropna(); c = c[c > 0]
    if len(c) < 260: continue                                  # need a year+ of own history
    if global_last is not None and (global_last - c.index[-1]).days > STALE_DAYS:
        continue                                               # stale name (not currently trading)
    is_us = "." not in tk
    if MKT == "us" and not is_us: continue
    if MKT == "foreign" and is_us: continue
    idx = c.index
    o = d["Open"].reindex(idx).ffill(); h = d["High"].reindex(idx).ffill()
    l = d["Low"].reindex(idx).ffill(); v = d["Volume"].reindex(idx).fillna(0.0)
    ohlc4 = (o + h + l + c) / 4.0
    ind = S.compute_indicators(c.to_frame(tk), ohlc4.to_frame(tk))     # PER-TICKER, own calendar
    srsi = ind["srsi"][tk].to_numpy(); se = ind["srsi_ema"][tk].to_numpy()
    dd = ind["dd_score"][tk].to_numpy(); ulc = ind["ulcer"][tk].to_numpy(); nr = ind["nrsi"][tk].to_numpy()
    above = srsi > se
    n = len(c); li = n - 1
    # most recent bullish cross within CROSS_WIN bars, still above now
    bac = None; cross_i = None
    for k in range(0, CROSS_WIN + 1):
        i = li - k
        if i < 1: break
        if above[i] and not above[i - 1]:
            bac = k; cross_i = i; break
    if bac is None or not above[li]:
        continue
    mc = mcap_last.get(tk, np.nan); mcb = mc / 1e9 if np.isfinite(mc) else np.nan
    if not (np.isfinite(mcb) and mcb >= MIN_MCAP_B):
        continue
    dv = float((c * v).rolling(63, min_periods=20).mean().iloc[-1])
    if not (np.isfinite(dv) and dv >= S.MINVOL):
        continue
    cap = S.cap_bucket(mc)
    ddv, ulv, srv, sev, nrv = dd[li], ulc[li], srsi[li], se[li], nr[li]
    is_buy = np.isfinite(ddv) and ddv > 0
    tail = is_buy and S.tuned_pass(ddv, ulv, bac, cap, srv, nrv, None)
    rows.append(dict(ticker=tk, market=("US" if is_us else "FX"), as_of=str(idx[li].date()),
                     sector=sec_of.get(tk) or "?", cap=cap, mcap_b=round(mcb, 1),
                     bars_since_cross=bac, cross_date=str(idx[cross_i].date()),
                     srsi=round(srv, 1), srsi_ema=round(sev, 1), dd=round(ddv, 2) if np.isfinite(ddv) else None,
                     ulcer=round(ulv, 1) if np.isfinite(ulv) else None,
                     ohlc4_rsi=round(nrv, 1) if np.isfinite(nrv) else None,
                     dvol_m=round(dv / 1e6, 1), full_buy=bool(is_buy), tail_target=bool(tail)))

COLS0 = ["ticker", "market", "as_of", "sector", "cap", "mcap_b", "bars_since_cross", "cross_date",
         "srsi", "srsi_ema", "dd", "ulcer", "ohlc4_rsi", "dvol_m", "full_buy", "tail_target"]
df = pd.DataFrame(rows, columns=COLS0) if rows else pd.DataFrame(columns=COLS0)
if len(df):
    df = df.sort_values(["tail_target", "full_buy", "srsi"], ascending=[False, False, True])
asof = str(global_last.date()) if global_last is not None else "n/a"
print(f"=== Sortino-RSI(14) recent UP-crosses  (own-calendar; as of {asof}, within {CROSS_WIN} bars, "
      f"${MIN_MCAP_B:g}B+, $vol>=${S.MINVOL/1e6:g}M) ===")
print(f"universe scanned: {len(universe)}   |   recent crosses (still above): {len(df)}   "
      f"|   full Pine BUYs (dd>0): {int(df.full_buy.sum()) if len(df) else 0}   "
      f"tail-targets: {int(df.tail_target.sum()) if len(df) else 0}\n")
if len(df):
    show = df if TOPN == 0 else df.head(TOPN)
    hdr = f"{'ticker':>8}{'mkt':>4}{'sector':>14}{'cap':>6}{'mcap$B':>8}{'bars':>5}{'crossed':>12}{'srsi':>6}{'>ema':>6}{'dd':>6}{'ulcer':>6}{'o4rsi':>6}{'$Mvol':>7}  flags"
    print(hdr); print("-" * len(hdr))
    for _, r in show.iterrows():
        fl = ("TAIL" if r.tail_target else ("BUY" if r.full_buy else "cross-only"))
        print(f"{r.ticker:>8}{r.market:>4}{str(r.sector)[:13]:>14}{r.cap:>6}{r.mcap_b:>8.1f}{r.bars_since_cross:>5}{r.cross_date:>12}"
              f"{r.srsi:>6.0f}{r.srsi_ema:>6.0f}{(r.dd if r.dd is not None else float('nan')):>6.2f}"
              f"{(r.ulcer or 0):>6.1f}{(r.ohlc4_rsi or 0):>6.0f}{r.dvol_m:>7.0f}  {fl}")
out = "/app/.data/studies/sortino_cross_scan.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}  ({len(df)} rows)")
