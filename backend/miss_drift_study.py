"""
Downside PEAD validation — earnings MISSES keep underperforming (short/avoid signal).

The one robust effect that fell out of the under-reaction study: after an earnings MISS,
the stock earns negative ABNORMAL return (stock - beta*market) over the following 1-3 months.
Here we validate it as a directional signal the way everything else in this project is validated
(win rate + disaster rate), and check WHERE it's strongest (miss magnitude x mcap) and whether
it's monotonic in miss size (a robustness tell, not noise).

Entry = post-announcement close (rto). Forward return = RAW abnormal (not oriented):
  abn_k = (stock_ret over k days) - beta * (market_ret over k days)
For a SHORT: negative abn is the win; a large ADVERSE (up) move is the disaster.
  short_win%  = fraction with abn < 0
  short_dis%  = fraction with abn > +20%   (short blows up)
Beats and all-events shown as reference so we can see misses specifically underperform.

Run:  docker compose exec -T backend python -u miss_drift_study.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from core.models import EarningsEvent

DRIFTS = [21, 42, 63]
BETA_WIN = 60
MIN_PRICE = 3.0
MIN_ABS_SURPRISE = 0.5

def mcap_bucket(m):
    if m is None or not np.isfinite(m): return "unk"
    if m < 500e6:  return "micro"
    if m < 2e9:    return "small"
    if m < 10e9:   return "mid"
    return "large"

def miss_mag(s):                 # s < 0
    a = abs(s)
    if a < 5:   return "mild(0-5%)"
    if a < 15:  return "moderate(5-15%)"
    if a < 50:  return "severe(15-50%)"
    return "huge(50%+)"

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"])
    spy = cds.get("SPY")
    if spy is None or not len(spy):
        print("no SPY"); return
    spy_ret = spy["Close"].pct_change()
    print("loading financial reports ...", flush=True)
    reports = load_financial_reports(tks)

    ev_by_tk = defaultdict(list)
    for e in EarningsEvent.objects.filter(eps_surprise_pct__isnull=False).values(
            "ticker", "report_date", "eps_surprise_pct", "before_after"):
        ev_by_tk[e["ticker"]].append(e)

    events = []
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + max(DRIFTS) + 5:
            continue
        evs = ev_by_tk.get(tk)
        if not evs:
            continue
        idx = df.index; close = df["Close"].values; n = len(close)
        mkt = spy["Close"].reindex(idx).ffill().values
        r = df["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
        beta = (both["s"].rolling(BETA_WIN).cov(both["m"]) / both["m"].rolling(BETA_WIN).var()).values
        rep = reports.get(tk)
        if rep is not None and len(rep):
            r2 = rep.dropna(subset=["avail_date", "shares_outstanding"]).sort_values("avail_date")
            pdd = pd.DatetimeIndex(pd.to_datetime(r2["avail_date"]).to_numpy()) if len(r2) else None
            sh = r2["shares_outstanding"].to_numpy(float) if len(r2) else None
        else:
            pdd, sh = None, None
        for e in evs:
            s = e["eps_surprise_pct"]
            if s is None or abs(s) < MIN_ABS_SURPRISE:
                continue
            rd = pd.Timestamp(e["report_date"]); pos = int(idx.searchsorted(rd))
            # "After" and unknown/empty before_after (~19%) -> next-day reprice; only explicit BMO
            # uses the report day (empty->BMO was a lookahead — see earnings_drift_study).
            after = "Before" not in (e["before_after"] or "")
            rfrom, rto = (pos, pos + 1) if after else (pos - 1, pos)
            if rfrom < BETA_WIN or rto >= n or rto + max(DRIFTS) >= n:
                continue
            if close[rto] < MIN_PRICE or mkt[rto] <= 0:
                continue
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            m = (sh[max(0, int(pdd.searchsorted(rd, "right")) - 1)] * close[rto]
                 if (pdd is not None and len(pdd)) else None)
            abn = {}; ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                a = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * 100   # RAW abnormal (not oriented)
                if not np.isfinite(a):
                    ok = False; break
                abn[k] = a
            if not ok:
                continue
            events.append(dict(tk=tk, s=s, mb=mcap_bucket(m), abn=abn,
                               side="miss" if s < 0 else "beat"))

    print("usable events:", len(events), flush=True)

    def row(g, label):
        if len(g) < 40:
            print("  %-24s n=%d (thin)" % (label, len(g))); return
        cells = []
        for k in DRIFTS:
            a = np.array([e["abn"][k] for e in g])
            cells.append("+%dd abn=%+5.1f%% shWin=%2d%% shDis=%2d%%" % (
                k, np.median(a), round((a < 0).mean()*100), round((a > 20).mean()*100)))
        print("  %-24s n=%-5d %s" % (label, len(g), " | ".join(cells)))

    misses = [e for e in events if e["side"] == "miss"]
    beats  = [e for e in events if e["side"] == "beat"]
    print("\n(abn = median forward ABNORMAL return; shWin = %% negative = short works; shDis = %% > +20%% = short blows up)")
    print("\n=== REFERENCE ===")
    row(events, "ALL events")
    row(beats,  "BEATS")
    row(misses, "MISSES (the signal)")
    print("\n=== MISSES by magnitude (is it monotonic? = robustness tell) ===")
    for mag in ("mild(0-5%)", "moderate(5-15%)", "severe(15-50%)", "huge(50%+)"):
        row([e for e in misses if miss_mag(e["s"]) == mag], mag)
    print("\n=== MISSES by mcap ===")
    for mb in ("micro", "small", "mid", "large"):
        row([e for e in misses if e["mb"] == mb], "mcap=" + mb)
    print("\n=== SEVERE+ misses (<-15%) by mcap (best short candidates?) ===")
    sev = [e for e in misses if abs(e["s"]) >= 15]
    for mb in ("micro", "small", "mid", "large"):
        row([e for e in sev if e["mb"] == mb], "sev mcap=" + mb)

if __name__ == "__main__":
    main()
