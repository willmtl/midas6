"""
Earnings drift study -- IV-IMPLIED expected-move benchmark.

Same event mechanic as earnings_drift_study.py, but the "expected move" is the
market's OWN ex-ante priced move from options, not an empirical peer median:

  E_iv       = (atm_iv/100) * sqrt(21/252)      # 30d ATM IV -> ~1-month expected move
  fill_ratio = |abnormal reaction| / E_iv       # how much of the priced move the news actually delivered
                 LOW  -> stock moved LESS than options priced -> UNDER-reaction -> hypothesis: drift to come
                 HIGH -> moved MORE than priced               -> OVER-reaction  -> hypothesis: fade

IV is taken as-of the last snapshot on/before the pre-announcement day (point-in-time, no lookahead).
Forward drift = oriented abnormal return (dir * (stock - beta*market)) at +21/+42/+63 trading days.

Run:  docker compose exec -T backend python -u earnings_drift_iv.py
"""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from core.models import EarningsEvent, OptionSnapshot

DRIFTS = [21, 42, 63]
BETA_WIN = 60
MIN_PRICE = 3.0
MIN_ABS_SURPRISE = 0.5
IV_HORIZON = 21.0        # trading days the 30d IV is scaled to (≈ 1 month, matches drift window)

def mcap_bucket(m):
    if m is None or not np.isfinite(m): return "unk"
    if m < 500e6:  return "micro"
    if m < 2e9:    return "small"
    if m < 10e9:   return "mid"
    return "large"

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"])
    spy = cds.get("SPY")
    if spy is None or not len(spy):
        print("no SPY candles"); return
    spy_ret = spy["Close"].pct_change()
    print("loading financial reports ...", flush=True)
    reports = load_financial_reports(tks)

    # IV history per ticker as a date-indexed Series (for as-of lookup)
    print("loading options IV history ...", flush=True)
    iv_rows = defaultdict(list)
    for o in OptionSnapshot.objects.filter(atm_iv__isnull=False).values("ticker", "date", "atm_iv"):
        iv_rows[o["ticker"]].append((o["date"], o["atm_iv"]))
    iv_series = {}
    for tk, rows in iv_rows.items():
        rows.sort()
        iv_series[tk] = pd.Series([r[1] for r in rows],
                                  index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows]))
    print("tickers with IV:", len(iv_series), flush=True)

    ev_by_tk = defaultdict(list)
    for e in EarningsEvent.objects.filter(eps_surprise_pct__isnull=False).values(
            "ticker", "report_date", "eps_surprise_pct", "before_after"):
        ev_by_tk[e["ticker"]].append(e)

    events = []
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + max(DRIFTS) + 5:
            continue
        ivs = iv_series.get(tk)
        if ivs is None or not len(ivs):
            continue
        evs = ev_by_tk.get(tk)
        if not evs:
            continue
        idx = df.index
        close = df["Close"].values
        n = len(close)
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
            rd = pd.Timestamp(e["report_date"])
            pos = int(idx.searchsorted(rd))
            after = "After" in (e["before_after"] or "")
            rfrom, rto = (pos, pos + 1) if after else (pos - 1, pos)
            if rfrom < BETA_WIN or rto >= n or rto + max(DRIFTS) >= n:
                continue
            if close[rto] < MIN_PRICE or close[rfrom] <= 0 or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            # IV as-of the pre-announcement day (no lookahead)
            iv = ivs.asof(idx[rfrom])
            if iv is None or not np.isfinite(iv) or iv <= 0:
                continue
            E_iv = (iv / 100.0) * np.sqrt(IV_HORIZON / 252.0)
            if E_iv <= 0:
                continue
            d = np.sign(s)
            b = beta[rfrom]; bc = min(max(b if np.isfinite(b) else 1.0, 0.0), 3.0)
            react_mkt = mkt[rto] / mkt[rfrom] - 1.0
            R_abn = (close[rto] / close[rfrom] - 1.0) - bc * react_mkt   # abnormal reaction (unoriented)
            fill = abs(R_abn) / E_iv
            m = (sh[max(0, int(pdd.searchsorted(rd, "right")) - 1)] * close[rto]
                 if (pdd is not None and len(pdd)) else None)
            drift = {}; ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                fd = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * d
                if not np.isfinite(fd):
                    ok = False; break
                drift[k] = fd * 100
            if not ok:
                continue
            events.append(dict(tk=tk, react=idx[rto], fill=fill, iv=iv,
                               dir=int(d), mb=mcap_bucket(m), drift=drift))

    print("usable events with IV:", len(events), flush=True)
    if len(events) < 200:
        print("too few"); return

    def report(sub, label):
        if len(sub) < 60:
            print("  %-22s n=%d (too few)" % (label, len(sub))); return
        fills = np.array([e["fill"] for e in sub])
        lo, hi = np.quantile(fills, [1/3, 2/3])
        cohorts = [
            ("UNDER (fill<=%.2f)" % lo, [e for e in sub if e["fill"] <= lo]),
            ("MID",                     [e for e in sub if lo < e["fill"] < hi]),
            ("OVER  (fill>=%.2f)" % hi, [e for e in sub if e["fill"] >= hi]),
        ]
        print("  %s (n=%d, median fill=%.2f)" % (label, len(sub), np.median(fills)))
        for nm, g in cohorts:
            if len(g) < 20:
                print("      %-20s n=%d (thin)" % (nm, len(g))); continue
            cells = []
            for k in DRIFTS:
                a = np.array([e["drift"][k] for e in g])
                cells.append("+%dd med=%+5.1f%% win=%2d%% dis=%2d%%" % (
                    k, np.median(a), round((a > 0).mean()*100), round((a < -20).mean()*100)))
            print("      %-20s n=%-5d %s" % (nm, len(g), " | ".join(cells)))

    print("\n=== ALL (by IV fill_ratio; UNDER = moved less than options priced = hypothesis: drift) ===")
    report(events, "ALL")
    print("\n=== BEATS only ===")
    report([e for e in events if e["dir"] > 0], "beats")
    print("\n=== MISSES only ===")
    report([e for e in events if e["dir"] < 0], "misses")
    for mb in ("micro", "small", "mid", "large"):
        sub = [e for e in events if e["mb"] == mb]
        if len(sub) >= 120:
            print("\n=== mcap=%s ===" % mb)
            report(sub, "mcap=%s" % mb)

if __name__ == "__main__":
    main()
