"""
Earnings PEAD under-/over-reaction (drift) study.

For each earnings event we measure:
  - reaction R  : the actual oriented return over the announcement window
                  (close before -> close after), oriented by the sign of the EPS surprise
                  (a beat "should" push price up, a miss down).
  - expected E  : the TYPICAL oriented reaction for a peer event -- same
                  (surprise-magnitude bucket x mcap bucket x beta bucket) -- built
                  POINT-IN-TIME from PRIOR events only (expanding baseline, no lookahead).
                  mcap & beta are measured as of the event date.
  - residual    : R - E.
                    residual < 0  -> UNDER-reaction (moved less than peers) -> hypothesis: more drift to come.
                    residual > 0  -> OVER-reaction  (moved more)            -> hypothesis: fade / reversion.
  - forward drift: oriented return from the post-announcement close over the next
                   +21 / +42 / +63 trading days -- where any under-reaction plays out.

We then check whether the UNDER-reaction cohort earns higher forward drift than the
OVER-reaction cohort (win rate + disaster rate = fraction of drifts < -20%).

Run:  docker compose exec -T backend python -u earnings_drift_study.py
"""
import django, os, datetime as dt
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from seq_fundamental_study import build_universe, load_candles, load_financial_reports
from core.models import EarningsEvent

DRIFTS = [21, 42, 63]
BETA_WIN = 60
MIN_PRIOR = 20          # min prior peer events required to trust a baseline bucket
MIN_PRICE = 3.0
MIN_ABS_SURPRISE = 0.5  # ignore ~zero surprises (no directional information), in %

def mcap_bucket(m):
    if m is None or not np.isfinite(m): return "unk"
    if m < 500e6:  return "micro"
    if m < 2e9:    return "small"
    if m < 10e9:   return "mid"
    return "large"

def beta_bucket(b):
    if b is None or not np.isfinite(b): return "unk"
    if b < 0.8:  return "lobeta"
    if b <= 1.2: return "midbeta"
    return "hibeta"

def surp_bucket(s):          # s = signed surprise %, bucket on magnitude
    a = abs(s)
    if a < 5:   return "s0"
    if a < 15:  return "s1"
    if a < 50:  return "s2"
    return "s3"

def main():
    tks = build_universe()
    print("loading candles + SPY ...", flush=True)
    cds = load_candles(tks + ["SPY"])
    spy = cds.get("SPY")
    if spy is None or not len(spy):
        print("no SPY candles"); return
    spy_ret = spy["Close"].pct_change()
    print("loading financial reports (PIT shares) ...", flush=True)
    reports = load_financial_reports(tks)

    # pull earnings events once, grouped by ticker
    ev_by_tk = defaultdict(list)
    for e in EarningsEvent.objects.filter(eps_surprise_pct__isnull=False).values(
            "ticker", "report_date", "eps_surprise_pct", "before_after"):
        ev_by_tk[e["ticker"]].append(e)

    events = []   # dict per usable event, collected across all tickers, then sorted by react date
    for tk, df in cds.items():
        if tk == "SPY" or df is None or len(df) < BETA_WIN + max(DRIFTS) + 5:
            continue
        evs = ev_by_tk.get(tk)
        if not evs:
            continue
        idx = df.index
        close = df["Close"].values
        n = len(close)
        # market close aligned to this ticker's trading days (for abnormal/market-adjusted returns)
        mkt = spy["Close"].reindex(idx).ffill().values
        # rolling beta vs SPY (aligned daily returns)
        r = df["Close"].pct_change()
        both = pd.concat([r.rename("s"), spy_ret.rename("m")], axis=1).reindex(idx)
        cov = both["s"].rolling(BETA_WIN).cov(both["m"])
        var = both["m"].rolling(BETA_WIN).var()
        beta = (cov / var).values
        # PIT shares
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
            pos = int(idx.searchsorted(rd))          # first trading idx >= report_date
            after = "After" in (e["before_after"] or "")
            if after:
                rfrom, rto = pos, pos + 1             # reported after close -> next day reprices
            else:
                rfrom, rto = pos - 1, pos             # before open / unknown -> report day reprices
            if rfrom < BETA_WIN or rto >= n or rto + max(DRIFTS) >= n:
                continue
            if close[rto] < MIN_PRICE or close[rfrom] <= 0 or mkt[rfrom] <= 0 or mkt[rto] <= 0:
                continue
            d = np.sign(s)                            # beat=+1 miss=-1
            b = beta[rfrom]
            bc = b if np.isfinite(b) else 1.0
            bc = min(max(bc, 0.0), 3.0)               # clip beta to a sane range
            # oriented ABNORMAL reaction = dir * (stock return - beta * market return)
            react_mkt = mkt[rto] / mkt[rfrom] - 1.0
            R = ((close[rto] / close[rfrom] - 1.0) - bc * react_mkt) * d
            if not np.isfinite(R):
                continue
            m = (sh[max(0, int(pdd.searchsorted(rd, "right")) - 1)] * close[rto]
                 if (pdd is not None and len(pdd)) else None)
            drift = {}
            ok = True
            for k in DRIFTS:
                dm = mkt[rto + k] / mkt[rto] - 1.0
                fd = ((close[rto + k] / close[rto] - 1.0) - bc * dm) * d   # oriented abnormal drift
                if not np.isfinite(fd):
                    ok = False; break
                drift[k] = fd * 100
            if not ok:
                continue
            events.append(dict(
                tk=tk, react=idx[rto], R=R,
                sb=surp_bucket(s), mb=mcap_bucket(m), bb=beta_bucket(b),
                dir=int(d), drift=drift))

    print("usable events:", len(events), flush=True)
    if len(events) < 200:
        print("too few events"); return

    # POINT-IN-TIME expanding peer baseline of oriented reaction R
    events.sort(key=lambda x: x["react"])
    running = defaultdict(list)   # bucket-key -> list of prior oriented R
    def baseline(ev):
        # finest bucket with enough prior history, falling back to coarser
        for key in ((ev["sb"], ev["mb"], ev["bb"]), (ev["sb"], ev["mb"]), (ev["sb"],), ("ALL",)):
            hist = running[key]
            if len(hist) >= MIN_PRIOR:
                return float(np.median(hist)), key
        return None, None
    for ev in events:
        E, key = baseline(ev)
        ev["E"] = E
        ev["resid"] = (ev["R"] - E) if E is not None else None
        # update running buckets AFTER computing (so baseline uses only prior events)
        for k in ((ev["sb"], ev["mb"], ev["bb"]), (ev["sb"], ev["mb"]), (ev["sb"],), ("ALL",)):
            running[k].append(ev["R"])

    scored = [e for e in events if e["resid"] is not None]
    print("events with a PIT baseline:", len(scored), flush=True)

    def report(sub, label):
        if len(sub) < 60:
            print("  %-26s n=%d (too few)" % (label, len(sub))); return
        resids = np.array([e["resid"] for e in sub])
        lo, hi = np.quantile(resids, [1/3, 2/3])
        cohorts = [
            ("UNDER (resid<=%.3f)" % lo, [e for e in sub if e["resid"] <= lo]),
            ("MID",                       [e for e in sub if lo < e["resid"] < hi]),
            ("OVER  (resid>=%.3f)" % hi,  [e for e in sub if e["resid"] >= hi]),
        ]
        print("  %s (n=%d)" % (label, len(sub)))
        for nm, g in cohorts:
            if len(g) < 20:
                print("      %-22s n=%d (thin)" % (nm, len(g))); continue
            cells = []
            for k in DRIFTS:
                a = np.array([e["drift"][k] for e in g])
                cells.append("+%dd med=%+5.1f%% win=%2d%% dis=%2d%%" % (
                    k, np.median(a), round((a > 0).mean()*100), round((a < -20).mean()*100)))
            print("      %-22s n=%-5d %s" % (nm, len(g), " | ".join(cells)))

    print("\n=== ALL events by reaction residual (UNDER = under-reacted = hypothesis: more drift) ===")
    report(scored, "ALL")
    print("\n=== BEATS only (surprise > 0) ===")
    report([e for e in scored if e["dir"] > 0], "beats")
    print("\n=== MISSES only (surprise < 0) ===")
    report([e for e in scored if e["dir"] < 0], "misses")
    for mb in ("micro", "small", "mid", "large"):
        sub = [e for e in scored if e["mb"] == mb]
        if len(sub) >= 120:
            print("\n=== mcap=%s ===" % mb)
            report(sub, "mcap=%s" % mb)

if __name__ == "__main__":
    main()
