#!/usr/bin/env python3
"""C EARNINGS-IN-HOLD study: what happens to a pick when it REPORTS EARNINGS during the hold month? Each pick is
bought at month-end `date` and sold at `ndate` (next month-end); flag it if the ticker had an EarningsEvent with
report_date in (date, ndate]. Compare hold-month return, downside (MAE = worst intra-month excursion) and win
rate for earnings-in-hold vs no-earnings, with a Welch t-test. Then split the earnings picks by EPS-surprise sign
and grounded beat/miss score — does buying INTO a print help (beats drift) or hurt (miss/vol)? PIT-actionable:
the earnings calendar is known before the buy, so 'avoid/prefer earnings-in-hold' is a real wireable rule.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/c_earnings_hold_study.py"""
import os, json, bisect, warnings, datetime as dt
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np
from core.models import EarningsEvent

J = json.load(open("/app/.data/studies/flagship_history.json"))
months = J["months"]

# earnings index: ticker -> (sorted report_dates, [surprise, grounded_score, grounded_label, guidance])
tickers = sorted({p["ticker"] for m in months for p in m["picks"]})
ev = {}
for tk, rd, sp, gs, gl, gg in EarningsEvent.objects.filter(ticker__in=tickers).order_by("ticker", "report_date")\
        .values_list("ticker", "report_date", "eps_surprise_pct", "grounded_score", "grounded_label", "guidance_eps_pct"):
    rec = ev.setdefault(tk, ([], [])); rec[0].append(rd); rec[1].append((sp, gs, gl, gg))
print(f"picks tickers={len(tickers)}  with earnings history={len(ev)}\n")


def earn_in(tk, d0, d1):
    """earnings events with report_date in (d0, d1]; returns list of (surprise, grounded_score, label, guid)."""
    rec = ev.get(tk)
    if not rec:
        return []
    lo = bisect.bisect_right(rec[0], d0); hi = bisect.bisect_right(rec[0], d1)
    return rec[1][lo:hi]


rows = []   # (ret, mae, has_earn, surprise, gscore)
for m in months:
    if m.get("ndate") in (None, ""):
        continue
    d0 = dt.date.fromisoformat(m["date"]); d1 = dt.date.fromisoformat(m["ndate"])
    for p in m["picks"]:
        if p.get("ret") is None:
            continue
        es = earn_in(p["ticker"], d0, d1)
        sp = gs = None
        if es:
            sps = [e[0] for e in es if e[0] is not None]; gss = [e[1] for e in es if e[1] is not None]
            sp = float(np.mean(sps)) if sps else None
            gs = float(np.mean(gss)) if gss else None
        rows.append((p["ret"], p.get("mae"), bool(es), sp, gs))

R = np.array([r[0] for r in rows], float)
HE = np.array([r[2] for r in rows], bool)


def stat(mask, lab):
    a = R[mask]; mae = np.array([rows[i][1] for i in range(len(rows)) if mask[i] and rows[i][1] is not None], float)
    if not len(a):
        print(f"  {lab:28} (none)"); return a
    print(f"  {lab:28} n={len(a):>4}  mean {a.mean()*100:+6.2f}%  median {np.median(a)*100:+6.2f}%  "
          f"win {(a>0).mean()*100:4.0f}%  MAE {mae.mean()*100:+6.1f}%" if len(mae) else
          f"  {lab:28} n={len(a):>4}  mean {a.mean()*100:+6.2f}%  median {np.median(a)*100:+6.2f}%  win {(a>0).mean()*100:4.0f}%")
    return a


def welch(a, b):
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    t = (a.mean()-b.mean())/np.sqrt(va+vb)
    return round(float(t), 2)


print(f"=== (1) EARNINGS-IN-HOLD vs NOT  (base rate reports-in-hold = {HE.mean()*100:.0f}% of picks) ===")
ae = stat(HE, "earnings IN hold month");  an = stat(~HE, "NO earnings in hold")
print(f"  Δ mean = {(R[HE].mean()-R[~HE].mean())*100:+.2f}%   Welch t = {welch(R[HE], R[~HE])}")

print("\n=== (2) EARNINGS picks split by EPS-SURPRISE sign (beat vs miss) ===")
sps = np.array([rows[i][3] if rows[i][3] is not None else np.nan for i in range(len(rows))])
stat(HE & (sps > 0), "beat (surprise>0)")
stat(HE & (sps < 0), "miss (surprise<0)")
stat(HE & np.isnan(sps), "earnings, no surprise data")
b = R[HE & (sps > 0)]; ms = R[HE & (sps < 0)]
if len(b) and len(ms):
    print(f"  beat−miss Δ = {(b.mean()-ms.mean())*100:+.2f}%   Welch t = {welch(b, ms)}")

print("\n=== (3) EARNINGS picks split by GROUNDED score (beat-and-guided vs miss-and-guided-down) ===")
gss = np.array([rows[i][4] if rows[i][4] is not None else np.nan for i in range(len(rows))])
stat(HE & (gss > 0), "grounded positive (>0)")
stat(HE & (gss == 0), "grounded neutral (=0)")
stat(HE & (gss < 0), "grounded negative (<0)")

print("\n=== (4) is it just DOWNSIDE (does reporting add tail risk)? worst-decile hold returns ===")
p10_e = np.percentile(R[HE], 10)*100 if HE.sum() else float("nan")
p10_n = np.percentile(R[~HE], 10)*100 if (~HE).sum() else float("nan")
print(f"  10th-pct hold return:  earnings-in-hold {p10_e:+.1f}%   no-earnings {p10_n:+.1f}%")
print(f"  fraction of picks < -20% in the month:  earnings {100*(R[HE]<-0.2).mean():.1f}%   no-earnings {100*(R[~HE]<-0.2).mean():.1f}%")
