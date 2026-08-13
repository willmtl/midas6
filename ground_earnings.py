#!/usr/bin/env python3
"""GROUND the earnings-news categorization in hard data instead of headline sentiment.

Our news classifier flattens an earnings report to one sentiment ("beat" -> positive), so a beat
that GUIDED DOWN (PODD/Aug-5: EPS +15% beat but fwd EPS estimate cut -4.3%) is mislabeled as
unambiguously good. This computes a multi-dimensional GROUNDED verdict per EarningsEvent from data
we already hold:

  EPS surprise   : EarningsEvent.eps_surprise_pct                 (broad)
  Revenue        : FinancialReport.revenue  -> revenue_yoy_pct; vs EstimateRevision.revenue_avg
                   -> revenue_surprise_pct                        (actuals broad, estimates recent)
  Guidance       : EstimateRevision fwd EPS (current vs ~30d pre-report) -> guidance_eps_pct
                   = the market's forward read; the PODD differentiator  (recent)

grounded_score = e + r + 1.5*g  (each of e/r/g in {-1,0,+1}; guidance weighted highest — it's what
the tape prices). grounded_label e.g. "beat_guided_down". Writes the 5 grounded fields on EarningsEvent.

Run: docker exec rotation-backend-1 python -u /app/ground_earnings.py
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
import datetime as dt
from collections import defaultdict
from core.models import EarningsEvent, FinancialReport, EstimateRevision

EPS_BAND = 2.0     # |eps_surprise| below this = inline
REV_BAND = 1.0
GUID_BAND = 1.0


def _sign(v, band):
    if v is None:
        return None
    return 1 if v > band else -1 if v < -band else 0


def build():
    # --- financials per ticker: sorted (period_end, revenue) for actual + YoY ---
    fr = defaultdict(list)
    for tk, pe, rev in (FinancialReport.objects.filter(revenue__isnull=False)
                        .values_list("ticker", "period_end", "revenue")):
        fr[tk].append((pe, rev))
    for tk in fr:
        fr[tk].sort()

    def revenue_for(tk, report_date):
        """(revenue_actual, revenue_yoy_pct) for the quarter reported on/just before report_date."""
        rows = fr.get(tk)
        if not rows:
            return None, None
        # the reported quarter ends BEFORE the report; take the latest period_end < report_date within 120d
        cand = [(pe, rv) for pe, rv in rows if report_date - dt.timedelta(days=130) <= pe < report_date]
        if not cand:
            return None, None
        pe, rev = cand[-1]
        yoy = None
        prior = [(p, r) for p, r in rows if abs((pe - p).days - 365) <= 20 and p < pe]
        if prior and prior[-1][1]:
            yoy = round((rev / prior[-1][1] - 1) * 100, 1)
        return rev, yoy

    # --- estimate revisions per ticker: guidance (fwd EPS) + revenue estimate, nearest asof ---
    er = defaultdict(list)
    for tk, plabel, asof, cur, d30, rev_avg in (
            EstimateRevision.objects.values_list(
                "ticker", "period_label", "asof", "eps_current", "eps_30d_ago", "revenue_avg")):
        er[tk].append((plabel, asof, cur, d30, rev_avg))

    def guidance_for(tk, report_date):
        """fwd EPS estimate change (current vs 30d-ago) from the snapshot nearest the report."""
        rows = [r for r in er.get(tk, []) if r[0] in ("+1y", "+1q")]
        if not rows:
            return None
        # nearest asof within 30d of the report (captures the post-report revision)
        near = sorted(rows, key=lambda r: abs((r[1] - report_date).days))
        for plabel, asof, cur, d30, _ in near:
            if abs((asof - report_date).days) <= 30 and cur and d30:
                return round((cur / d30 - 1) * 100, 1)
        return None

    updated = 0
    label_counts = defaultdict(int)
    for ev in EarningsEvent.objects.all().iterator():
        rev_act, rev_yoy = revenue_for(ev.ticker, ev.report_date)
        guid = guidance_for(ev.ticker, ev.report_date)

        # SOUND signals only: EPS surprise (broad) + forward GUIDANCE direction (recent). Revenue
        # SURPRISE is left null — EstimateRevision.revenue_avg is not reliably period-matched to the
        # just-reported quarter (annual vs quarterly), so it printed spurious misses. rev_yoy (actual
        # growth) is stored as context but NOT scored (growth != surprise). Guidance is weighted
        # heavily (2x): a cut forward estimate is what the tape actually punishes (the PODD case).
        e = _sign(ev.eps_surprise_pct, EPS_BAND)
        g = _sign(guid, GUID_BAND)
        score = 0.0
        if e is not None: score += e
        if g is not None: score += 2.0 * g
        score = max(-3.0, min(3.0, round(score, 2)))

        eps_word = "beat" if (e or 0) > 0 else "miss" if (e or 0) < 0 else "inline"
        parts = [eps_word]
        if g is not None and g < 0: parts.append("guided_down")
        elif g is not None and g > 0: parts.append("guided_up")
        label = "_".join(parts)

        ev.revenue_surprise_pct = None
        ev.revenue_yoy_pct = rev_yoy
        ev.guidance_eps_pct = guid
        ev.grounded_label = label
        ev.grounded_score = score
        ev.save(update_fields=["revenue_actual", "revenue_estimate", "revenue_surprise_pct",
                               "revenue_yoy_pct", "guidance_eps_pct", "grounded_label", "grounded_score"])
        updated += 1
        label_counts[label] += 1

    print(f"Grounded {updated} EarningsEvents.", flush=True)
    print("label distribution (top):", flush=True)
    for lab, n in sorted(label_counts.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {lab:22} {n}", flush=True)
    # show the PODD example
    p = EarningsEvent.objects.filter(ticker="PODD", report_date=dt.date(2026, 8, 5)).first()
    if p:
        print(f"\nPODD 2026-08-05: eps_surp {p.eps_surprise_pct:+.1f}%  rev_yoy {p.revenue_yoy_pct}  "
              f"guidance {p.guidance_eps_pct}  -> {p.grounded_label} (score {p.grounded_score}) "
              f"[headline said +3 positive]", flush=True)


if __name__ == "__main__":
    build()
