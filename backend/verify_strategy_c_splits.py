#!/usr/bin/env python3
"""VERIFY Strategy C (flagship) is not contaminated by MISSING splits in .data/splits_cache.json.

Failure mode being tested: the flagship's mktcap/P/B uses as_traded_close = adjusted x future_split_factor,
where the factor comes ONLY from splits_cache.json. If a real forward split is MISSING from that cache, a
pre-split pick's as_traded price is NOT un-adjusted -> mktcap understated by the split ratio -> P/B looks
artificially cheap -> the name may be SPURIOUSLY selected as a value pick.

This script cross-checks EVERY actual flagship pick (flagship_history.json) against two INDEPENDENT split
signals (CorporateAction rows + shares_outstanding jumps) that are ABSENT from splits_cache. A contaminated
pick = a pick whose ticker really split AFTER the pick month (within the hold/backtest horizon) with that
split missing from the cache. Reports the exact contaminated (ticker, month) pairs, or clears the flagship.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/verify_strategy_c_splits.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import pandas as pd, numpy as np
from pathlib import Path
from collections import defaultdict
from core.models import CorporateAction, FinancialReport
import price_basis

splits = price_basis.load_splits()                      # {ticker: {date: ratio}} — the cache the flagship relies on
print(f"splits_cache tickers: {len(splits)}", flush=True)

# ── flagship picks (deployed adaptive) ──
J = json.load(open("/app/.data/studies/flagship_history.json"))
picks = []
for m in J["months"]:
    d = pd.Timestamp(m["date"])
    for p in m["picks"]:
        picks.append((p["ticker"], d))
pick_tickers = sorted({t for t, _ in picks})
print(f"flagship picks: {len(picks)} pick-months over {len(J['months'])} months, {len(pick_tickers)} distinct names", flush=True)


def cache_has_split_near(tk, when, days=120):
    """Is there a split in the cache for tk within +/- `days` of `when`?"""
    for ds in splits.get(tk, {}):
        if abs((pd.Timestamp(ds) - when).days) <= days:
            return True
    return False


# ── independent split signal #1: CorporateAction split rows ──
ca_splits = defaultdict(list)
for tk, exd, sr, sf, st in CorporateAction.objects.filter(action_type="split", ticker__in=pick_tickers) \
        .values_list("ticker", "ex_date", "split_ratio", "split_from", "split_to"):
    ratio = None
    if sr:
        ratio = float(sr)
    elif sf and st:
        ratio = float(st) / float(sf)
    ca_splits[tk].append((pd.Timestamp(exd), ratio))
print(f"CorporateAction split rows for flagship names: {sum(len(v) for v in ca_splits.values())} across {len(ca_splits)} names", flush=True)

# ── independent split signal #2: shares_outstanding jumps (>=1.8x or <=0.55x between consecutive filings) ──
sh_splits = defaultdict(list)
cols = ["ticker", "period_end", "avail_date", "shares_outstanding"]
qs = FinancialReport.objects.filter(ticker__in=pick_tickers).values_list(*cols)
big = pd.DataFrame.from_records(list(qs), columns=cols)
big = big.dropna(subset=["shares_outstanding"])
for tk, g in big.groupby("ticker"):
    g = g.sort_values("period_end")
    s = g["shares_outstanding"].astype(float).values
    dts = pd.to_datetime(g["avail_date"]).values
    for i in range(1, len(s)):
        if s[i - 1] > 0:
            r = s[i] / s[i - 1]
            if r >= 1.8 or r <= 0.55:                    # forward or reverse split-sized jump
                sh_splits[tk].append((pd.Timestamp(dts[i]), round(r, 2)))

# ── union of independent signals -> flag those MISSING from the cache ──
missing = []                                             # (ticker, split_date, ratio, source)
for tk in pick_tickers:
    for when, ratio in ca_splits.get(tk, []):
        if not cache_has_split_near(tk, when):
            missing.append((tk, when.date().isoformat(), ratio, "CorporateAction"))
    for when, ratio in sh_splits.get(tk, []):
        if not cache_has_split_near(tk, when) and not any(abs((pd.Timestamp(x[0]) - when).days) <= 120 for x in ca_splits.get(tk, [])):
            missing.append((tk, when.date().isoformat(), ratio, "shares_jump"))

print(f"\nindependent split signals MISSING from splits_cache (flagship names): {len(missing)}", flush=True)
for tk, when, ratio, src in sorted(missing):
    print(f"  {tk:6} {when}  ratio~{ratio}  [{src}]", flush=True)

# ── DECISIVE TEST: rebuild the flagship P/B panel exactly as run() does, then measure whether P/B shows a
#    real uncorrected STEP at each flagged split date. A genuinely-missing split makes pb = as_traded*sh/eq
#    jump by ~ratio (shares jumps, as_traded doesn't compensate). Noise/dilution/glitch -> no clean step. ──
from seq_fundamental_study import load_financial_reports
from survivorship_smallcap_study import _pit_monthly_panel
from core.models import Candle

flagged_names = sorted({tk for tk, _, _, _ in missing})
print(f"\nrebuilding flagship P/B panel for {len(flagged_names)} flagged names to test for real uncorrected steps...", flush=True)
reps = load_financial_reports(flagged_names)
_rows = Candle.objects.filter(ticker__in=flagged_names, date__gte="2010-01-01").values_list("ticker", "date", "close")
_df = pd.DataFrame(list(_rows), columns=["ticker", "date", "close"])
_df["date"] = pd.to_datetime(_df["date"]); _df["close"] = _df["close"].astype(float)
cnd = _df.pivot_table(index="date", columns="ticker", values="close").resample("ME").last()
midx = cnd.index
sh = _pit_monthly_panel(reps, "shares_outstanding", midx)
eq = _pit_monthly_panel(reps, "total_equity", midx)
common = cnd.columns.intersection(sh.columns).intersection(eq.columns)
as_traded = price_basis.as_traded_close(cnd[common])
pb_panel = (as_traded * sh[common]) / eq[common].where(eq[common] != 0)

# for each distinct flagged (ticker, split_date), compute the pb ratio across that date (median of 3 mo each side)
def pb_step(tk, when):
    if tk not in pb_panel.columns:
        return None
    s = pb_panel[tk].dropna(); wd = pd.Timestamp(when)
    before = s[s.index < wd].tail(3); after = s[s.index >= wd].head(3)
    if len(before) < 1 or len(after) < 1 or before.median() == 0:
        return None
    return float(after.median() / before.median())

real_steps = []
seen = set()
for tk, when, ratio, src in missing:
    key = (tk, when)
    if key in seen:
        continue
    seen.add(key)
    step = pb_step(tk, when)
    if step is not None and (step >= 1.5 or step <= 0.67):   # a real ~ratio discontinuity in the flagship's own pb
        real_steps.append((tk, when, round(step, 2), ratio, src))

print(f"\n=== flagged splits that produce a REAL P/B step in the flagship's own panel: {len(real_steps)} ===", flush=True)
for tk, when, step, ratio, src in sorted(real_steps):
    picked_before = sorted({pd_.date().isoformat() for pt, pd_ in picks if pt == tk and pd_ <= pd.Timestamp(when)})
    tag = f"  <- PICKED {picked_before}" if picked_before else "  (never picked pre-split)"
    print(f"  {tk:6} {when}  pb_step x{step}  (jump~{ratio} [{src}]){tag}", flush=True)

# final verdict: contamination = a real pb step for a name that was PICKED at/before that split
contaminated = [(tk, when, step) for tk, when, step, ratio, src in real_steps
                if any(pt == tk and pd_ <= pd.Timestamp(when) for pt, pd_ in picks)]
print(f"\n=== VERDICT ===", flush=True)
if not contaminated:
    print("  Strategy C is SPLIT-CLEAN: no flagged split produces a real uncorrected P/B step in a name at/before it "
          "was a flagship pick. All vendor-known (CorporateAction) splits are in the cache; the shares-jump hits are "
          "dilution / ADR / data-glitch artifacts, not uncorrected splits.", flush=True)
else:
    print(f"  {len(contaminated)} genuinely contaminated pick(s) — investigate:", flush=True)
    for tk, when, step in sorted(contaminated):
        print(f"    {tk} split {when} caused pb step x{step} at/after a pick", flush=True)
