#!/usr/bin/env python3
"""EARNINGS GUIDANCE-RAISE study (user hypothesis: 'stocks only go up when they revise their
forecast up at earnings'  ==  the classic BEAT-AND-RAISE premium).

Translates the claim into a testable decomposition using EarningsEvent.grounded_label
(beat/inline/miss  x  guided_up/none/down) plus the numeric guidance_eps_pct (fwd-EPS estimate
current vs ~30d pre-report). We measure, per bucket, TWO forward windows around the print:
  REACTION : close(t-1) -> close(t+1)      (through the announcement; captures BMO and AMC gaps)
  DRIFT20  : close(t+1) -> close(t+21)     (post-announcement drift, ~1 trading month, reaction stripped)
  DRIFT60  : close(t+1) -> close(t+61)     (~3 months)
t = first trading day >= report_date. Entry gated at the entry bar by PRICE_FLOOR + $5M DVOL floor.
Each return is net of COST_BPS round-trip and demeaned vs SPY over the identical window (strip beta).
Reported per bucket: n, mean, median, win%, SPY-excess, std, t-stat. The hypothesis PASSES only if the
guided_up cells carry the drift and beat-WITHOUT-raise does not.

HONESTY on coverage: guidance-tagged events are the ~13mo LLM slice (~700 labelled guided_*, ~1087 with
numeric guidance_eps_pct); beat/inline/miss is fully populated (~40k, 2015+). We therefore report the
guidance cells POOLED (N too thin to also split cap x sector), and separately give the cap-segmented view
of the coarse guid3 flag {up/none/down}. Sector split shown only for the fully-populated beat3 buckets.
Saved to BacktestResult[earnings_guidance_drift] (+JSON) per project rule.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/earnings_guidance_drift.py"""
import os, sys, json
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from seq_fundamental_study import load_candles, load_financial_reports, MIN_BARS, _chunk
from signal_discovery import _universe, DVOL_FLOOR, PRICE_FLOOR

COST_BPS = 10                         # round-trip; guidance-tagged names skew large/liquid
GUID_EPS_TH = 1.0                     # |guidance_eps_pct| >= 1% counts as a real up/down revision
DRIFT_HS = [20, 60]                   # post-announcement holding horizons (trading days)


def cap_bucket(dvol):
    if dvol is None or not np.isfinite(dvol) or dvol <= 0:
        return "cap?"
    m = dvol / 1e6
    return "ADV>500M" if m >= 500 else "ADV100-500M" if m >= 100 else "ADV20-100M" if m >= 20 else "ADV5-20M"
CAPS = ["ADV>500M", "ADV100-500M", "ADV20-100M", "ADV5-20M"]


def guid3(label, gpct):
    """Coarse revision direction. Prefer the LLM grounded tag; fall back to numeric guidance_eps_pct."""
    if label.endswith("guided_up"):
        return "up"
    if label.endswith("guided_down"):
        return "down"
    if gpct is not None and np.isfinite(gpct):
        if gpct >= GUID_EPS_TH:
            return "up"
        if gpct <= -GUID_EPS_TH:
            return "down"
    return "none"


def beat3(label):
    return "beat" if label.startswith("beat") else "miss" if label.startswith("miss") else "inline"


def main():
    universe, delisted = _universe()
    from core.models import EarningsEvent, Candle, Fundamental
    ev = list(EarningsEvent.objects.values_list(
        "ticker", "report_date", "grounded_label", "guidance_eps_pct", "eps_surprise_pct"))
    have = set(t for t, *_ in ev)
    names = [t for t in universe if t in have]
    print(f"universe {len(universe)} | earnings-covered {len(have)} | tradable ∩ {len(names)} | events {len(ev)}", flush=True)

    # sector map (for beat3 sector split)
    sect = dict(Fundamental.objects.exclude(sector="").values_list("ticker", "sector")) \
        if hasattr(Fundamental, "sector") else {}

    # events per ticker
    edates = defaultdict(list)
    for tk, rd, lab, gp, es in ev:
        edates[tk].append((pd.Timestamp(rd), lab or "", gp, es))

    spy = pd.DataFrame(list(Candle.objects.filter(ticker="SPY").values_list("date", "close")), columns=["d", "c"])
    spy["d"] = pd.to_datetime(spy["d"]); spy_c = spy.set_index("d")["c"].astype(float).sort_index()

    # accumulators: window -> keyfn-space -> list[(net_ret, spy_excess)]
    WINDOWS = ["reaction"] + [f"drift{h}" for h in DRIFT_HS]
    by_label = {w: defaultdict(list) for w in WINDOWS}      # grounded_label (the 9-way beat x guid cross)
    by_guid = {w: defaultdict(list) for w in WINDOWS}       # guid3 pooled
    by_guidcap = {w: defaultdict(list) for w in WINDOWS}    # (cap, guid3)
    by_beatsect = {w: defaultdict(list) for w in WINDOWS}   # (sector, beat3) fully-populated
    cost = COST_BPS / 1e4
    done = 0
    for ch in _chunk(names, 40):
        candles = load_candles(ch)
        for tk, sdf in candles.items():
            if len(sdf) < MIN_BARS:
                continue
            dates = sdf.index; close = sdf["Close"].values; vol = sdf["Volume"].values; n = len(close)
            dvol20 = pd.Series(close * vol).rolling(20, min_periods=10).mean().values
            spy_al = spy_c.reindex(dates).ffill().values
            di = dates.values
            sc = sect.get(tk) or "?"
            for rd, lab, gp, es in edates.get(tk, []):
                pos = int(np.searchsorted(di, np.datetime64(rd), side="left"))
                if pos >= n:
                    continue
                t = pos
                lab = lab or ("beat" if (es is not None and es > 0) else "miss" if (es is not None and es < 0) else "inline")
                g = guid3(lab, gp); b = beat3(lab)

                def rec(win, a, b_idx):
                    if a < 0 or b_idx >= n or b_idx <= a:
                        return
                    pa, pb = float(close[a]), float(close[b_idx])
                    if pa <= PRICE_FLOOR or not np.isfinite(pa) or not np.isfinite(pb):
                        return
                    # liquidity gate at the entry bar
                    if not (np.isfinite(dvol20[a]) and dvol20[a] >= DVOL_FLOOR):
                        return
                    r = (pb - pa) / pa - cost
                    sa, sb = spy_al[a], spy_al[b_idx]
                    sr = (sb - sa) / sa if (np.isfinite(sa) and np.isfinite(sb) and sa > 0) else 0.0
                    cap = cap_bucket(float(dvol20[a]) if np.isfinite(dvol20[a]) else None)
                    by_label[win][lab].append((r, r - sr))
                    by_guid[win][g].append((r, r - sr))
                    by_guidcap[win][(cap, g)].append((r, r - sr))
                    by_beatsect[win][(sc, b)].append((r, r - sr))

                rec("reaction", t - 1, t + 1)
                for h in DRIFT_HS:
                    rec(f"drift{h}", t + 1, t + 1 + h)
        done += len(ch)
        print(f"  scanned {done}/{len(names)}", flush=True)

    def stats(rows, minn=30):
        if len(rows) < minn:
            return None
        a = np.array([x[0] for x in rows], float) * 100
        e = np.array([x[1] for x in rows], float) * 100
        m = np.isfinite(a); a, e = a[m], e[m]
        if len(a) < minn:
            return None
        sd = a.std()
        return dict(n=len(a), mean=round(a.mean(), 3), med=round(float(np.median(a)), 3),
                    win=round((a > 0).mean() * 100, 1), excess=round(float(np.nanmean(e)), 3),
                    std=round(float(sd), 3), t=round(a.mean() / (sd / np.sqrt(len(a))), 2) if sd > 0 else 0.0)

    out = {"windows": {}}
    LABEL_ORDER = ["beat_guided_up", "beat", "beat_guided_down", "inline_guided_up", "inline",
                   "inline_guided_down", "miss_guided_up", "miss", "miss_guided_down"]

    for win in WINDOWS:
        print(f"\n=== {win.upper()}  (net {COST_BPS}bps, %/trade, demeaned vs SPY) ===", flush=True)
        # 1) the 9-way beat x guidance cross -- the crux of the user's claim
        print(f"  {'grounded_label':22}{'n':>7}{'mean':>8}{'med':>7}{'win%':>7}{'exSPY':>8}{'t':>7}", flush=True)
        lab_out = {}
        for lab in LABEL_ORDER:
            s = stats(by_label[win].get(lab, []))
            if not s:
                continue
            lab_out[lab] = s
            print(f"  {lab:22}{s['n']:>7}{s['mean']:>8.3f}{s['med']:>7.3f}{s['win']:>7.1f}{s['excess']:>8.3f}{s['t']:>7.2f}", flush=True)
        # 2) coarse guid3 pooled
        print(f"  {'-- guid3 pooled --':22}", flush=True)
        guid_out = {}
        for g in ["up", "none", "down"]:
            s = stats(by_guid[win].get(g, []))
            if not s:
                continue
            guid_out[g] = s
            print(f"  {'guid='+g:22}{s['n']:>7}{s['mean']:>8.3f}{s['med']:>7.3f}{s['win']:>7.1f}{s['excess']:>8.3f}{s['t']:>7.2f}", flush=True)
        out["windows"][win] = {"by_label": lab_out, "by_guid3": guid_out}

    # 3) cap-segmented guid3 on the drift20 window (per HARD RULE: segment where N allows)
    print(f"\n=== DRIFT20 by CAP x guid3 (segmented; cells n>=30) ===", flush=True)
    print(f"  {'cap':14}{'guid':6}{'n':>7}{'mean':>8}{'win%':>7}{'exSPY':>8}{'t':>7}", flush=True)
    guidcap_out = {}
    for cap in CAPS + ["cap?"]:
        for g in ["up", "none", "down"]:
            s = stats(by_guidcap["drift20"].get((cap, g), []))
            if not s:
                continue
            guidcap_out[f"{cap}|{g}"] = s
            print(f"  {cap:14}{g:6}{s['n']:>7}{s['mean']:>8.3f}{s['win']:>7.1f}{s['excess']:>8.3f}{s['t']:>7.2f}", flush=True)
    out["drift20_by_cap_guid"] = guidcap_out

    # 4) sector split of fully-populated beat3 (drift20) -- guided cells too thin to sector-split
    print(f"\n=== DRIFT20 by SECTOR x beat3 (fully-populated buckets; cells n>=50) ===", flush=True)
    print(f"  {'sector':24}{'beat':7}{'n':>7}{'mean':>8}{'exSPY':>8}{'t':>7}", flush=True)
    beatsect_out = {}
    for (sc, b) in sorted(by_beatsect["drift20"].keys(), key=lambda kv: (str(kv[0]), str(kv[1]))):
        s = stats(by_beatsect["drift20"][(sc, b)], minn=50)
        if not s:
            continue
        beatsect_out[f"{sc}|{b}"] = s
        print(f"  {sc[:23]:24}{b:7}{s['n']:>7}{s['mean']:>8.3f}{s['excess']:>8.3f}{s['t']:>7.2f}", flush=True)
    out["drift20_by_sector_beat"] = beatsect_out

    payload = dict(computed_at=pd.Timestamp.utcnow().isoformat(), cost_bps=COST_BPS,
                   guid_eps_th=GUID_EPS_TH, drift_hs=DRIFT_HS, **out,
                   caveat="Beat-and-raise test of 'stocks only go up when they guide forecast up'. "
                   "grounded_label = beat/inline/miss x guided_up/none/down (guided_* = ~13mo LLM slice, "
                   "thin N); guidance_eps_pct numeric fallback (|>=1%|). t=first trading day>=report_date. "
                   "REACTION close(t-1)->close(t+1); DRIFTh close(t+1)->close(t+1+h). Net cost, demeaned vs "
                   "SPY same window, $5M DVOL + price gate at entry bar. Guided cells too thin to also "
                   "split cap x sector; cap split shown on coarse guid3, sector split on beat3 only.")
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/earnings_guidance_drift.json").write_text(json.dumps(payload, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="earnings_guidance_drift",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[earnings_guidance_drift]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    main()
