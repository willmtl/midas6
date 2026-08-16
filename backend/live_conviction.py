#!/usr/bin/env python3
"""Score TODAY's live rotation basket on the 5-point conviction scale and surface the 'perfect plays'
(score >=4) worth levering / buying calls on. Uses the live rotation_picks payload + live RSI(10) from
candles. -> BacktestResult[perfect_plays] + prints.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/live_conviction.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from core.models import BacktestResult, Candle
from django.utils import timezone


def rsi10(ticker):
    rows = list(Candle.objects.filter(ticker=ticker).order_by("-date").values_list("close", flat=True)[:40])
    if len(rows) < 15:
        return None
    c = pd.Series(rows[::-1], dtype=float)
    d = c.diff()
    up = d.clip(lower=0).rolling(10).mean()
    dn = (-d.clip(upper=0)).rolling(10).mean()
    rs = up / dn.replace(0, np.nan)
    r = (100 - 100 / (1 + rs)).iloc[-1]
    return round(float(r), 1) if pd.notna(r) else None


rp = BacktestResult.objects.get(kind="rotation_picks")
picks = rp.payload["picks"]
pbs = sorted([p["pb_ratio"] for p in picks if p.get("pb_ratio")])
cheap_cut = pbs[max(0, len(pbs) // 3 - 1)] if pbs else 0

scored = []
for p in picks:
    pb = p.get("pb_ratio")
    de = p.get("debt_to_equity")
    ni = p.get("net_income")
    rank = p.get("rank", 99)
    r = rsi10(p["pick"])
    flags = {
        "deep_value": bool(pb is not None and pb <= cheap_cut),
        "profitable": bool(ni is not None and ni > 0),
        "very_low_debt": bool(de is not None and de < 0.5),
        "oversold": bool(r is not None and r < 45),
        "strong_sector": bool(rank <= 5),
    }
    score = sum(flags.values())
    scored.append({"sector": p["sector"], "pick": p["pick"], "pb": pb, "de": de, "rsi10": r,
                   "rank": rank, "score": score, "flags": [k for k, v in flags.items() if v],
                   "guard": p.get("guard_status"),
                   "tier": "PERFECT PLAY" if score >= 4 else ("solid" if score == 3 else "standard"),
                   "action": ("lever 1.5-2x / defined-risk calls" if score >= 4
                              else ("1x, small overweight ok" if score == 3 else "1x baseline"))})

scored.sort(key=lambda x: -x["score"])
print(f"\n{'sector':22} {'pick':10} {'score':>5}  {'tier':13} P/B {'D/E':>5} {'RSI':>5}  signals", flush=True)
for s in scored:
    print(f"  {s['sector'][:20]:20} {s['pick']:10} {s['score']:>3}/5  {s['tier']:13} "
          f"{str(s['pb']):>5} {str(s['de']):>5} {str(s['rsi10']):>5}  {'+'.join(s['flags'])}", flush=True)
perfect = [s for s in scored if s["score"] >= 4]
print(f"\nPERFECT PLAYS (score>=4): {[s['pick'] for s in perfect] or 'none today'}", flush=True)

# augment the live rotation_picks payload so the Rotation Pick tab shows conviction inline
smap = {(s["sector"], s["pick"]): s for s in scored}
for p in picks:
    s = smap.get((p.get("sector"), p.get("pick")))
    if s:
        p["conviction"] = s["score"]
        p["conviction_tier"] = s["tier"]
        p["conviction_flags"] = s["flags"]
rp.payload["picks"] = picks
rp.save(update_fields=["payload"])
print("Augmented rotation_picks with conviction", flush=True)

BacktestResult.objects.update_or_create(
    kind="perfect_plays",
    defaults={"payload": {"computed_at": pd.Timestamp.utcnow().isoformat(), "scored": scored,
                          "cheap_cut": cheap_cut,
                          "perfect_plays": [s["pick"] for s in perfect],
                          "note": "Conviction 0-5 (deep_value+profitable+very_low_debt+oversold+strong_sector). "
                                  "score>=4 = lever 1.5-2x / defined-risk calls; options unbacktested (theta/IV)."},
              "computed_at": timezone.now()})
print("Saved BacktestResult[perfect_plays]", flush=True)
