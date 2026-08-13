#!/usr/bin/env python3
"""News OVERREACTION detector + reversion backtest -> BacktestResult[news_overreaction] + JSON.

An overreaction = a MATERIAL news event whose β-adjusted abnormal day move CONTRADICTS the news
sentiment: GOOD news (net local_rating > 0) that CRASHES (day_abn <= -THR), or BAD news (rating < 0)
that POPS (day_abn >= +THR). The thesis (and the PODD/Aug-5 case) is that these REVERT.

We aggregate NewsItem to one event per (ticker, reaction-date), classify each as an overreaction or a
CONFIRMATION (move agrees with sentiment), and backtest the forward return from the reaction close
over 1/3/5/10/20 trading days — comparing overreactions (should revert) to confirmations (should
continue) and the material-news baseline. Bucketed by move size.

reaction-date = the session that first trades on the news (>=16:00 ET -> next day), matching how
day_abn is computed. Entry at the reaction close (the move already happened). day_abn is already
market-adjusted (β vs SPY). Episode-deduped per ticker; NO fees; directional (universe survivorship).

Run:  docker exec rotation-backend-1 python -u /app/news_overreaction.py
Opts: --thr PCT (default 5)  --min-impact N (default 2)
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from zoneinfo import ZoneInfo
from studies import _episode_starts, _tstat_from_returns

ET = ZoneInfo("America/New_York")
HZ = [1, 3, 5, 10, 20]
DEFAULT_THR = 5.0
DEFAULT_IMPACT = 2
GAP = 5                       # episode-dedup gap (trading days) between same-ticker events
BUCKETS = [("5-10%", 5, 10), ("10-15%", 10, 15), ("15-20%", 15, 20),
           ("20-25%", 20, 25), ("25%+", 25, 1e9)]
DEEP_LO, DEEP_HI = 10.0, 25.0   # the "dig-in" band: 10-25% down on good news
OUT = Path(__file__).resolve().parent / ".data" / "studies" / "news_overreaction.json"


def reaction_date(dt):
    """Session that first trades on the news (>=16:00 ET -> next day)."""
    t = pd.Timestamp(dt)
    et = t.tz_convert(ET) if t.tzinfo else t.tz_localize("UTC").tz_convert(ET)
    d = et.normalize().tz_localize(None)
    return d + pd.Timedelta(days=1) if et.hour >= 16 else d


class Acc:
    __slots__ = ("n", "s", "ss", "w")
    def __init__(self): self.n = self.s = self.ss = self.w = 0
    def add1(self, v, win):
        if v is None or not np.isfinite(v): return
        self.n += 1; self.s += v; self.ss += v * v; self.w += int(win)
    def stats(self):
        if not self.n: return dict(n=0, mean=None, wr=None, t=None)
        mean = self.s / self.n; wr = 100 * self.w / self.n; t = None
        if self.n >= 3:
            var = (self.ss - self.s * self.s / self.n) / (self.n - 1)
            if var > 0: t = round(float(mean / (np.sqrt(var) / np.sqrt(self.n))), 1)
        return dict(n=self.n, mean=round(100 * mean, 3), wr=round(wr, 1), t=t)


def build(thr, min_impact):
    from core.models import NewsItem, Candle
    # Aggregate items -> one event per (ticker, reaction-date): net sentiment + the day's abnormal move.
    rows = (NewsItem.objects.filter(local_impact__gte=min_impact, day_abn__isnull=False)
            .exclude(junk=True).exclude(off_ticker=True).exclude(day_suspect=True)
            .values_list("ticker", "dt", "day_abn", "local_rating", "cat_llm"))
    ev = {}
    for tk, dt, abn, rating, cat in rows.iterator():
        rd = reaction_date(dt)
        e = ev.setdefault((tk, rd), {"abn": abn, "rating": 0, "cats": {}})
        e["rating"] += (rating or 0)
        if cat:
            e["cats"][cat] = e["cats"].get(cat, 0) + abs(rating or 0) + 1
    print(f"{len(ev)} material news events (impact>={min_impact}) across "
          f"{len({k[0] for k in ev})} tickers", flush=True)

    # GROUNDED earnings verdict (ground_earnings.py): for earnings-driven events, use the hard-data
    # verdict (EPS surprise + forward guidance) instead of headline sentiment — so a beat-that-guided-
    # down (PODD) is NOT counted as good news. Keyed to the reaction date (report_date and +1 session).
    from core.models import EarningsEvent
    gv = {}
    for etk, erd, gs in (EarningsEvent.objects.filter(grounded_score__isnull=False)
                         .values_list("ticker", "report_date", "grounded_score")):
        ts = pd.Timestamp(erd)
        gv[(etk, ts)] = gs
        gv[(etk, ts + pd.Timedelta(days=1))] = gs
    n_grounded = sum(1 for k in ev if k in gv)
    print(f"{n_grounded} events have a grounded earnings verdict (headline overridden)", flush=True)

    tickers = sorted({k[0] for k in ev})
    qs = (Candle.objects.filter(ticker__in=tickers, interval="1d")
          .values_list("ticker", "date", "open", "close"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "date", "Open", "Close"])
    big["date"] = pd.to_datetime(big["date"])
    frames = {tk: g.sort_values("date").set_index("date")[["Open", "Close"]].astype(float)
              for tk, g in big.groupby("ticker", sort=False)}

    # class -> horizon -> Acc ;  and per-ticker event index lists for episode-dedup
    CLASSES = ["over_dn", "over_up", "conf_dn", "conf_up", "baseline"]
    acc = {c: {H: Acc() for H in HZ} for c in CLASSES}
    bkt = {b[0]: {H: Acc() for H in HZ} for b in BUCKETS}       # downside-overreaction size buckets
    by_ticker = {}    # (tk, cls) -> [(idx, {H: fwd})]
    deep_gap, deep_intraday = [], []   # 10-25% good-news crashes: overnight gap % + intraday open->close %
    n_events = 0

    for (tk, rd), e in ev.items():
        f = frames.get(tk)
        if f is None or len(f) < 30:
            continue
        s = f["Close"]; o = f["Open"]
        pos = s.index.searchsorted(rd)          # first candle on/after the reaction date
        if pos >= len(s):
            continue
        idx = int(pos)
        base = float(s.iloc[idx])
        if base <= 0:
            continue
        fwd = {}
        for H in HZ:
            j = idx + H
            fwd[H] = (float(s.iloc[j]) / base - 1.0) if j < len(s) else None
        abn = e["abn"]
        # Prefer the grounded earnings verdict where we have it; else the summed headline sentiment.
        grounded = gv.get((tk, rd))
        rating = grounded if grounded is not None else e["rating"]
        if abs(abn) < thr:
            continue
        n_events += 1
        # classify
        cls = None
        if rating > 0 and abn <= -thr: cls = "over_dn"     # good news, crashed -> bounce?
        elif rating < 0 and abn >= thr: cls = "over_up"    # bad news, popped -> fade?
        elif rating < 0 and abn <= -thr: cls = "conf_dn"   # bad news, crashed (agrees)
        elif rating > 0 and abn >= thr: cls = "conf_up"    # good news, popped (agrees)
        by_ticker.setdefault((tk, "baseline"), []).append((idx, fwd))
        if cls:
            by_ticker.setdefault((tk, cls), []).append((idx, fwd))
            if cls == "over_dn":
                for label, lo, hi in BUCKETS:
                    if lo <= abs(abn) < hi:
                        by_ticker.setdefault((tk, f"bkt:{label}"), []).append((idx, fwd))
                        break
                # DEEP DIVE: the 10-25% good-news-crash band — gap vs intraday + category
                if DEEP_LO <= abs(abn) < DEEP_HI:
                    dom = max(e["cats"], key=e["cats"].get) if e["cats"] else "?"
                    by_ticker.setdefault((tk, f"deepcat:{dom}"), []).append((idx, fwd))
                    by_ticker.setdefault((tk, "deep"), []).append((idx, fwd))
                    if idx > 0:
                        pc = float(s.iloc[idx - 1])
                        if pc > 0:
                            deep_gap.append((float(o.iloc[idx]) / pc - 1.0) * 100)      # overnight gap
                            deep_intraday.append((base / float(o.iloc[idx]) - 1.0) * 100)  # open->close

    # episode-dedup per (ticker, class) then accumulate
    def _accumulate(target, class_key, sign):
        for (tk, cls), lst in by_ticker.items():
            if cls != class_key:
                continue
            lst.sort()
            keep = _episode_starts([i for i, _ in lst], gap=GAP)
            for idx, fwd in lst:
                if idx not in keep:
                    continue
                for H in HZ:
                    v = fwd[H]
                    if v is None:
                        continue
                    # win = reversion in the overreaction's expected direction (sign): +1 expect up
                    win = (v > 0) if sign >= 0 else (v < 0)
                    target[H].add1(v, win)

    _accumulate(acc["over_dn"], "over_dn", +1)   # expect bounce (up)
    _accumulate(acc["over_up"], "over_up", -1)   # expect fade (down)
    _accumulate(acc["conf_dn"], "conf_dn", -1)   # expect continue down
    _accumulate(acc["conf_up"], "conf_up", +1)   # expect continue up
    _accumulate(acc["baseline"], "baseline", +1)
    for label, _, _ in BUCKETS:
        _accumulate(bkt[label], f"bkt:{label}", +1)

    def rows_for(accmap):
        return [{"H": H, **accmap[H].stats()} for H in HZ if accmap[H].stats()["n"]]

    # DEEP DIVE on the 10-25% good-news-crash band: overall reversion, by news category, gap profile.
    deep_acc = {H: Acc() for H in HZ}
    _accumulate(deep_acc, "deep", +1)
    deep_cats = sorted({cls.split("cat:", 1)[1] for (tk, cls) in by_ticker if cls.startswith("deepcat:")})
    deep_by_cat = {}
    for c in deep_cats:
        cacc = {H: Acc() for H in HZ}
        _accumulate(cacc, f"deepcat:{c}", +1)
        rows = rows_for(cacc)
        if rows and rows[0]["n"] >= 15:
            deep_by_cat[c] = rows
    gap_arr = np.array(deep_gap) if deep_gap else np.array([0.0])
    intr_arr = np.array(deep_intraday) if deep_intraday else np.array([0.0])
    deep_dive = {
        "band": f"{DEEP_LO:.0f}-{DEEP_HI:.0f}% down on good news",
        "overall": rows_for(deep_acc),
        "by_category": deep_by_cat,
        "gap_profile": {
            "n": len(deep_gap),
            "pct_gap_down": round(float((gap_arr < -2).mean() * 100), 1),   # opened >2% below prior close
            "avg_overnight_gap": round(float(gap_arr.mean()), 2),           # prior close -> open
            "avg_intraday_open_to_close": round(float(intr_arr.mean()), 2), # open -> close (recovery/further drop)
            "pct_intraday_recovered": round(float((intr_arr > 0).mean() * 100), 1),  # bounced within the day
        },
    }

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"thr_pct": thr, "min_impact": min_impact, "horizons": HZ,
                   "episode_gap_days": GAP, "n_events": n_events},
        "classes": {c: rows_for(acc[c]) for c in CLASSES},
        "downside_by_size": {label: rows_for(bkt[label]) for label, _, _ in BUCKETS},
        "deep_dive": deep_dive,
        "legend": {
            "over_dn": "OVERREACTION — good news (rating>0) but crashed (abn<=-thr): expect BOUNCE (mean fwd>0, rev%=P(fwd>0))",
            "over_up": "OVERREACTION — bad news (rating<0) but popped (abn>=+thr): expect FADE (mean fwd<0, rev%=P(fwd<0))",
            "conf_dn": "confirmation — bad news + crash (agrees): expect continue down",
            "conf_up": "confirmation — good news + pop (agrees): expect continue up",
        },
        "note": ("Entry at the reaction close (>=16:00 ET news -> next session). day_abn is β-adjusted "
                 "(market-neutral). Episode-deduped per ticker (gap=5d); NO fees; directional. "
                 "rev% = reversion rate in the expected direction; edge is over_dn vs conf_dn."),
    }
    return payload


def _p(tag, rows):
    for r in rows:
        print(f"  {tag:9} H{r['H']:>2}  n={r['n']:>5}  mean {r['mean']:>+7.2f}%  "
              f"rev {r['wr']:>5}%  t={r['t']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=DEFAULT_THR)
    ap.add_argument("--min-impact", type=int, default=DEFAULT_IMPACT)
    args = ap.parse_args()
    payload = build(args.thr, args.min_impact)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="news_overreaction",
            defaults={"payload": json.loads(json.dumps(payload, default=str)),
                      "computed_at": timezone.now()})
        print("Saved BacktestResult[news_overreaction]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)

    print(f"\n=== NEWS OVERREACTION reversion (|day_abn|>={args.thr}%, impact>={args.min_impact}) ===", flush=True)
    print("DOWNSIDE overreaction (good news, crashed) — does it BOUNCE?", flush=True)
    _p("over_dn", payload["classes"]["over_dn"])
    print(" vs CONFIRMATION (bad news, crashed) — does it keep falling?", flush=True)
    _p("conf_dn", payload["classes"]["conf_dn"])
    print("UPSIDE overreaction (bad news, popped) — does it FADE? (mean<0 / rev%=P(fwd<0))", flush=True)
    _p("over_up", payload["classes"]["over_up"])
    print("BASELINE (all material big moves):", flush=True)
    _p("base", payload["classes"]["baseline"])
    print("Downside overreaction by move size (bounce):", flush=True)
    for label, _, _ in BUCKETS:
        _p(label, payload["downside_by_size"][label])

    dd = payload["deep_dive"]
    print(f"\n=== DEEP DIVE — {dd['band']} ===", flush=True)
    g = dd["gap_profile"]
    print(f"  gap profile (n={g['n']}): {g['pct_gap_down']}% opened as a gap-down(>2%) | "
          f"avg overnight gap {g['avg_overnight_gap']}% | avg intraday(open->close) {g['avg_intraday_open_to_close']}% | "
          f"{g['pct_intraday_recovered']}% recovered intraday", flush=True)
    print("  overall bounce:", flush=True)
    _p("deep", dd["overall"])
    print("  by news category (bounce, n>=15):", flush=True)
    for c, rows in sorted(dd["by_category"].items(), key=lambda kv: -(kv[1][-1]["mean"] or -99)):
        r5 = next((x for x in rows if x["H"] == 5), rows[-1])
        r20 = next((x for x in rows if x["H"] == 20), rows[-1])
        print(f"    {c:16} n={r5['n']:>4}  5d {r5['mean']:>+6.2f}% (t{r5['t']})  "
              f"20d {r20['mean']:>+6.2f}% (t{r20['t']})", flush=True)


if __name__ == "__main__":
    main()
