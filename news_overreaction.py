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
BUCKETS = [("5-10%", 5, 10), ("10-20%", 10, 20), ("20%+", 20, 1e9)]
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
            .values_list("ticker", "dt", "day_abn", "local_rating"))
    ev = {}
    for tk, dt, abn, rating in rows.iterator():
        rd = reaction_date(dt)
        e = ev.setdefault((tk, rd), {"abn": abn, "rating": 0})
        e["rating"] += (rating or 0)
    print(f"{len(ev)} material news events (impact>={min_impact}) across "
          f"{len({k[0] for k in ev})} tickers", flush=True)

    tickers = sorted({k[0] for k in ev})
    qs = (Candle.objects.filter(ticker__in=tickers, interval="1d")
          .values_list("ticker", "date", "close"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "date", "Close"])
    big["date"] = pd.to_datetime(big["date"])
    closes = {tk: g.sort_values("date").set_index("date")["Close"].astype(float)
              for tk, g in big.groupby("ticker", sort=False)}

    # class -> horizon -> Acc ;  and per-ticker event index lists for episode-dedup
    CLASSES = ["over_dn", "over_up", "conf_dn", "conf_up", "baseline"]
    acc = {c: {H: Acc() for H in HZ} for c in CLASSES}
    bkt = {b[0]: {H: Acc() for H in HZ} for b in BUCKETS}       # downside-overreaction size buckets
    by_ticker = {}    # (tk, cls) -> [(idx, {H: fwd})]
    n_events = 0

    for (tk, rd), e in ev.items():
        s = closes.get(tk)
        if s is None or len(s) < 30:
            continue
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
        abn = e["abn"]; rating = e["rating"]
        big_move = abs(abn) >= thr
        if not big_move:
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

    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"thr_pct": thr, "min_impact": min_impact, "horizons": HZ,
                   "episode_gap_days": GAP, "n_events": n_events},
        "classes": {c: rows_for(acc[c]) for c in CLASSES},
        "downside_by_size": {label: rows_for(bkt[label]) for label, _, _ in BUCKETS},
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


if __name__ == "__main__":
    main()
