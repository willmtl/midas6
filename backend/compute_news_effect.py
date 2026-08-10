"""
Same-day news EFFECT classifier — pure candle math, no LLM, no lookahead.

For EVERY NewsItem, decide whether the news moved the stock on its OWN reaction session,
capturing the overnight / pre-market gap. This lets the drift studies restrict to news that
actually mattered that day (user: "seperate the news into 2 categories, effect on the day and
no effect ... so this way we can analyze only the news that had an effect on the day").

Reaction session t:
  - Convert the news timestamp to US/Eastern. If it printed AT/AFTER 16:00 ET (post-close, incl.
    overnight), the market can only react on the NEXT session; otherwise (pre-market / intraday)
    it reacts the SAME session. Snap that date forward to the first available trading bar.

day_abn = β-adjusted abnormal return over prior_close(t-1) -> close(t):
    (close[t]/close[t-1] - 1) - β·(spy[t]/spy[t-1] - 1)     (×100, signed %)
  β = rolling 60d cov/var vs SPY, LAGGED to t-1 (clipped [0,3]) so the news-day return doesn't
  leak into its own beta. Because the window opens at the PRIOR close, an overnight gap is baked in.

day_effect = |day_abn| >= max(2·σ, 3.0%), σ = trailing-60d std of daily abnormal returns (lagged),
  i.e. a 2-sigma abnormal move (vol-normalized) with a 3% floor for very quiet names.

day_suspect = the move is likely a BAD CANDLE / data artifact, not real news. Two complementary
detectors (either → suspect):
  - ILLIQUID / OTC junk: reaction close < $1, OR trailing-20d median dollar-volume < $250K. These
    names (GTII, AYRWF, ...) have unreliable prints; a "social-impact-report −95%" is not a real move.
  - SNAP-BACK spike: a large move (|day_abn| >= 25%) that the NEXT session reverses by >= half,
    opposite-signed — the classic single-bad-print round-trip. A real crash (e.g. CORT −49% on an FDA
    rejection) stays down and is NOT flagged.
When suspect, day_effect is forced False so the drift studies drop it from BOTH the effect and the
control groups.

Forward "results since" (ret_1m / ret_3m / ret_1y): raw total return from the reaction-session close
forward ~21 / 63 / 252 trading days (signed %). Answers "how did the stock actually do after this
news?" in the dashboard. Null when the news is too recent to have the full window yet.

Writes NewsItem.day_abn (%) + day_effect (bool) + day_suspect (bool) + ret_1m/ret_3m/ret_1y (%).
Idempotent; re-runnable. NOTE: forward returns need a FULL run (not --only-null, which only touches
rows with a null day_abn) to backfill rows already scored for the same-day effect.
Run: docker compose exec -T backend python -u compute_news_effect.py [--limit N] [--only-null]
"""
import django, os, sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
import numpy as np, pandas as pd
from collections import defaultdict
from zoneinfo import ZoneInfo
from django.utils import timezone
from core.models import NewsItem, Candle

BETA_WIN = 60
VOL_WIN = 60
SIGMA_K = 2.0          # 2-sigma abnormal move counts as "moved"
FLOOR_PCT = 3.0        # ... but never call <3% abnormal an effect (quiet-name floor)
# data-quality (suspect) thresholds
MIN_PX = 1.0           # sub-dollar reaction close -> unreliable print
MIN_DOLLAR_VOL = 250_000.0   # trailing-20d median $-volume below this = illiquid/OTC junk
SPIKE_PCT = 25.0       # a move this big ...
SNAPBACK_FRAC = 0.5    # ... that next session reverses by >= half (opposite sign) = bad-print round-trip
DV_WIN = 20
# forward "results since" horizons: trading days ahead of the reaction close (1mo / 3mo / 1yr)
HORIZONS = {"ret_1m": 21, "ret_3m": 63, "ret_1y": 252}
UPDATE_FIELDS = ["day_abn", "day_effect", "day_suspect", "ret_1m", "ret_3m", "ret_1y"]
ET = ZoneInfo("America/New_York")


def reaction_date(dt):
    """Calendar date of the session that first trades on this news (US/Eastern; >=16:00 ET -> next day)."""
    et = pd.Timestamp(dt).tz_convert(ET) if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt).tz_localize("UTC").tz_convert(ET)
    d = et.normalize().tz_localize(None)
    if et.hour >= 16:
        d = d + pd.Timedelta(days=1)
    return d


def load_candles(tickers, interval="1d"):
    qs = (Candle.objects.filter(ticker__in=list(tickers), interval=interval)
          .values_list("ticker", "date", "close", "volume"))
    big = pd.DataFrame.from_records(list(qs), columns=["ticker", "date", "Close", "Volume"])
    if big.empty:
        return {}
    big["date"] = pd.to_datetime(big["date"])
    out = {}
    for tk, g in big.groupby("ticker", sort=False):
        out[tk] = g.sort_values("date").set_index("date")[["Close", "Volume"]]
    return out


def main(limit=None, only_null=False):
    qs = NewsItem.objects.all()
    if only_null:
        qs = qs.filter(day_abn__isnull=True)
    tickers = sorted(qs.values_list("ticker", flat=True).distinct())
    if limit:
        tickers = tickers[:limit]
    print(f"news tickers: {len(tickers)} (only_null={only_null})", flush=True)

    cds = load_candles(tickers + ["SPY"])
    spy = cds.get("SPY")
    if spy is None:
        print("no SPY candles — abort", flush=True); return {"updated": 0}
    spy_ret = spy["Close"].pct_change()

    now = timezone.now()
    updated = eff = susp = skipped = 0
    batch = []
    for ti, tk in enumerate(tickers):
        df = cds.get(tk)
        if df is None or len(df) < BETA_WIN + 5:
            # can't compute a reliable beta/vol -> leave null, count as skipped
            skipped += NewsItem.objects.filter(ticker=tk).count()
            continue
        idx = df.index
        close = df["Close"]
        r = close.pct_change()
        m = spy_ret.reindex(idx).ffill()
        cov = r.rolling(BETA_WIN).cov(m)
        var = m.rolling(BETA_WIN).var()
        beta = (cov / var).clip(0.0, 3.0).replace([np.inf, -np.inf], np.nan)
        beta_lag = beta.shift(1)                       # no news-day leakage into its own beta
        abn = (r - beta_lag * m) * 100.0               # daily abnormal return series (%)
        sigma = abn.rolling(VOL_WIN).std().shift(1)     # trailing vol of abnormal returns, lagged
        dvol = (close * df["Volume"]).rolling(DV_WIN).median().shift(1)   # trailing $-volume, lagged
        n = len(close)
        c = close.values
        abn_v = abn.values

        items = qs.filter(ticker=tk).values("id", "dt")
        for it in items:
            rd = reaction_date(it["dt"])
            pos = int(idx.searchsorted(rd, side="left"))   # first bar with date >= reaction date
            if pos < max(BETA_WIN, VOL_WIN) or pos >= n:
                skipped += 1
                continue
            b = beta_lag.iloc[pos]
            if not np.isfinite(b) or c[pos - 1] <= 0:
                skipped += 1
                continue
            da = (c[pos] / c[pos - 1] - 1.0 - b * m.iloc[pos]) * 100.0
            if not np.isfinite(da):
                skipped += 1
                continue
            # data-quality: bad-candle / artifact detectors
            suspect = c[pos] < MIN_PX
            dv = dvol.iloc[pos]
            if np.isfinite(dv) and dv < MIN_DOLLAR_VOL:
                suspect = True
            if abs(da) >= SPIKE_PCT and pos + 1 < n:
                an = abn_v[pos + 1]                          # next-session abnormal move
                if np.isfinite(an) and np.sign(an) == -np.sign(da) and abs(an) >= SNAPBACK_FRAC * abs(da):
                    suspect = True                           # spike that snapped back = bad print
            sg = sigma.iloc[pos]
            thr = max(SIGMA_K * sg, FLOOR_PCT) if np.isfinite(sg) and sg > 0 else FLOOR_PCT
            de = bool(abs(da) >= thr) and not suspect        # suspect moves never count as effect
            obj = NewsItem(id=it["id"])
            obj.day_abn = round(float(da), 3); obj.day_effect = de; obj.day_suspect = bool(suspect)
            # forward "results since" — raw total return from the reaction close forward N trading days
            base = c[pos]
            for hkey, hn in HORIZONS.items():
                fp = pos + hn
                setattr(obj, hkey,
                        round(float((c[fp] / base - 1.0) * 100.0), 3)
                        if (fp < n and base > 0 and np.isfinite(c[fp])) else None)
            batch.append(obj)
            updated += 1
            eff += de
            susp += bool(suspect)
            if len(batch) >= 2000:
                NewsItem.objects.bulk_update(batch, UPDATE_FIELDS, batch_size=1000)
                batch = []
        if (ti + 1) % 100 == 0:
            print(f"  {ti + 1}/{len(tickers)} tickers, updated={updated} effect={eff} suspect={susp}", flush=True)
    if batch:
        NewsItem.objects.bulk_update(batch, UPDATE_FIELDS, batch_size=1000)

    tot = updated + skipped
    print(f"DONE updated={updated} effect={eff} ({(100*eff/updated if updated else 0):.1f}% moved) "
          f"suspect={susp} ({(100*susp/updated if updated else 0):.1f}%) skipped={skipped} of {tot}", flush=True)
    return {"updated": updated, "effect": eff, "suspect": susp, "skipped": skipped}


if __name__ == "__main__":
    lim = None; onull = False
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            lim = int(sys.argv[i + 1])
        if a == "--only-null":
            onull = True
    main(limit=lim, only_null=onull)
