"""Market-adjusted news event study — keyed on OUR MODEL's read, with IV + short/long beta.

Isolates a news event's true effect from the market move using beta, and keys the analysis on the
on-box qwen classifier's read of each headline (local_dir / local_impact), NOT EODHD's (broken,
near-constant) sentiment. For each on-ticker event:

    AR_t = R_stock,t - beta * R_market,t          (abnormal = actual minus market-explained)

Also attaches, as of the event day (point-in-time):
  - ATM implied vol (OptionSnapshot.atm_iv, annualized %) and the IV-implied daily move
    (atm_iv / sqrt(252)); the SURPRISE ratio = day-0 abnormal move / implied daily move = how many
    implied-sigmas the news actually moved the stock (expected-vs-actual).
  - beta on BOTH a 252-day (slow) and 60-day (fast) window, so a name whose character changed
    recently isn't mis-bucketed by a stale year-long beta.

Groupings returned: by_model_dir_beta (252d), by_model_dir_beta60 (60d), by_model_dir_impact,
by_iv_dir (IV regime x direction), by_news_type_beta. One event per (ticker, day) = that day's
highest-impact headline. Light single-pass job (no MP); strengthens as the LLM drain fills the
local_dir/local_impact backlog.
"""
import numpy as np
import pandas as pd

BETA_MIN = 60                        # need some history before beta is meaningful
BETA_SHORT = 60                      # fast-beta window (trading days)
_SQRT_252 = np.sqrt(252.0)           # annualized IV -> daily move
_EXCLUDE_TYPES = {"", "macro", "other"}
_DIR_LABEL = {1: "up", 0: "flat", -1: "down"}


def _bucket_iv(v):
    if v is None or v != v:
        return "NA"
    if v < 30:
        return "low (<30)"
    if v < 50:
        return "mid (30-50)"
    if v < 80:
        return "high (50-80)"
    return "very high (>=80)"


def _load_iv(tickers):
    """{ticker: Series of ATM IV (%) indexed by date} from OptionSnapshot (daily, ~2022-09+)."""
    from core.models import OptionSnapshot
    qs = (OptionSnapshot.objects.filter(ticker__in=list(tickers), atm_iv__isnull=False)
          .values_list("ticker", "date", "atm_iv"))
    tmp = {}
    for tk, d, iv in qs:
        tmp.setdefault(tk, []).append((d, iv))
    out = {}
    for tk, rows in tmp.items():
        rows.sort()
        out[tk] = pd.Series([iv for _, iv in rows],
                            index=pd.to_datetime([d for d, _ in rows]))
    return out


def run_news_market_study(min_events=30, exclude_types=_EXCLUDE_TYPES):
    """Return {n_events, n_tickers, by_model_dir_beta, by_model_dir_beta60, by_model_dir_impact,
    by_iv_dir, by_news_type_beta}. On-ticker events (off_ticker=False) with a model direction."""
    from core.models import NewsItem
    from api.tasks import _get_dfs
    from pit_fundamentals import _rolling_beta, bucket_beta

    rows = list(NewsItem.objects
                .filter(off_ticker=False, local_dir__isnull=False)
                .exclude(cat_llm__in=list(exclude_types))
                .values_list("ticker", "dt", "cat_llm", "local_dir", "local_impact"))
    if not rows:
        return {"error": "no classified on-ticker events yet", "n_events": 0, "n_tickers": 0}

    ev = pd.DataFrame(rows, columns=["ticker", "dt", "news_type", "mdir", "mimp"])
    # Map each headline to the first SESSION that can react to it. Convert to ET; news at/after the
    # 16:00 ET close (where most earnings/guidance land) can't move the tape until the NEXT trading
    # day, so its reaction bar is the next calendar day (bfill then snaps to the next open session).
    # Before-close news reacts the same session. Date math in ET (naive after strip) also removes the
    # UTC evening date-boundary skew. Without this, after-close news was mis-attributed to a bar that
    # had already closed before the news existed.
    et = pd.to_datetime(ev["dt"], utc=True).dt.tz_convert("America/New_York")
    react = et.dt.normalize().dt.tz_localize(None)                       # naive midnight of ET date
    react = react + pd.to_timedelta((et.dt.hour >= 16).astype(int), unit="D")   # +1d if after close
    ev["day"] = react
    ev["mimp"] = ev["mimp"].fillna(0).astype(int)
    ev = ev.sort_values("mimp", ascending=False).drop_duplicates(["ticker", "day"])

    tickers = sorted(ev["ticker"].unique())
    dfs = _get_dfs(tickers + ["SPY"])
    spy = dfs.get("SPY")
    if spy is None or spy.empty:
        return {"error": "no SPY candles", "n_events": 0, "n_tickers": 0}
    spy_close = spy["Close"]
    spy_open = spy["Open"]
    iv_map = _load_iv(tickers)

    b_dir, b_dir60, b_dirimp, b_iv, b_type = {}, {}, {}, {}, {}
    b_all, b_dironly = {}, {}          # grand overall + by-direction-only summaries
    n_events = used_tickers = 0

    def _acc(d, key, ad0, car5, car20, raw0, mkt0, iv, surp, gap1):
        s = d.get(key)
        if s is None:
            s = d[key] = {"ar0": [], "car5": [], "car20": [], "raw0": [], "mkt0": [],
                          "iv": [], "surp": [], "gap1": []}
        s["ar0"].append(ad0); s["car5"].append(car5); s["car20"].append(car20)
        s["raw0"].append(raw0); s["mkt0"].append(mkt0); s["iv"].append(iv); s["surp"].append(surp)
        s["gap1"].append(gap1)

    for tk, g in ev.groupby("ticker", sort=False):
        df = dfs.get(tk)
        if df is None or len(df) < BETA_MIN:
            continue
        close = df["Close"].astype(float)
        r = close.pct_change()
        rspy = spy_close.reindex(close.index).ffill().pct_change()
        beta = _rolling_beta(close, spy_close)                 # 252d (default)
        beta_s = _rolling_beta(close, spy_close, window=BETA_SHORT)   # 60d
        ar = r - beta.shift(1) * rspy
        ar_v, rspy_v, r_v = ar.values, rspy.values, r.values
        beta_prev = beta.shift(1).values
        beta_s_prev = beta_s.shift(1).values
        close_arr = close.values
        open_arr = df["Open"].astype(float).values
        spy_open_al = spy_open.reindex(close.index).ffill().values
        spy_close_al = spy_close.reindex(close.index).ffill().values
        # IV as of each bar (annualized %), forward-filled from the option snapshots
        ivs = iv_map.get(tk)
        iv_v = (ivs.reindex(close.index, method="ffill").values
                if ivs is not None else np.full(len(close), np.nan))
        n = len(close)
        pos = close.index.get_indexer(g["day"].values, method="bfill")
        used = False
        seen_bars = set()   # distinct reaction dates can bfill onto the SAME bar (e.g. Fri-after-close
                            # and Mon-morning news both react Monday). g is impact-desc, so the first
                            # hit is the highest-impact headline; skip the rest to avoid double-counting
                            # the identical forward window.
        for e, ntype, mdir, mimp in zip(pos, g["news_type"].values, g["mdir"].values, g["mimp"].values):
            if e < 1 or e >= n:
                continue
            if e in seen_bars:
                continue
            seen_bars.add(e)
            b = beta_prev[e]
            if b != b:
                continue
            ad0 = ar_v[e]
            if ad0 != ad0:
                continue
            # CAR only when the FULL forward window exists (per news/ticker) — a partial window near
            # the data's end would understate the drift, and our news is recent-heavy. Incomplete →
            # NaN, so _finalize's nanmean excludes it rather than diluting with truncated sums. Also
            # require every bar in the window to be valid: np.nansum treats a NaN (trading halt / gap
            # in the series) as 0, silently dropping that day's abnormal return; sum→NaN if any missing.
            w5 = ar_v[e + 1:e + 6]
            car5 = float(w5.sum()) if (e + 6 <= n and not np.isnan(w5).any()) else np.nan
            w20 = ar_v[e + 1:e + 21]
            car20 = float(w20.sum()) if (e + 21 <= n and not np.isnan(w20).any()) else np.nan
            # following-morning move: abnormal overnight GAP into the next session's open (after-hours
            # + pre-market repositioning). Reversion off a day-0 over-reaction usually starts here —
            # a gap OPPOSITE to AR_d0 = the fade beginning.
            if e + 1 < n:
                gap1 = (open_arr[e + 1] / close_arr[e] - 1.0) - b * (spy_open_al[e + 1] / spy_close_al[e] - 1.0)
            else:
                gap1 = np.nan
            iv = iv_v[e]
            # surprise ratio: RAW day-0 move in units of the IV-implied 1-sigma daily move. Uses the
            # raw (not abnormal) move because options price the TOTAL move — comparing abnormal to a
            # total-implied benchmark mixes units. This matches the IV-calibration metric.
            surp = (r_v[e] / (iv / 100.0 / _SQRT_252)) if (iv == iv and iv > 0) else np.nan
            bb = bucket_beta(b)
            bs = bucket_beta(beta_s_prev[e])
            dl = _DIR_LABEL.get(int(mdir), "?")
            _acc(b_dir, (dl, bb), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_dir60, (dl, bs), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_dirimp, (dl, int(mimp)), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_iv, (_bucket_iv(iv), dl), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_type, (ntype, bb), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_all, ("ALL events", ""), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            _acc(b_dironly, (dl, ""), ad0, car5, car20, r_v[e], rspy_v[e], iv, surp, gap1)
            n_events += 1
            used = True
        if used:
            used_tickers += 1

    def _tstat(vals):
        # One-sample t vs 0 on the non-NaN observations. NOTE: same-ticker event windows overlap in
        # time, so these obs aren't fully independent — treat |t|>=~2 as "worth a look", not proof.
        a = np.asarray(vals, float)
        a = a[np.isfinite(a)]
        if len(a) < 3:
            return None
        sd = a.std(ddof=1)
        if not (sd > 0):
            return None
        return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2)

    def _finalize(d, k0, k1):
        out = []
        for key, s in d.items():
            ar0 = np.array(s["ar0"], float)
            mkt0 = np.array(s["mkt0"], float)
            if len(ar0) < min_events:
                continue
            down = mkt0 < 0
            iv = np.array(s["iv"], float)
            surp = np.array(s["surp"], float)
            car20a = np.asarray(s["car20"], float)
            out.append({
                k0: key[0], k1: key[1], "n": int(len(ar0)),
                "avg_ar_d0": round(float(np.nanmean(ar0)) * 100, 3),
                "t_ar0": _tstat(ar0),
                "avg_car5": round(float(np.nanmean(s["car5"])) * 100, 3),
                "avg_car20": round(float(np.nanmean(s["car20"])) * 100, 3),
                "n_car20": int(np.isfinite(car20a).sum()),   # true sample behind CAR-20 (< n: recent events lack 20 fwd days)
                "t_car20": _tstat(s["car20"]),
                "avg_gap1": round(float(np.nanmean(s["gap1"])) * 100, 3),   # next-morning abnormal gap
                "avg_raw_d0": round(float(np.nanmean(s["raw0"])) * 100, 3),
                "avg_mkt_d0": round(float(np.nanmean(mkt0)) * 100, 3),
                "avg_iv": (round(float(np.nanmean(iv)), 1) if np.isfinite(iv).any() else None),
                "avg_surprise": (round(float(np.nanmean(surp)), 2) if np.isfinite(surp).any() else None),
                "n_mkt_down": int(down.sum()),
                "avg_ar_d0_up": round(float(np.nanmean(ar0[~down])) * 100, 3) if (~down).any() else None,
                "avg_ar_d0_down": round(float(np.nanmean(ar0[down])) * 100, 3) if down.any() else None,
            })
        out.sort(key=lambda x: -x["n"])
        return out

    _ov = _finalize(b_all, "scope", "_")
    return {
        "n_events": n_events, "n_tickers": used_tickers,
        "overall": _ov[0] if _ov else {},                       # grand average across ALL events
        "by_direction": _finalize(b_dironly, "model_dir", "_"),  # aggregated over beta/impact/IV
        "by_model_dir_beta": _finalize(b_dir, "model_dir", "beta_bucket"),
        "by_model_dir_beta60": _finalize(b_dir60, "model_dir", "beta60_bucket"),
        "by_model_dir_impact": _finalize(b_dirimp, "model_dir", "impact"),
        "by_iv_dir": _finalize(b_iv, "iv_bucket", "model_dir"),
        "by_news_type_beta": _finalize(b_type, "news_type", "beta_bucket"),
    }


def run_and_save(min_events=30):
    """Compute the study and upsert the single 'latest' NewsEventStudy row for the dashboard."""
    from core.models import NewsEventStudy
    res = run_news_market_study(min_events=min_events)
    data = {k: v for k, v in res.items() if k not in ("n_events", "n_tickers")}
    NewsEventStudy.objects.update_or_create(label="latest", defaults={
        "n_events": res.get("n_events", 0), "n_tickers": res.get("n_tickers", 0), "data": data})
    return {"n_events": res.get("n_events", 0), "n_tickers": res.get("n_tickers", 0),
            "groupings": list(data.keys())}
