#!/usr/bin/env python3
"""Backfill dated SHORT-INTEREST history from Polygon -> .data/short_interest.jsonl (as-of archive).

Fundamental.short_pct_float is a current snapshot (no history), so short interest was un-backtestable.
Polygon /stocks/v1/short-interest returns bi-monthly settlement history back to ~2017 (short_interest shares,
avg_daily_volume, days_to_cover) — deep enough for the 2021-2026 H4 window, and covered even on small-caps.
Writes one jsonl row per (ticker, settlement_date) so h4_c_short can join it as-of and bucket the dip-buy.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_short_interest.py
"""
import os, json, time, urllib.request, urllib.error, warnings
warnings.filterwarnings("ignore")

OUT = "/app/.data/short_interest.jsonl"
KEY = os.environ.get("POLYGON_API_KEY")


def fetch(tk):
    url = f"https://api.polygon.io/stocks/v1/short-interest?ticker={tk}&limit=1000&apiKey={KEY}"
    rows = []
    for _ in range(6):                       # follow next_url paging
        try:
            r = json.load(urllib.request.urlopen(url, timeout=20))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2); continue
            return rows
        except Exception:
            return rows
        for x in r.get("results", []):
            rows.append({"ticker": tk, "date": x.get("settlement_date"),
                         "short_interest": x.get("short_interest"),
                         "avg_daily_volume": x.get("avg_daily_volume"),
                         "days_to_cover": x.get("days_to_cover")})
        nxt = r.get("next_url")
        if not nxt:
            break
        url = nxt + f"&apiKey={KEY}"
    return rows


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from h4_on_signals_study import candidate_windows
    import intraday_data as ID
    allowedC, _ = candidate_windows("C")
    cached = {p.stem for p in (ID.DATA / "4h").glob("*.parquet")}
    names = sorted(set(allowedC) | cached)      # C names + any cached-4h name (reusable archive)
    print(f"backfilling short interest for {len(names)} names...", flush=True)
    n_rows = n_ok = 0
    from pathlib import Path
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, tk in enumerate(names):
            rows = fetch(tk)
            if rows:
                n_ok += 1
                for r in rows:
                    if r["date"] and r["short_interest"] is not None:
                        f.write(json.dumps(r) + "\n"); n_rows += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(names)}  names_with_data={n_ok}  rows={n_rows}", flush=True)
            time.sleep(0.05)
    print(f"DONE: {n_ok}/{len(names)} names, {n_rows} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
