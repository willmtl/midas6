import warnings; warnings.filterwarnings("ignore")
import json, os, csv
import data_store
from _fresh_helpers import daily_fresh, weekly_fresh, sector_signal_nomacd, FRESH_WINDOW_D, FRESH_WINDOW_W

top = json.load(open("_top100.json"))
fund = json.load(open(".data/fundamentals/all_fundamentals.json"))["tickers"]
sectors = json.load(open("_signals.json"))
sec_by_name = {r["sector"]: r for r in sectors}

GICS = {
    "Technology": "Technology", "Communication Services": "Communication Services",
    "Healthcare": "Healthcare", "Real Estate": "Real Estate",
    "Consumer Defensive": "Consumer Staples", "Financial Services": "Financials",
    "Industrials": "Industrials", "Utilities": "Utilities", "Energy": "Energy",
    "Basic Materials": "Materials", "Consumer Cyclical": "Consumer Discretionary",
}

rows = []
for tk in top:
    f = fund.get(tk, {})
    y = f.get("dividend_yield") or f.get("trailing_div_yield") or f.get("forward_div_yield")
    cap, pe, gsec = f.get("market_cap"), f.get("pe_ratio"), f.get("sector") or ""
    df = data_store.load_ticker(tk, "5y")
    if df is None or df.empty or len(df) < 60:
        d_state = d_days = d_since = w_state = w_weeks = w_since = None
        d_missing = w_missing = ["no_data"]
    else:
        d_state, d_days, d_since, d_missing = daily_fresh(df)
        w_state, w_weeks, w_since, w_missing = weekly_fresh(df)

    rot = GICS.get(gsec)
    rsig = sec_by_name.get(rot) if rot else None
    rot_sig, rot_fresh = sector_signal_nomacd(rsig)
    safe = (isinstance(y, (int, float)) and 0 < y <= 7.0
            and isinstance(cap, (int, float)) and cap >= 10e9
            and isinstance(pe, (int, float)) and 0 < pe <= 40)
    rows.append({
        "ticker": tk,
        "dividend_yield": round(y, 2) if isinstance(y, (int, float)) else None,
        "market_cap": cap,
        "pe_ratio": round(pe, 1) if isinstance(pe, (int, float)) else None,
        "gics_sector": gsec,
        "industry": f.get("industry"),
        "current_price": f.get("current_price"),
        "analyst_rating": f.get("analyst_rating"),
        # daily fresh
        "fresh_state": d_state, "fresh_days": d_days, "fresh_since": d_since, "fresh_missing": d_missing,
        # weekly fresh
        "fresh_w_state": w_state, "fresh_w_weeks": w_weeks, "fresh_w_since": w_since, "fresh_w_missing": w_missing,
        # rotation sector cross-ref
        "rotation_sector": rot,
        "rotation_signal": rot_sig,
        "rotation_sector_fresh": rot_fresh,
        "safe_dividend": safe,
    })

payload = {
    "generated_from": "top100 dividend yield (rotation tracked universe)",
    "fundamentals_asof": json.load(open(".data/fundamentals/all_fundamentals.json")).get("fetched_at"),
    "candles_asof": "2026-07-23",
    "safe_filter": "yield<=7% AND market_cap>=$10B AND 0<PE<=40",
    "fresh_windows": {"daily_trading_days": FRESH_WINDOW_D, "weekly_weeks": FRESH_WINDOW_W},
    "count": len(rows),
    "stocks": rows,
}
os.makedirs(".data/dividends", exist_ok=True)
json.dump(payload, open(".data/dividends/top100_dividends.json", "w"), indent=2, default=str)

with open(".data/dividends/top100_dividends.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ticker", "yield", "mktcap_B", "pe", "gics", "fresh_D", "fresh_D_days",
                "fresh_W", "fresh_W_weeks", "rotation_sector", "rotation_signal",
                "rotation_sector_fresh", "safe"])
    for r in rows:
        w.writerow([r["ticker"], r["dividend_yield"],
                    round(r["market_cap"]/1e9, 1) if isinstance(r["market_cap"], (int, float)) else "",
                    r["pe_ratio"], r["gics_sector"], r["fresh_state"] or "", r["fresh_days"] or "",
                    r["fresh_w_state"] or "", r["fresh_w_weeks"] or "",
                    r["rotation_sector"] or "", r["rotation_signal"] or "",
                    r["rotation_sector_fresh"] or "", "Y" if r["safe_dividend"] else ""])

# ---- summary ----
dfresh = [r for r in rows if r["fresh_state"] == "FRESH"]
wfresh = [r for r in rows if r["fresh_w_state"] == "FRESH"]
both = [r for r in rows if r["fresh_state"] == "FRESH" and r["fresh_w_state"] == "FRESH"]
print(f"Saved {len(rows)} -> .data/dividends/top100_dividends.json (+.csv)")
print(f"Windows: daily={FRESH_WINDOW_D} trading days, weekly={FRESH_WINDOW_W} weeks\n")
print(f"DAILY fresh: {len(dfresh)}   WEEKLY fresh: {len(wfresh)}   BOTH: {len(both)}\n")
print("=== FRESH on BOTH daily & weekly (strongest) ===")
for r in sorted(both, key=lambda r: r["dividend_yield"], reverse=True):
    print(f"  {r['ticker']:6} {r['dividend_yield']:5.2f}%  D:{r['fresh_days']}d W:{r['fresh_w_weeks']}w  "
          f"{r['gics_sector'][:20]:20} | sector={r['rotation_signal'] or '-'}")
print("\n=== WEEKLY fresh (all) ===")
for r in sorted(wfresh, key=lambda r: (r['fresh_w_weeks'] if r['fresh_w_weeks'] is not None else 99)):
    d = f"D:{r['fresh_state']}" if r['fresh_state'] else "D:-"
    print(f"  {r['ticker']:6} {r['dividend_yield']:5.2f}%  W:{r['fresh_w_weeks']}w ({r['fresh_w_since']})  "
          f"{r['gics_sector'][:20]:20} | {d} | sector={r['rotation_signal'] or '-'}")
