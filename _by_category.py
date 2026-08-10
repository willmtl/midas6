import warnings; warnings.filterwarnings("ignore")
import json, os
import data_fetcher, data_store
from _fresh_helpers import daily_fresh, weekly_fresh, sector_signal_nomacd

full = json.load(open(".data/fundamentals/all_fundamentals.json"))["tickers"]
sectors = json.load(open("_signals.json"))
sec_by_name = {r["sector"]: r for r in sectors}

GICS = {
    "Technology": "Technology", "Communication Services": "Communication Services",
    "Healthcare": "Healthcare", "Real Estate": "Real Estate",
    "Consumer Defensive": "Consumer Staples", "Financial Services": "Financials",
    "Industrials": "Industrials", "Utilities": "Utilities", "Energy": "Energy",
    "Basic Materials": "Materials", "Consumer Cyclical": "Consumer Discretionary",
}

def yld(v):
    return v.get("dividend_yield") or v.get("trailing_div_yield") or v.get("forward_div_yield")

# candidates per sector: safe dividend payers, top 4 by yield
cands = {}
for tk, v in full.items():
    if not isinstance(v, dict) or v.get("sector") not in GICS:
        continue
    y, cap, pe = yld(v), v.get("market_cap"), v.get("pe_ratio")
    if (isinstance(y, (int, float)) and 0 < y <= 7.0
            and isinstance(cap, (int, float)) and cap >= 10e9
            and isinstance(pe, (int, float)) and 0 < pe <= 40):
        cands.setdefault(v["sector"], []).append((tk, y, cap, pe))
for s in cands:
    cands[s].sort(key=lambda r: r[1], reverse=True)
    cands[s] = cands[s][:4]

alltk = sorted({t for lst in cands.values() for (t, *_) in lst})
print(f"Fetching candles for {len(alltk)} candidate tickers...")
data_fetcher.fetch_tickers(alltk, period="5y", interval="1d")

rank = {"FRESH": 2, "POTENTIAL": 1, None: 0}
out = {}
for gics in GICS:
    scored = []
    for (tk, y, cap, pe) in cands.get(gics, []):
        df = data_store.load_ticker(tk, "5y")
        if df is not None and len(df) >= 60:
            d_state, d_days, _, _ = daily_fresh(df)
            w_state, w_weeks, _, _ = weekly_fresh(df)
        else:
            d_state = d_days = w_state = w_weeks = None
        rot_sig, rot_fresh = sector_signal_nomacd(sec_by_name.get(GICS[gics], {}))
        scored.append({
            "ticker": tk, "yield": round(y, 2), "mktcap_B": round(cap/1e9, 1), "pe": round(pe, 1),
            "fresh_D": d_state, "fresh_D_days": d_days, "fresh_W": w_state, "fresh_W_weeks": w_weeks,
            "rotation_signal": rot_sig, "rotation_sector_fresh": rot_fresh,
        })
    # pick best: prefer weekly-fresh, then daily-fresh, then bullish sector, then yield
    scored.sort(key=lambda r: (rank[r["fresh_W"]], rank[r["fresh_D"]],
                               2 if r["rotation_signal"] in ("BULLISH", "ROTATE IN") else 0,
                               r["yield"]), reverse=True)
    out[gics] = {"rotation_sector": GICS[gics], "pick": scored[0] if scored else None, "alternates": scored[1:]}

json.dump(out, open(".data/dividends/by_category.json", "w"), indent=2, default=str)

print("\n=== Best SAFE dividend pick per category (fresh = daily / weekly) ===")
print(f"{'Category (GICS)':24} {'Pick':6} {'Yield':>6} {'Cap':>7} {'PE':>5}  {'FreshD':>8} {'FreshW':>8}  {'Sector':16}")
for gics, d in out.items():
    p = d["pick"]
    if not p:
        print(f"{gics:24} — no safe pick —"); continue
    fd = f"{p['fresh_D']}({p['fresh_D_days']}d)" if p["fresh_D"] == "FRESH" else (p["fresh_D"] or "-")
    fw = f"{p['fresh_W']}({p['fresh_W_weeks']}w)" if p["fresh_W"] == "FRESH" else (p["fresh_W"] or "-")
    print(f"{gics:24} {p['ticker']:6} {p['yield']:5.2f}% {p['mktcap_B']:6.0f}B {p['pe']:5.1f}  {fd:>8} {fw:>8}  "
          f"{(p['rotation_signal'] or '-')}/{p['rotation_sector_fresh'] or '-'}")
    alts = ", ".join(f"{a['ticker']}({a['yield']}%)" for a in d["alternates"])
    if alts:
        print(f"{'':24}   alts: {alts}")
