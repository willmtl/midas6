"""Enrich the flagship-history trace with company names + precomputed aggregates for the HTML document.
ALL fundamentals in the trace are POINT-IN-TIME as of the purchase date (45-day report lag) — never a
current snapshot, never a future close. We deliberately do NOT add a "forward P/E" because we only have
today's estimate, which would be look-ahead on a historical purchase.

Reads  /app/.data/studies/flagship_history.json
Writes /app/.data/studies/flagship_history_enriched.json
Run: docker exec rotation-backend-1 python -u /app/flagship_history_enrich.py
"""
import os, json
import numpy as np
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from pathlib import Path
from collections import defaultdict
from core.models import DelistedCompany

_CK = os.environ.get("CONFIG", "adaptive")
_SUF = "" if _CK == "adaptive" else f"_{_CK}"
IN = Path(f"/app/.data/studies/flagship_history{_SUF}.json")
OUT = Path(f"/app/.data/studies/flagship_history{_SUF}_enriched.json")


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def med_pos(xs):
    """Median over POSITIVE values only — for P/E, where negatives (loss-makers) are 'not meaningful' and
    would drag a mixed median toward zero. The loss-making share is reported separately."""
    xs = [x for x in xs if x is not None and np.isfinite(x) and x > 0]
    return float(np.median(xs)) if xs else None


def main():
    d = json.load(open(IN))
    months = d["months"]

    # names for delisted tickers
    tickers = {p["ticker"] for m in months for p in m["picks"]}
    names = {dc["ticker"]: dc["name"] for dc in
             DelistedCompany.objects.filter(ticker__in=list(tickers)).values("ticker", "name") if dc["name"]}
    for m in months:
        for p in m["picks"]:
            if not p.get("company"):
                p["company"] = names.get(p["ticker"])

    by_stock = defaultdict(lambda: {"ticker": None, "company": None, "sectors": set(), "n": 0, "delisted": False,
                                    "sum_ret": 0.0, "sum_contrib": 0.0, "best": None, "worst": None, "wins": 0,
                                    "conv": 0, "first": None, "last": None, "pe": [], "roe": [], "de": [], "pb": [],
                                    "mc": [], "rev_g": [], "gpa": [], "trades": []})
    by_sector = defaultdict(lambda: {"sector": None, "etf": None, "in_top": 0, "picked": 0, "skipped": 0,
                                     "sum_ret": 0.0, "n_ret": 0})
    curve = []
    eq_f = eq_s = 1.0
    all_picks = []
    n_conviction = total_picks = n_prof = n_small = 0
    allpb = []; allpe = []; allroe = []; allde = []; allmc = []

    for m in months:
        eq_f *= (1 + m["basket_ret"]); eq_s *= (1 + m["spy_ret"])
        curve.append({"date": m["date"], "flagship": round(eq_f, 4), "spy": round(eq_s, 4),
                      "ret": m["basket_ret"], "spy_ret": m["spy_ret"]})
        for s in m["top_sectors"]:
            bs = by_sector[s["etf"]]; bs["sector"] = s["sector"]; bs["etf"] = s["etf"]; bs["in_top"] += 1
        for sk in m["skipped"]:
            by_sector[sk["etf"]]["skipped"] += 1
        for p in m["picks"]:
            if p["ret"] is None:
                continue
            total_picks += 1
            if p["conviction"]:
                n_conviction += 1
            if p.get("roe") is not None and p["roe"] > 0:
                n_prof += 1
            if p.get("mktcap_usd") is not None and p["mktcap_usd"] < 2e9:
                n_small += 1
            allpb.append(p.get("pb")); allpe.append(p.get("pe")); allroe.append(p.get("roe"))
            allde.append(p.get("de")); allmc.append(p.get("mktcap_usd"))
            bs = by_sector[p["etf"]]; bs["sector"] = p["sector"]; bs["etf"] = p["etf"]
            bs["picked"] += 1; bs["sum_ret"] += p["ret"]; bs["n_ret"] += 1
            b = by_stock[p["ticker"]]
            b["ticker"] = p["ticker"]; b["company"] = b["company"] or p.get("company")
            b["sectors"].add(p["sector"]); b["n"] += 1; b["delisted"] = b["delisted"] or p["delisted"]
            b["sum_ret"] += p["ret"]; b["sum_contrib"] += p["ret"] * p["weight"]
            if p["ret"] > 0:
                b["wins"] += 1
            if p["conviction"]:
                b["conv"] += 1
            for key, col in (("pe", "pe"), ("roe", "roe"), ("de", "de"), ("pb", "pb"), ("mktcap_usd", "mc"),
                             ("rev_g", "rev_g"), ("gpa", "gpa")):
                if p.get(key) is not None:
                    b[col].append(p[key])
            if b["first"] is None:
                b["first"] = m["date"]
            b["last"] = m["date"]
            b["best"] = p["ret"] if b["best"] is None else max(b["best"], p["ret"])
            b["worst"] = p["ret"] if b["worst"] is None else min(b["worst"], p["ret"])
            b["trades"].append({"date": m["date"], "sector": p["sector"], "pb": p.get("pb"), "pe": p.get("pe"),
                                "roe": p.get("roe"), "de": p.get("de"), "gpa": p.get("gpa"), "rev_g": p.get("rev_g"),
                                "mc": p.get("mktcap_usd"), "weight": p.get("weight"), "ret": p["ret"],
                                "conviction": p["conviction"]})
            all_picks.append({"date": m["date"], "ticker": p["ticker"], "company": p.get("company"),
                              "sector": p["sector"], "ret": p["ret"], "pb": p.get("pb"), "pe": p.get("pe"),
                              "roe": p.get("roe"), "conviction": p["conviction"], "delisted": p["delisted"]})

    worst_picks = sorted(all_picks, key=lambda x: x["ret"])[:20]
    best_picks = sorted(all_picks, key=lambda x: -x["ret"])[:20]

    # buy-and-hold-till-TODAY per name: return from the FIRST purchase-month close to the latest close
    # (never sold). Uses split/div-adjusted closes; delisted names -> to their last traded price.
    import pandas as pd
    from seq_fundamental_study import load_candles
    _cands = load_candles(list(by_stock.keys()))

    def _hold_today(tk, first_date):
        df = _cands.get(tk)
        if df is None or not len(df):
            return None
        c = df["Close"]
        p0 = c.asof(pd.Timestamp(first_date))
        p1 = c.iloc[-1]
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            return float(p1 / p0 - 1.0)
        return None

    # per-PICK held-to-today: what each best/worst pick did from ITS month to the latest close (never sold).
    # best_picks/worst_picks reference the same dict objects, so they inherit this.
    for p in all_picks:
        p["hold_today"] = _hold_today(p["ticker"], p["date"])

    stocks = []
    for b in by_stock.values():
        stocks.append({"ticker": b["ticker"], "company": b["company"], "sectors": sorted(b["sectors"]),
                       "n": b["n"], "delisted": b["delisted"], "avg_ret": b["sum_ret"] / b["n"],
                       "sum_contrib": b["sum_contrib"], "best": b["best"], "worst": b["worst"],
                       "win_rate": round(100 * b["wins"] / b["n"], 0), "conv": b["conv"],
                       "first": b["first"], "last": b["last"], "med_pe": med(b["pe"]), "med_roe": med(b["roe"]),
                       "med_de": med(b["de"]), "med_pb": med(b["pb"]), "med_mc": med(b["mc"]),
                       "med_rev_g": med(b["rev_g"]), "med_gpa": med(b["gpa"]), "trades": b["trades"],
                       "hold_today": _hold_today(b["ticker"], b["first"])})
    stocks.sort(key=lambda x: -x["sum_contrib"])

    sectors = []
    for b in by_sector.values():
        sectors.append({"sector": b["sector"], "etf": b["etf"], "in_top": b["in_top"], "picked": b["picked"],
                        "skipped": b["skipped"], "avg_pick_ret": (b["sum_ret"] / b["n_ret"]) if b["n_ret"] else None})
    sectors.sort(key=lambda x: -x["picked"])

    # calendar-year returns
    years = {}
    for m in months:
        y = m["date"][:4]
        years.setdefault(y, []).append(m)
    cal = []
    for y in sorted(years):
        mm = years[y]
        s = float(np.prod([1 + x["basket_ret"] for x in mm]) - 1) * 100
        sp = float(np.prod([1 + x["spy_ret"] for x in mm]) - 1) * 100
        cal.append({"year": y, "strategy": round(s, 1), "spy": round(sp, 1), "months": len(mm),
                    "excess": round(s - sp, 1)})

    best_m = max(curve, key=lambda c: c["ret"]); worst_m = min(curve, key=lambda c: c["ret"])
    out = {
        "computed_at": d.get("computed_at"), "perf": d["perf"], "params": d["params"],
        "summary": {"months": len(months), "first": months[0]["date"], "last": months[-1]["date"],
                    "unique_stocks": len(stocks), "unique_sectors_picked": sum(1 for s in sectors if s["picked"]),
                    "total_picks": total_picks, "conviction_picks": n_conviction,
                    "conviction_pct": round(100 * n_conviction / total_picks, 1) if total_picks else 0,
                    "pct_profitable": round(100 * n_prof / total_picks, 1) if total_picks else 0,
                    "pct_smallcap": round(100 * n_small / total_picks, 1) if total_picks else 0,
                    "pct_lossmaking": round(100 * sum(1 for x in allpe if x is not None and x < 0)
                                            / total_picks, 1) if total_picks else 0,
                    "delisted_picks": d["perf"].get("delisted_picks"),
                    "med_pb": med(allpb), "med_pe": med_pos(allpe), "med_roe": med(allroe), "med_de": med(allde),
                    "med_mktcap": med(allmc),
                    "best_month": {"date": best_m["date"], "ret": best_m["ret"]},
                    "worst_month": {"date": worst_m["date"], "ret": worst_m["ret"]},
                    "final_100k_flagship": round(100000 * eq_f), "final_100k_spy": round(100000 * eq_s)},
        "by_stock": stocks, "by_sector": sectors, "curve": curve, "calendar": cal,
        "best_picks": best_picks, "worst_picks": worst_picks, "months": months,
    }
    OUT.write_text(json.dumps(out, default=str))
    S = out["summary"]
    print(f"enriched -> {OUT}")
    print(f"  months {S['months']}  stocks {S['unique_stocks']}  picks {S['total_picks']}  "
          f"profitable {S['pct_profitable']}%  small-cap {S['pct_smallcap']}%")
    print(f"  median at purchase: P/B {S['med_pb']:.2f}  P/E {S['med_pe']:.1f}  ROE {S['med_roe']:.3f}  "
          f"D/E {S['med_de']:.2f}  mktcap ${S['med_mktcap']/1e6:.0f}M")
    print(f"  $100k -> ${S['final_100k_flagship']:,} vs SPY ${S['final_100k_spy']:,}")
    print("  calendar:", {c["year"]: c["strategy"] for c in cal})


if __name__ == "__main__":
    main()
