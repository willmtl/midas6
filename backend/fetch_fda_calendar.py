#!/usr/bin/env python3
"""FDA CATALYST — LIVE forward calendar (makes the validated run-up book deployable). Scrapes the RTTNews FDA/PDUFA
calendar (free, public, egress-confirmed) -> parses company/ticker/drug/decision-date -> /app/.data/fda_forward_calendar.json.
Then the LIVE SCANNER: given today, flag names whose PDUFA date sits in the 42->5 trading-day run-up window (BUY/HOLD),
and those <5 days out (SELL before the binary). Only US/CA-listed tickers (country gate). Idempotent; run on a schedule
to accumulate forward dates as they post. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_fda_calendar.py"""
import os, json, re, datetime as dt, urllib.request
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from core.models import Candle

URL = "https://www.rttnews.com/corpinfo/fdacalendar.aspx"
OUT = "/app/.data/fda_forward_calendar.json"
HDR = {"User-Agent": "Mozilla/5.0 (research; william2@webisoft.com)"}


def usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def parse(html):
    rows = []
    # each entry begins at a "Company Name" cell; slice to the next one
    parts = html.split('data-th="Company Name"')
    for chunk in parts[1:]:
        chunk = chunk[:2500]
        tickers = re.findall(r'/companies/\d+/[^"]+">([A-Z0-9.\-]{1,12})</a>', chunk)
        m_date = re.search(r'(\d{2}/\d{2}/\d{4})', chunk)
        m_drug = re.search(r'data-th="Drug"[^>]*>([^<]+)', chunk)
        company = re.sub(r'\s+', ' ', chunk.split('<br')[0].split('>')[-1]).strip()
        outcome = "approved" if re.search(r'\bApproved\b', chunk) else ("crl" if "Complete Response" in chunk or ">CRL" in chunk else "pending")
        if not tickers or not m_date:
            continue
        try:
            d = dt.datetime.strptime(m_date.group(1), "%m/%d/%Y").date()
        except ValueError:
            continue
        rows.append({"company": company[:60], "tickers": tickers, "drug": (m_drug.group(1).strip() if m_drug else None),
                     "date": d.isoformat(), "outcome": outcome})
    return rows


def main():
    html = urllib.request.urlopen(urllib.request.Request(URL, headers=HDR), timeout=40).read().decode("utf-8", "ignore")
    rows = parse(html)
    today = max((c for c in Candle.objects.filter(ticker="SPY", interval="1d").values_list("date", flat=True)), default=None)
    today = today or dt.date(2026, 9, 20)
    print(f"parsed {len(rows)} RTTNews calendar rows | today={today}", flush=True)

    # pick a tradeable US/CA ticker per row (prefer plain US, else .TO/.V); must have candles
    have = set(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())
    for r in rows:
        tk = next((t for t in r["tickers"] if usca(t) and t in have), None)
        r["ticker"] = tk

    fwd = [r for r in rows if r["ticker"] and dt.date.fromisoformat(r["date"]) >= today and r["outcome"] == "pending"]
    json.dump({"fetched": today.isoformat(), "rows": rows, "forward_tradeable": fwd}, open(OUT, "w"), indent=1, default=str)
    print(f"tradeable US/CA rows: {sum(1 for r in rows if r['ticker'])} | FORWARD (future, pending, tradeable): {len(fwd)}", flush=True)

    # LIVE SCANNER: trading-days to each future PDUFA (approx via calendar days * 5/7); run-up window = 5..42 td (~7..59 cal days)
    def cal_to_td(days):
        return round(days * 5 / 7)
    print(f"\n=== LIVE FDA RUN-UP SCANNER (buy ~42td before PDUFA, sell ~5td before) — {today} ===", flush=True)
    if not fwd:
        print("  (no forward tradeable PDUFA dates in the current RTTNews window — free page is shallow; accumulate via schedule)", flush=True)
    for r in sorted(fwd, key=lambda x: x["date"]):
        cal = (dt.date.fromisoformat(r["date"]) - today).days
        td = cal_to_td(cal)
        if td < 5:
            sig = "SELL (≤5td to print)"
        elif td <= 42:
            sig = "IN WINDOW — HOLD/BUY"
        else:
            sig = f"watch ({td}td out)"
        print(f"  {r['date']} {r['ticker']:6} {str(r['drug'])[:28]:28} PDUFA in {td:>3}td  -> {sig}", flush=True)

    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="fda_forward_calendar", defaults={
            "payload": {"fetched": today.isoformat(), "n_rows": len(rows),
                        "n_forward_tradeable": len(fwd), "forward": fwd}, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[fda_forward_calendar]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
