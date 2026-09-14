#!/usr/bin/env python3
"""PROBE: can we reconstruct the real front-week ATM straddle (=market expected move) the day before an
earnings print from Polygon REST? Test on AAPL 2024-05-02 (realized +8.32%). Steps: list contracts as-of t-1
with expiry just after earnings, pick strike nearest spot, pull each contract's daily bar on t-1,
straddle = call_close + put_close, implied_move% = straddle/spot. Verify before scaling.
Run in the egress container: docker exec rotation-celery-worker-1 python -u /app/probe_straddle.py"""
import os, sys
sys.path.insert(0, "/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from api.tasks import _polygon_get, _polygon_paginate
from core.models import Candle
import datetime as dt

TK = "AAPL"; REPORT = dt.date(2024, 5, 2); TM1 = dt.date(2024, 5, 1)
spot = float(Candle.objects.filter(ticker=TK, interval="1d", date=TM1).values_list("close", flat=True).first())
print(f"{TK}  t-1={TM1}  spot=${spot:.2f}  report={REPORT}", flush=True)

# list contracts alive as-of t-1, expiring in the ~2 weeks after earnings, strikes within +/-8% of spot
lo, hi = round(spot * 0.92, 1), round(spot * 1.08, 1)
contracts = _polygon_paginate(
    "/v3/reference/options/contracts", cap=2000,
    underlying_ticker=TK, as_of=str(TM1),
    **{"expiration_date.gte": str(REPORT + dt.timedelta(days=0)),
       "expiration_date.lte": str(REPORT + dt.timedelta(days=14)),
       "strike_price.gte": lo, "strike_price.lte": hi},
    expired="true", limit=1000)
print(f"contracts returned: {len(contracts)}", flush=True)
if not contracts:
    print("NO CONTRACTS — check endpoint/params/plan", flush=True); sys.exit(1)

exps = sorted(set(c["expiration_date"] for c in contracts))
print(f"expiries available: {exps[:6]}", flush=True)
exp = exps[0]                                   # nearest expiry after earnings
cands = [c for c in contracts if c["expiration_date"] == exp]
# ATM strike = closest to spot
strikes = sorted(set(c["strike_price"] for c in cands))
atm = min(strikes, key=lambda k: abs(k - spot))
print(f"chosen expiry {exp}  ATM strike {atm}", flush=True)


def bar_close(opt_ticker, day):
    d = _polygon_get(f"/v2/aggs/ticker/{opt_ticker}/range/1/day/{day}/{day}", adjusted="true")
    if d and d.get("results"):
        return float(d["results"][0]["c"])
    return None


call = next((c for c in cands if c["strike_price"] == atm and c["contract_type"] == "call"), None)
put = next((c for c in cands if c["strike_price"] == atm and c["contract_type"] == "put"), None)
cc = bar_close(call["ticker"], TM1) if call else None
pc = bar_close(put["ticker"], TM1) if put else None
print(f"call {call['ticker'] if call else '-'}  close={cc}", flush=True)
print(f"put  {put['ticker'] if put else '-'}  close={pc}", flush=True)
if cc and pc:
    straddle = cc + pc
    print(f"\nSTRADDLE = ${straddle:.2f}  ->  IMPLIED MOVE = {straddle/spot*100:.2f}%", flush=True)
    print(f"REALIZED move (t-1->t+1) was +8.32%  |  30d-IV proxy said ~2.11%", flush=True)
