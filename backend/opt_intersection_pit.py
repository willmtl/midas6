#!/usr/bin/env python3
"""PIT version of the liquid+low-IV intersection. For each historical flagship pick, as of its trade date:
 (a) IV = nearest OptionSnapshot STRICTLY BEFORE the trade date (no future fallback);
 (b) deep-ITM (~40% ITM) call OI fetched HISTORICALLY via ThetaData option_history_open_interest(date=trade),
     at the standard monthly expiration ~3 months out — liquidity judged as-of that day, not today.
Reports the TRUE share of pick-months that were low-IV AND had a liquid deep-ITM chain at the time.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/opt_intersection_pit.py [--limit N]
"""
import os, sys, json, time, datetime as dt, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from collections import defaultdict
from core.models import OptionSnapshot, Candle
from thetadata import ThetaClient

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
IV_LO = 40.0                 # "low IV" gate
OI_LIQ = 200                 # deep-ITM OI to call the chain tradeable at $100k-$1M scale
OPT_START = dt.date(2022, 9, 1)
c = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def third_friday(y, m):
    d = dt.date(y, m, 1)
    # first Friday, then +2 weeks
    fri = d + dt.timedelta((4 - d.weekday()) % 7)
    return fri + dt.timedelta(14)


def exp_3mo(D):
    m = D.month + 3; y = D.year + (m - 1) // 12; m = (m - 1) % 12 + 1
    return third_friday(y, m)


# --- picks ---
H = json.load(open("/app/.data/studies/flagship_history.json"))
rows = []
for mo in H.get("months", []):
    D = mo.get("date")
    if not D:
        continue
    Dd = dt.date.fromisoformat(str(D)[:10])
    if Dd < OPT_START:
        continue
    for p in (mo.get("picks") or []):
        if p.get("ticker") and p.get("weight"):
            rows.append((Dd, p["ticker"], float(p["weight"])))
if LIMIT:
    rows = rows[:LIMIT]
print(f"pick-months 2022-09+: {len(rows)}", flush=True)

# --- PIT IV (strictly before trade) ---
iv_by_t = defaultdict(list)
for r in OptionSnapshot.objects.filter(atm_iv__isnull=False).values_list("ticker", "date", "atm_iv"):
    iv_by_t[r[0]].append((r[1], float(r[2])))
for t in iv_by_t:
    iv_by_t[t].sort()


def iv_before(t, D):
    ser = iv_by_t.get(t)
    if not ser:
        return None
    prior = [iv for (dte, iv) in ser if dte < D]      # STRICTLY before — no future fallback
    return prior[-1] if prior else None


def spot_at(t, D):
    cd = (Candle.objects.filter(ticker=t, interval="1d", date__lte=D).order_by("-date")
          .values_list("close", flat=True).first())
    return float(cd) if cd else None


def deep_itm_oi(t, D, spot):
    """Historical OI of the ~40%-ITM call at the 3mo monthly expiration, as of date D."""
    exp = exp_3mo(D)
    for attempt in range(3):
        try:
            df = _pd(c.option_history_open_interest(t, exp, date=D, strike="*", right="call"))
            break
        except Exception as ex:
            if "429" in str(ex) or "RESOURCE" in str(ex).upper():
                time.sleep(1.5); continue
            return None
    else:
        return None
    if df is None or not len(df) or "strike" not in df.columns:
        return None
    s = df["strike"].astype(float)
    scale = 1000.0 if s.max() > spot * 20 else 1.0
    df = df.assign(K=s / scale)
    target = spot * 0.6
    i = (df["K"] - target).abs().idxmin()
    return int(df.loc[i].get("open_interest", 0) or 0)


n = len(rows)
have_iv = optioned_hist = lowiv = liquid = viable = 0
wtot = sum(w for _, _, w in rows) or 1.0
w_viable = 0.0
samples = []
for k, (D, t, w) in enumerate(rows):
    iv = iv_before(t, D)
    if iv is not None:
        have_iv += 1
    spot = spot_at(t, D)
    oi = deep_itm_oi(t, D, spot) if spot else None
    time.sleep(0.03)
    if oi is not None and oi > 0:
        optioned_hist += 1
    is_lowiv = iv is not None and iv < IV_LO
    is_liquid = oi is not None and oi >= OI_LIQ
    if is_lowiv:
        lowiv += 1
    if is_liquid:
        liquid += 1
    if is_lowiv and is_liquid:
        viable += 1; w_viable += w
        if len(samples) < 15:
            samples.append((D.isoformat(), t, iv, oi, w))
    if (k + 1) % 40 == 0:
        print(f"  ...{k+1}/{n} processed (viable so far {viable})", flush=True)

print(f"\n=== PIT liquid+low-IV intersection ({n} pick-months, 2022-09+) ===", flush=True)
print(f"  IV known (snapshot strictly pre-trade):     {have_iv} ({have_iv/n*100:.0f}%)", flush=True)
print(f"  had ANY historical deep-ITM call OI>0:      {optioned_hist} ({optioned_hist/n*100:.0f}%)", flush=True)
print(f"  low IV (<{IV_LO:.0f}):                            {lowiv} ({lowiv/n*100:.0f}%)", flush=True)
print(f"  deep-ITM OI>={OI_LIQ} (tradeable):              {liquid} ({liquid/n*100:.0f}%)", flush=True)
print(f"  *** VIABLE (low-IV AND liquid deep-ITM):    {viable} ({viable/n*100:.0f}%) ***", flush=True)
print(f"      as share of BOOK WEIGHT:                {w_viable/wtot*100:.0f}%", flush=True)
print("\n  sample PIT-viable picks (date, ticker, IV, deep-ITM OI, weight):", flush=True)
for s in samples:
    print(f"    {s[0]}  {s[1]:6} IV {s[2]:.0f}%  OI {s[3]:>6}  w={s[4]:.1f}", flush=True)
