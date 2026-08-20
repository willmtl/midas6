#!/usr/bin/env python3
"""Where is the option OPEN INTEREST across the whole chain for the 4 flagship picks? Scan the nearest few
expirations, list every strike with OI>0 (calls+puts), and flag the ATM strike. Answers: is ANY strike liquid?
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/opt_liquidity_probe.py"""
import os, datetime as dt, warnings
warnings.filterwarnings("ignore")
from thetadata import ThetaClient

c = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
PICKS = {"FATE": 2.65, "INSP": 61.69, "GNK": 26.20, "MSEX": 58.80}


def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _dates(exp_df):
    out = []
    for v in _pd(exp_df)["expiration"]:
        try:
            out.append(dt.date.fromisoformat(str(v)[:10]))
        except Exception:
            pass
    return sorted(set(out))


today = dt.date(2026, 8, 20)
for tk, spot in PICKS.items():
    print(f"\n===== {tk}  spot ${spot} =====", flush=True)
    try:
        exps = _dates(c.option_list_expirations(tk))
    except Exception as e:
        print(f"  no expirations: {str(e)[:80]}", flush=True); continue
    near = [e for e in exps if 0 <= (e - today).days <= 160][:5]
    grand = 0
    for exp in near:
        dte = (exp - today).days
        try:
            oi = _pd(c.option_snapshot_open_interest(tk, exp, right="both"))
        except Exception as e:
            print(f"  {exp} ({dte}d): OI fail {str(e)[:60]}", flush=True); continue
        if oi is None or not len(oi) or "open_interest" not in oi.columns:
            continue
        s = oi["strike"].astype(float)
        scale = 1000.0 if s.max() > spot * 20 else 1.0
        oi = oi.assign(K=s / scale)
        live = oi[oi["open_interest"].fillna(0) > 0]
        tot = int(oi["open_interest"].fillna(0).sum())
        grand += tot
        if not len(live):
            print(f"  {exp} ({dte:>3}d): total OI 0", flush=True); continue
        top = live.sort_values("open_interest", ascending=False).head(6)
        parts = [f"${r.K:g}{r.right[0]}={int(r.open_interest)}" for r in top.itertuples()]
        print(f"  {exp} ({dte:>3}d): total OI {tot:>5} | top strikes: {', '.join(parts)}", flush=True)
    print(f"  >>> {tk} total OI across nearest {len(near)} expirations: {grand}", flush=True)
