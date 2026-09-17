#!/usr/bin/env python3
"""Per-stock LONG/FLAT timing on the SMC trend state (user: "buy only when blue, exit when red, stock by stock").
State = SMC structure: bullish (BOS/CHoCH up) = BLUE => hold long; bearish = RED => go to cash. Traded
independently per name, entering the bar AFTER the flip (no look-ahead), flip cost 10bps/side. For each stock we
compare the timed strategy to just BUY-AND-HOLDING that stock, then aggregate (win rate, EW portfolio) vs SPY.
Daily and weekly state, swing and internal structure. Saves BacktestResult[smc_state_timing].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/smc_state_timing.py"""
import os, json, math, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from core.models import Candle

SMC_DIR = "/app/.data/tradingview/smc"
OUT = "/app/.data/studies/smc_state_timing.json"
COST = 10.0  # bps per side per flip
UP = {"swing": ["Bullish_BOS", "Bullish_CHoCH"], "internal": ["Internal_Bullish_BOS", "Internal_Bullish_CHoCH"]}
DN = {"swing": ["Bearish_BOS", "Bearish_CHoCH"], "internal": ["Internal_Bearish_BOS", "Internal_Bearish_CHoCH"]}


def nz(v):
    return v is not None and v == v and v != 0 and not (isinstance(v, (int, float)) and abs(v) > 1e50)


def main():
    recs = []
    for f in sorted(glob.glob(SMC_DIR + "/*.json")):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    tickers = [r["ticker"] for r in recs]
    print(f"loaded {len(recs)} SMC files", flush=True)

    from django.db import connection
    with connection.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    close = {}
    for i in range(0, len(tickers + ['SPY']), 100):
        chunk = (tickers + ['SPY'])[i:i + 100]
        q = Candle.objects.filter(ticker__in=chunk, interval="1d", date__gte="2014-06-01").values_list("ticker", "date", "close")
        df = pd.DataFrame(list(q), columns=["ticker", "date", "close"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]); df["close"] = df["close"].astype(float)
        for tk, g in df.groupby("ticker", sort=False):
            close[tk] = g.sort_values("date").set_index("date")["close"].sort_index()
    spy = close.get("SPY")
    spy_ret = spy.pct_change()

    def state_series(rec, tf, kind, cidx):
        """+1 bullish (blue) / -1 bearish (red) per SMC event, ffilled onto the stock's trading-day index."""
        ev = rec.get(tf) or []
        dates, vals, cur = [], [], 0
        for row in sorted(ev, key=lambda x: x.get("t", 0)):
            t = row.get("t")
            if t is None:
                continue
            up = any(nz(row.get(e)) for e in UP[kind]); dn = any(nz(row.get(e)) for e in DN[kind])
            if up and not dn:
                cur = 1
            elif dn and not up:
                cur = -1
            dates.append(pd.Timestamp(int(t), unit="s").normalize()); vals.append(cur)
        if not dates:
            return None
        s = pd.Series(vals, index=pd.DatetimeIndex(dates))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if tf == "weekly":
            s = s.shift(1)   # weekly bar stamped at week-OPEN, value known only at week-CLOSE -> lag one week
                             # so we never trade a week on its own (look-ahead) outcome
        return s.reindex(cidx.union(s.index)).ffill().reindex(cidx).fillna(0)

    def annual(daily_ret):
        r = daily_ret.dropna()
        if len(r) < 252:
            return {}
        eq = (1 + r).prod(); yrs = len(r) / 252.0
        return {"cagr_pct": float((eq ** (1 / yrs) - 1) * 100) if eq > 0 else -100.0,
                "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else 0.0,
                "maxdd_pct": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100)}

    res = {"n_files": len(recs), "cost_bps_per_side": COST, "arms": {}}
    print(f"\n=== SMC blue/red per-stock LONG/FLAT timing vs BUY-&-HOLD ({len(tickers)} mega-caps, {COST}bps/flip) ===", flush=True)
    print(f"{'arm':>22} {'winVsBH%':>9} {'medΔCAGR':>9} {'portCAGR':>9} {'portSh':>7} {'B&HCAGR':>8} {'B&HSh':>6} {'timeInMkt':>9}", flush=True)

    for tf in ("daily", "weekly"):
        for kind in ("swing", "internal"):
            per = []            # (ticker, strat_cagr, bh_cagr, ...)
            strat_rets = []; bh_rets = []; invested = []
            for rec in recs:
                tk = rec["ticker"]
                c = close.get(tk)
                if c is None or c.notna().sum() < 500:
                    continue
                ret = c.pct_change()
                st = state_series(rec, tf, kind, c.index)
                if st is None:
                    continue
                pos = (st == 1).astype(float).shift(1).fillna(0.0)     # long only while blue, enter next bar
                flips = pos.diff().abs().fillna(0.0)
                sret = pos * ret - flips * (COST / 1e4)
                a_s = annual(sret); a_b = annual(ret)
                if a_s and a_b:
                    per.append((tk, a_s["cagr_pct"], a_b["cagr_pct"], a_s["sharpe"], a_b["sharpe"]))
                    strat_rets.append(sret.rename(tk)); bh_rets.append(ret.rename(tk)); invested.append(pos.rename(tk))
            if not per:
                continue
            SR = pd.concat(strat_rets, axis=1); BR = pd.concat(bh_rets, axis=1); IV = pd.concat(invested, axis=1)
            port_s = SR.mean(axis=1); port_b = BR.mean(axis=1)
            win_ret = np.mean([1 for _, sc, bc, _, _ in per if sc > bc]) if per else 0
            win_ret = float(np.mean([sc > bc for _, sc, bc, _, _ in per]) * 100)
            med_d = float(np.median([sc - bc for _, sc, bc, _, _ in per]))
            aps = annual(port_s); apb = annual(port_b)
            tim = float(IV.mean(axis=1).mean() * 100)   # avg fraction of names held (time in market)
            lab = f"{tf}/{kind}"
            res["arms"][lab] = {"n_stocks": len(per), "win_vs_bh_pct": round(win_ret, 1), "median_dCAGR_pp": round(med_d, 2),
                                "port_strat": aps, "port_bh": apb, "time_in_market_pct": round(tim, 1)}
            print(f"{lab:>22} {win_ret:>8.1f}% {med_d:>+8.2f} {aps.get('cagr_pct',0):>+8.1f}% {aps.get('sharpe',0):>6.2f} "
                  f"{apb.get('cagr_pct',0):>+7.1f}% {apb.get('sharpe',0):>5.2f} {tim:>8.1f}%", flush=True)

    res["spy"] = annual(spy_ret)
    print(f"\n  SPY buy&hold: CAGR {res['spy'].get('cagr_pct',0):+.1f}% / Sharpe {res['spy'].get('sharpe',0):.2f}", flush=True)
    print("  (winVsBH% = share of stocks where blue/red timing beat just holding them; portCAGR/Sh = EW portfolio of the timed strategies)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="smc_state_timing", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[smc_state_timing]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
