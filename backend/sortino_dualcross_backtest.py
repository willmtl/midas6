#!/usr/bin/env python
"""DUAL-CROSS strategy + RSI-gate DECOMPOSITION. BUY when Sortino-RSI crosses UP its EMA AND DD crosses
ABOVE 0 within ALIGN_WIN bars. Exit = Pine exit (Sortino down-cross -> OHLC4-RSI confirm -> DD<0; hard DD>=3).
Tests 4 RSI(14)-daily>50 gate variants in one pass: none / SPY-only / stock-only / both. Per-ticker own
calendar. $1B+ liquid. Saves BacktestResult[sortino_dualcross]."""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); os.environ["MKT_GATE"] = "0"
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

ALIGN_WIN = int(os.environ.get("ALIGN_WIN", 5)); RSI_MIN = float(os.environ.get("RSI_MIN", 50))
DD_HARD = S.DD_HARD; WARM = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
HZ = [21, 63]; MCAP_FLOOR = 1e9; START = pd.Timestamp(S.START_DATE)
MODES = ["none", "spy", "stock", "both"]


def dualcross_trips(cl, srsi, se, dd, nrsi, ne, valid, warm, W, prsi, spyrsi, gate):
    n = len(cl); scu = np.zeros(n, bool); scd = np.zeros(n, bool); ncd = np.zeros(n, bool); ddcu = np.zeros(n, bool)
    for i in range(1, n):
        if srsi[i-1] <= se[i-1] and srsi[i] > se[i]: scu[i] = True
        if srsi[i-1] >= se[i-1] and srsi[i] < se[i]: scd[i] = True
        if nrsi[i-1] >= ne[i-1] and nrsi[i] < ne[i]: ncd[i] = True
        if dd[i-1] <= 0 and dd[i] > 0: ddcu[i] = True
    trips = []; in_pos = False; entry = -1; last_sc = -1; last_ddc = -1; warn = False; waiting = False
    for i in range(n):
        if scu[i]: last_sc = i
        if ddcu[i]: last_ddc = i
        if not (valid[i] and i >= warm):
            continue
        if in_pos and scd[i]:
            warn = True; waiting = bool(nrsi[i] > ne[i])
        if in_pos and warn and waiting and ncd[i]: waiting = False
        if in_pos and warn and scu[i]: warn = False; waiting = False
        both = (last_sc >= 0 and last_ddc >= 0 and (i - last_sc) <= W and (i - last_ddc) <= W)
        sok = (not np.isfinite(prsi[i])) or prsi[i] > RSI_MIN     # treat NaN warmup as pass
        mok = (not np.isfinite(spyrsi[i])) or spyrsi[i] > RSI_MIN
        rsi_ok = {"none": True, "spy": mok, "stock": sok, "both": (sok and mok)}[gate]
        buy = (not in_pos) and both and srsi[i] > se[i] and dd[i] > 0 and rsi_ok
        nsell = in_pos and warn and (not waiting) and nrsi[i] < ne[i] and dd[i] < 0
        hsell = in_pos and entry >= 0 and i > entry and dd[i] >= DD_HARD
        if buy:
            in_pos = True; entry = i; last_sc = -1; last_ddc = -1; warn = False; waiting = False
            trips.append([i, -1])
        elif nsell or hsell:
            in_pos = False; trips[-1][1] = i; entry = -1; warn = waiting = False
    return trips


b = S.pickle.load(open(S.PANELS, "rb"))
uni = list(b["common"]); mcap_m = b.get("mktcap_usd"); bankrupt = set(b["bankrupt_tk"])
cd = S.load_candles(uni)
_spc = S.load_candles(["SPY"])["SPY"]["Close"].dropna(); _spc = _spc[_spc > 0]
spy_rsi = S.wilder_rsi(_spc, 14)
ent = {m: [] for m in MODES}; pf = {m: [] for m in MODES}; all_dates = set()
for tk in uni:
    d = cd.get(tk)
    if d is None or "Close" not in d: continue
    c = d["Close"].dropna(); c = c[c > 0]
    if len(c) < WARM + 30: continue
    idx = c.index
    o = d["Open"].reindex(idx).ffill(); h = d["High"].reindex(idx).ffill(); l = d["Low"].reindex(idx).ffill()
    v = d["Volume"].reindex(idx).fillna(0.0); ohlc4 = (o + h + l + c) / 4.0
    ind = S.compute_indicators(c.to_frame(tk), ohlc4.to_frame(tk))
    srsi = ind["srsi"][tk].to_numpy(); se = ind["srsi_ema"][tk].to_numpy()
    nr = ind["nrsi"][tk].to_numpy(); ne = ind["nrsi_ema"][tk].to_numpy()
    dd = ind["dd_score"][tk].to_numpy(); ulc = ind["ulcer"][tk].to_numpy()
    cl = c.to_numpy(); valid = np.isfinite(cl); ret = c.pct_change().to_numpy()
    dvol = (c * v).rolling(63, min_periods=20).mean().to_numpy()
    mc = mcap_m[tk].reindex(idx, method="ffill").to_numpy() if (mcap_m is not None and tk in mcap_m.columns) else None
    dts = idx.to_numpy(); isb = tk in bankrupt
    prsi = S.wilder_rsi(c, 14).to_numpy(); spyrsi = spy_rsi.reindex(idx, method="ffill").to_numpy()
    for m in MODES:
        for (ei, xir) in dualcross_trips(cl, srsi, se, dd, nr, ne, valid, WARM, ALIGN_WIN, prsi, spyrsi, m):
            if dts[ei] < START.to_datetime64(): continue
            openend = xir < 0; xi = (len(cl) - 1) if openend else xir
            if xi <= ei or not (dvol[ei] >= S.MINVOL): continue
            mcv = mc[ei] if mc is not None else np.nan
            if not (np.isfinite(mcv) and mcv >= MCAP_FLOOR): continue
            cap = S.cap_bucket(mcv); rt = (cl[xi] / cl[ei] - 1.0) * 100.0
            tail = bool(S.tuned_pass(dd[ei], ulc[ei], 0, cap, srsi[ei], nr[ei], None))
            rec = dict(ticker=tk, cap=cap, rt_ret=round(rt, 2), hold=int(xi - ei), tail=tail)
            for hz in HZ:
                j = ei + hz
                rec[f"fwd{hz}"] = (cl[j] / cl[ei] - 1.0) * 100.0 if (j < len(cl) and cl[j] > 0) else np.nan
            ent[m].append(rec)
            seg_d = dts[ei + 1:xi + 1]; seg_r = np.nan_to_num(ret[ei + 1:xi + 1], nan=0.0)
            if isb and openend and len(seg_r): seg_r[-1] = -1.0
            pf[m].append(dict(entry=dts[ei], exit=dts[xi], dates=seg_d, rets=seg_r, tail=tail))
            all_dates.update(seg_d.tolist())

master = np.array(sorted(all_dates)); pos = {d: i for i, d in enumerate(master)}; M = len(master)
sq = S.load_candles(["SPY", "QQQ"])

def book(trips, maxpos=0, tail_only=False):
    ts = [x for x in trips if (x["tail"] or not tail_only)]
    if maxpos > 0:
        ts2 = sorted(ts, key=lambda x: x["entry"]); adm = []; ends = []
        for x in ts2:
            ends = [e for e in ends if e > x["entry"]]
            if len(ends) < maxpos: adm.append(x); ends.append(x["exit"])
        ts = adm
    dsum = np.zeros(M); wsum = np.zeros(M)
    for x in ts:
        p = np.array([pos[d] for d in x["dates"]])
        if len(p): np.add.at(dsum, p, x["rets"]); np.add.at(wsum, p, 1.0)
    port = np.where(wsum > 0, dsum / wsum, 0.0); inv = np.where(wsum > 0)[0]
    if len(inv) < 2: return None
    a, z = inv[0], inv[-1]; pr = pd.Series(port[a:z+1], index=pd.to_datetime(master[a:z+1]))
    eq = (1 + pr).cumprod(); yrs = (pr.index[-1] - pr.index[0]).days / 365.25
    return dict(n=len(ts), total=round(float(eq.iloc[-1]-1)*100), cagr=round(((eq.iloc[-1])**(1/yrs)-1)*100, 1) if yrs > 0 else 0,
                sharpe=round(float(pr.mean()/pr.std()*np.sqrt(252)), 2) if pr.std() > 0 else 0,
                maxdd=round(float(((eq-eq.cummax())/eq.cummax()).min()*100), 1))

def bench(sym, a, z):
    s = sq.get(sym, {}).get("Close");
    if s is None: return None
    s = s.dropna(); s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(z))]
    return round(float(s.iloc[-1]/s.iloc[0]-1)*100) if len(s) > 1 else None

LAB = {"none": "no RSI gate", "spy": "SPY RSI>50 only", "stock": "stock RSI>50 only", "both": "both RSI>50"}
print(f"DUAL-CROSS RSI-gate decomposition (align<= {ALIGN_WIN}, RSI(14)>{RSI_MIN:g} daily)\n")
print(f"{'variant':<20}{'trades':>7}{'mean':>7}{'med':>7}{'win%':>6}{'ALL-EW':>8}{'Shrp':>6}{'maxDD':>7}{'max10':>7}{'max5':>7}{'TAIL':>7}")
print("-" * 90)
out = {}
for m in MODES:
    t = pd.DataFrame(ent[m]); rt = t.rt_ret.values
    bk = {k: book(pf[m], **kw) for k, kw in [("all", {}), ("max10", dict(maxpos=10)), ("max5", dict(maxpos=5)), ("tail", dict(tail_only=True))]}
    a = bk["all"]
    print(f"{LAB[m]:<20}{len(t):>7}{np.mean(rt):>+7.2f}{np.median(rt):>+7.2f}{np.mean(rt>0)*100:>6.0f}"
          f"{a['total']:>+8}{a['sharpe']:>6}{a['maxdd']:>7}{bk['max10']['total']:>+7}{bk['max5']['total']:>+7}{bk['tail']['total']:>+7}")
    out[m] = dict(trades=len(t), mean=round(float(np.mean(rt)), 2), median=round(float(np.median(rt)), 2), books={k: v for k, v in bk.items()})
# benchmark span from the 'none' book
_r = out["none"]
print("-" * 90)
sp = bench("SPY", str(pd.to_datetime(master[0]).date()), str(pd.to_datetime(master[-1]).date()))
qq = bench("QQQ", str(pd.to_datetime(master[0]).date()), str(pd.to_datetime(master[-1]).date()))
print(f"{'SPY / QQQ':<20}{'':>7}{'':>7}{'':>7}{'':>6}{sp:>+8}{'':>6}{'':>7}{'':>7}{'':>7}{qq:>+7}")
payload = dict(strategy=f"sortino_dualcross RSI-gate decomposition (align<={ALIGN_WIN}, RSI>{RSI_MIN:g})",
               computed_at=pd.Timestamp.utcnow().isoformat(), align_win=ALIGN_WIN, rsi_min=RSI_MIN,
               variants=out, spy=sp, qqq=qq)
json.dump(payload, open("/app/.data/studies/sortino_dualcross.json", "w"), indent=1, default=str)
try:
    from core.models import BacktestResult as BR
    from django.utils import timezone as tz
    BR.objects.update_or_create(kind="sortino_dualcross", defaults=dict(computed_at=tz.now(), payload=payload))
    print("\nsaved BacktestResult[sortino_dualcross]")
except Exception as e:
    print("BR save skipped:", e)
