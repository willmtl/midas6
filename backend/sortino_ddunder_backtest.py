#!/usr/bin/env python
"""Test: BUY the DD-cross-above-0 only when DD had been BELOW 0 for a LONG time first (regime flip, not a
1-bar wiggle). Dual-cross base (Sortino-RSI cross + DD cross within ALIGN_WIN), NO RSI gate. Sweeps the
minimum consecutive bars DD must have been <0 before the cross. Per-ticker own calendar, $1B+ liquid."""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); os.environ["MKT_GATE"] = "0"
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

ALIGN_WIN = int(os.environ.get("ALIGN_WIN", 5)); DD_HARD = S.DD_HARD
WARM = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
MCAP_FLOOR = 1e9; START = pd.Timestamp(S.START_DATE)
UNDERS = [0, 10, 20, 40, 60]        # min consecutive bars DD<0 required before the cross-up


def trips_for(cl, srsi, se, dd, nrsi, ne, valid, warm, W, under_min):
    n = len(cl); scu = np.zeros(n, bool); scd = np.zeros(n, bool); ncd = np.zeros(n, bool); ddcu = np.zeros(n, bool)
    run_neg = 0
    for i in range(1, n):
        if srsi[i-1] <= se[i-1] and srsi[i] > se[i]: scu[i] = True
        if srsi[i-1] >= se[i-1] and srsi[i] < se[i]: scd[i] = True
        if nrsi[i-1] >= ne[i-1] and nrsi[i] < ne[i]: ncd[i] = True
        if dd[i-1] <= 0 and dd[i] > 0 and run_neg >= under_min:   # cross up AFTER >=under_min bars below 0
            ddcu[i] = True
        run_neg = run_neg + 1 if (np.isfinite(dd[i]) and dd[i] <= 0) else 0
    trips = []; in_pos = False; entry = -1; last_sc = -1; last_ddc = -1; warn = False; waiting = False
    for i in range(n):
        if scu[i]: last_sc = i
        if ddcu[i]: last_ddc = i
        if not (valid[i] and i >= warm): continue
        if in_pos and scd[i]:
            warn = True; waiting = bool(nrsi[i] > ne[i])
        if in_pos and warn and waiting and ncd[i]: waiting = False
        if in_pos and warn and scu[i]: warn = False; waiting = False
        both = (last_sc >= 0 and last_ddc >= 0 and (i - last_sc) <= W and (i - last_ddc) <= W)
        buy = (not in_pos) and both and srsi[i] > se[i] and dd[i] > 0
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
ent = {u: [] for u in UNDERS}; pf = {u: [] for u in UNDERS}; all_dates = set()
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
    for u in UNDERS:
        for (ei, xir) in trips_for(cl, srsi, se, dd, nr, ne, valid, WARM, ALIGN_WIN, u):
            if dts[ei] < START.to_datetime64(): continue
            openend = xir < 0; xi = (len(cl) - 1) if openend else xir
            if xi <= ei or not (dvol[ei] >= S.MINVOL): continue
            mcv = mc[ei] if mc is not None else np.nan
            if not (np.isfinite(mcv) and mcv >= MCAP_FLOOR): continue
            cap = S.cap_bucket(mcv); rt = (cl[xi] / cl[ei] - 1.0) * 100.0
            tail = bool(S.tuned_pass(dd[ei], ulc[ei], 0, cap, srsi[ei], nr[ei], None))
            ent[u].append(dict(rt_ret=rt, tail=tail))
            seg_d = dts[ei + 1:xi + 1]; seg_r = np.nan_to_num(ret[ei + 1:xi + 1], nan=0.0)
            if isb and openend and len(seg_r): seg_r[-1] = -1.0
            pf[u].append(dict(entry=dts[ei], exit=dts[xi], dates=seg_d, rets=seg_r, tail=tail))
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
    return dict(n=len(ts), total=round(float(eq.iloc[-1]-1)*100), sharpe=round(float(pr.mean()/pr.std()*np.sqrt(252)), 2) if pr.std() > 0 else 0,
                maxdd=round(float(((eq-eq.cummax())/eq.cummax()).min()*100), 1))
def bench(sym):
    s = sq.get(sym, {}).get("Close"); s = s.dropna(); s = s[(s.index >= pd.to_datetime(master[0])) & (s.index <= pd.to_datetime(master[-1]))]
    return round(float(s.iloc[-1]/s.iloc[0]-1)*100) if len(s) > 1 else None

print(f"BUY DD-cross-0 after >=N bars UNDER 0 (dual-cross, no RSI gate, align<= {ALIGN_WIN})\n")
print(f"{'N bars under':>13}{'trades':>8}{'mean':>7}{'med':>7}{'win%':>6}{'big50':>7}{'ALL-EW':>8}{'Shrp':>6}{'maxDD':>7}{'max10':>7}{'TAIL':>7}")
print("-" * 86)
out = {}
for u in UNDERS:
    t = pd.DataFrame(ent[u]); rt = t.rt_ret.values
    a = book(pf[u]); b10 = book(pf[u], maxpos=10); tl = book(pf[u], tail_only=True)
    print(f"{u:>13}{len(t):>8}{np.mean(rt):>+7.2f}{np.median(rt):>+7.2f}{np.mean(rt>0)*100:>6.0f}{np.mean(rt>=50)*100:>7.1f}"
          f"{a['total']:>+8}{a['sharpe']:>6}{a['maxdd']:>7}{b10['total']:>+7}{tl['total']:>+7}")
    out[str(u)] = dict(trades=len(t), mean=round(float(np.mean(rt)), 2), median=round(float(np.median(rt)), 2),
                       win=round(float(np.mean(rt > 0)*100), 1), all_ew=a['total'], sharpe=a['sharpe'], max10=b10['total'], tail=tl['total'])
print("-" * 86)
print(f"{'SPY / QQQ':>13}{'':>8}{'':>7}{'':>7}{'':>6}{'':>7}{bench('SPY'):>+8}{'':>6}{'':>7}{'':>7}{bench('QQQ'):>+7}")
payload = dict(strategy=f"sortino DD-cross-0 after N bars under (dual-cross, no RSI, align<={ALIGN_WIN})",
               computed_at=pd.Timestamp.utcnow().isoformat(), unders=out, spy=bench("SPY"), qqq=bench("QQQ"))
json.dump(payload, open("/app/.data/studies/sortino_ddunder.json", "w"), indent=1, default=str)
try:
    from core.models import BacktestResult as BR
    from django.utils import timezone as tz
    BR.objects.update_or_create(kind="sortino_ddunder", defaults=dict(computed_at=tz.now(), payload=payload))
    print("\nsaved BacktestResult[sortino_ddunder]")
except Exception as e:
    print("BR save skipped:", e)
