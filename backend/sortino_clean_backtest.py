#!/usr/bin/env python
"""CLEAN Sortino-DD backtest — rebuilt after the union-calendar bug ([[sortino-union-calendar-bug]]).
EVERYTHING is computed PER-TICKER on each name's OWN trading calendar (indicators + Pine state machine),
then the portfolio is aggregated onto a master date grid from own-calendar daily returns (no gap artifacts).
Supersedes the union-calendar BacktestResult[sortino_dd_scanner].

Outputs: headline forward-return stats, round-trip, TAIL vs non-tail, Ulcer-floor sweep, cap x sector
segmentation, portfolio books (ALL-EW / max5 / max10 / TAIL-only) vs SPY/QQQ. Saves BacktestResult +
/app/.data/studies/sortino_clean_backtest.json + sortino_clean_entries.csv.
"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
os.environ["MKT_GATE"] = "0"
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S

S.ULCER_MIN = 0.0                          # base full-BUY = cross + dd>0 (Ulcer applied as tail flag / sweep)
WARM = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
HZ = [5, 10, 21, 42, 63]
MCAP_FLOOR = 1e9
START = pd.Timestamp(S.START_DATE)

b = S.pickle.load(open(S.PANELS, "rb"))
uni = list(b["common"]); mcap_m = b.get("mktcap_usd")
del_sector = b["delisted_sector"]; surv_sector = b["surv_sector"]; bankrupt = set(b["bankrupt_tk"])
sec_of = {t: (surv_sector.get(t) or del_sector.get(t)) for t in uni}
cd = S.load_candles(uni)

entries = []           # per-trade dicts (attribution)
trips_pf = []          # (entry_date, exit_date, dates[], rets[], ulcer, is_tail, is_bank) for the portfolio
all_dates = set()

for tk in uni:
    d = cd.get(tk)
    if d is None or "Close" not in d: continue
    c = d["Close"].dropna(); c = c[c > 0]
    if len(c) < WARM + 30: continue
    idx = c.index
    o = d["Open"].reindex(idx).ffill(); h = d["High"].reindex(idx).ffill()
    l = d["Low"].reindex(idx).ffill(); v = d["Volume"].reindex(idx).fillna(0.0)
    ohlc4 = (o + h + l + c) / 4.0
    ind = S.compute_indicators(c.to_frame(tk), ohlc4.to_frame(tk))
    srsi = ind["srsi"][tk].to_numpy(); se = ind["srsi_ema"][tk].to_numpy()
    nr = ind["nrsi"][tk].to_numpy(); ne = ind["nrsi_ema"][tk].to_numpy()
    dd = ind["dd_score"][tk].to_numpy(); ulc = ind["ulcer"][tk].to_numpy()
    disthi = ind["dist_hi"][tk].to_numpy()
    cl = c.to_numpy(); valid = np.isfinite(cl)
    ret = c.pct_change().to_numpy()
    dvol = (c * v).rolling(63, min_periods=20).mean().to_numpy()
    mom = c.pct_change(S.MOM_WIN).to_numpy()
    mc = mcap_m[tk].reindex(idx, method="ffill").to_numpy() if (mcap_m is not None and tk in mcap_m.columns) else None
    trips = S.run_state_machine(0, cl, srsi, se, nr, ne, dd, ulc, valid, WARM)
    isb = tk in bankrupt
    dts = idx.to_numpy()
    sec = sec_of.get(tk) or "?"
    for (ei, xi_raw, reason, bac) in trips:
        if dts[ei] < START.to_datetime64(): continue
        open_end = xi_raw < 0
        xi = (len(cl) - 1) if open_end else xi_raw
        if xi <= ei: continue
        if not (dvol[ei] >= S.MINVOL): continue
        mcv = mc[ei] if mc is not None else np.nan
        if not (np.isfinite(mcv) and mcv >= MCAP_FLOOR): continue
        cap = S.cap_bucket(mcv)
        ulv, ddv, srv, nrv = ulc[ei], dd[ei], srsi[ei], nr[ei]
        tail = bool(S.tuned_pass(ddv, ulv, bac, cap, srv, nrv, None))
        rt = (cl[xi] / cl[ei] - 1.0) * 100.0
        rec = dict(ticker=tk, sector=sec, cap=cap, date=str(idx[ei].date()),
                   dd_score=float(ddv), ulcer=float(ulv), srsi=float(srv), nrsi=float(nrv),
                   bars_after_cross=int(bac), dist_hi=float(disthi[ei]) if np.isfinite(disthi[ei]) else None,
                   mcap_b=round(float(mcv) / 1e9, 2), mom=float(mom[ei]) if np.isfinite(mom[ei]) else None,
                   dvol_m=round(float(dvol[ei]) / 1e6, 1), rt_ret=round(rt, 2), hold_days=int(xi - ei),
                   exit_reason=("open" if open_end else reason), tail_target=tail, full_buy=True)
        for hz in HZ:
            j = ei + hz
            rec[f"fwd{hz}"] = round((cl[j] / cl[ei] - 1.0) * 100.0, 2) if (j < len(cl) and cl[j] > 0) else None
        entries.append(rec)
        # portfolio path: own-calendar daily returns from entry+1..exit
        seg_d = dts[ei + 1:xi + 1]; seg_r = ret[ei + 1:xi + 1].copy()
        seg_r = np.nan_to_num(seg_r, nan=0.0)
        if isb and open_end and len(seg_r):
            seg_r[-1] = -1.0                      # confirmed-bankrupt still-open -> -100% on last bar
        trips_pf.append(dict(entry=dts[ei], exit=dts[xi], dates=seg_d, rets=seg_r,
                             ulcer=float(ulv) if np.isfinite(ulv) else 0.0, tail=tail))
        all_dates.update(seg_d.tolist())

t = pd.DataFrame(entries)
print(f"CLEAN backtest: {len(t)} BUYs, {t.ticker.nunique()} names, {t.date.min()}..{t.date.max()}\n")

# ---------- master grid + portfolio ----------
master = np.array(sorted(all_dates))
pos = {d: i for i, d in enumerate(master)}
M = len(master)
spyqqq = S.load_candles(["SPY", "QQQ"])

def book(trips, maxpos=0, weight_by=None, tail_only=False):
    ts = [x for x in trips if (x["tail"] or not tail_only)]
    if maxpos > 0:
        ts_sorted = sorted(ts, key=lambda x: x["entry"]); adm = []; ends = []
        for x in ts_sorted:
            ends = [e for e in ends if e > x["entry"]]
            if len(ends) < maxpos:
                adm.append(x); ends.append(x["exit"])
        ts = adm
    dsum = np.zeros(M); wsum = np.zeros(M)
    for x in ts:
        w = x["ulcer"] if weight_by == "ulcer" else 1.0
        p = np.array([pos[d] for d in x["dates"]])
        if len(p) == 0: continue
        np.add.at(dsum, p, w * x["rets"]); np.add.at(wsum, p, w)
    port = np.where(wsum > 0, dsum / wsum, 0.0)
    inv = np.where(wsum > 0)[0]
    if len(inv) < 2: return None
    a, z = inv[0], inv[-1]
    pr = pd.Series(port[a:z + 1], index=pd.to_datetime(master[a:z + 1]))
    eq = (1 + pr).cumprod(); total = float(eq.iloc[-1] - 1) * 100
    yrs = (pr.index[-1] - pr.index[0]).days / 365.25
    cagr = ((eq.iloc[-1]) ** (1 / yrs) - 1) * 100 if yrs > 0 else float("nan")
    sharpe = float(pr.mean() / pr.std() * np.sqrt(252)) if pr.std() > 0 else float("nan")
    dd = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)
    return dict(n=len(ts), total=round(total, 0), cagr=round(cagr, 1), sharpe=round(sharpe, 2),
                maxdd=round(dd, 1), start=str(pr.index[0].date()), end=str(pr.index[-1].date()),
                eq=eq)

def bench(sym, a, z):
    s = spyqqq.get(sym, {}).get("Close")
    if s is None: return None
    s = s.dropna(); s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(z))]
    if len(s) < 2: return None
    return round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 0)

# ---------- reports ----------
def st(df, lab):
    if not len(df): print(f"{lab:<24} (none)"); return
    rt = df.rt_ret.values
    print(f"{lab:<24}{len(df):>6}  mean {np.mean(rt):>+6.2f}%  med {np.median(rt):>+6.2f}%  "
          f"win {np.mean(rt>0)*100:>4.0f}%  big50 {np.mean(rt>=50)*100:>4.1f}%  hold {df.hold_days.mean():>3.0f}d")

print("=== ROUND-TRIP (entry->exit close) ===")
st(t, "ALL full BUYs"); st(t[t["tail_target"]], "  TAIL-targets"); st(t[~t["tail_target"]], "  non-tail BUYs")

print("\n=== FORWARD RETURNS (mean %) ===   " + "  ".join(f"fwd{h}" for h in HZ))
for lab, df in [("ALL", t), ("TAIL", t[t["tail_target"]]), ("non-tail", t[~t["tail_target"]])]:
    print(f"  {lab:<9}" + "  ".join(f"{df[f'fwd{h}'].mean():>+6.1f}" for h in HZ))

print("\n=== ULCER-FLOOR SWEEP (round-trip, among full BUYs) ===")
for u in [0, 5, 10, 14, 20]:
    g = t[t.ulcer >= u]
    if len(g):
        print(f"  ulcer>={u:<3}{len(g):>6}  mean {g.rt_ret.mean():>+6.2f}%  med {g.rt_ret.median():>+6.2f}%  "
              f"win {(g.rt_ret>0).mean()*100:>4.0f}%  big50 {(g.rt_ret>=50).mean()*100:>4.1f}%")

print("\n=== PORTFOLIO BOOKS (clean) ===")
books = {}
specs = [("ALL-EW", dict()), ("max10-EW", dict(maxpos=10)), ("max5-EW", dict(maxpos=5)),
         ("TAIL-only EW", dict(tail_only=True)), ("TAIL max5", dict(maxpos=5, tail_only=True))]
for lab, kw in specs:
    r = book(trips_pf, **kw)
    if r:
        books[lab] = r
        print(f"  {lab:<16}n={r['n']:>6}  total {r['total']:>+9.0f}%  CAGR {r['cagr']:>5.1f}%  "
              f"Sharpe {r['sharpe']:>5.2f}  maxDD {r['maxdd']:>6.1f}%  ({r['start']}..{r['end']})")
ref = books.get("ALL-EW")
if ref:
    sp = bench("SPY", ref["start"], ref["end"]); qq = bench("QQQ", ref["start"], ref["end"])
    print(f"  {'SPY (same span)':<16}total {sp:>+9.0f}%\n  {'QQQ (same span)':<16}total {qq:>+9.0f}%")

print("\n=== cap x sector (round-trip, n>=30) ===")
seg = t.groupby(["cap", "sector"]).agg(n=("rt_ret", "size"), mean=("rt_ret", "mean"),
                                       med=("rt_ret", "median"), win=("rt_ret", lambda x: (x > 0).mean() * 100))
seg = seg[seg.n >= 30].sort_values("mean", ascending=False)
for (cap, sc), r in seg.head(15).iterrows():
    print(f"  {cap:>6} {str(sc)[:14]:>14} n={int(r.n):>4}  mean {r['mean']:>+6.2f}%  med {r['med']:>+6.2f}%  win {r['win']:>3.0f}%")

# ---------- persist ----------
OUT = "/app/.data/studies"; os.makedirs(OUT, exist_ok=True)
def strip(bk): return {k: {kk: vv for kk, vv in v.items() if kk != "eq"} for k, v in bk.items()}
payload = dict(strategy="sortino_clean_backtest (per-ticker own-calendar; union-bug fixed)",
               computed_at=pd.Timestamp.utcnow().isoformat(), n_entries=len(t), n_names=int(t.ticker.nunique()),
               round_trip=dict(all=st.__self__ if False else None),
               headline={f"fwd{h}": {"mean": round(float(t[f'fwd{h}'].mean()), 2)} for h in HZ},
               books=strip(books),
               tail_vs_buy=dict(
                   tail=dict(n=int(t["tail_target"].sum()), mean=round(float(t[t["tail_target"]].rt_ret.mean()), 2),
                             median=round(float(t[t["tail_target"]].rt_ret.median()), 2),
                             win=round(float((t[t["tail_target"]].rt_ret > 0).mean() * 100), 1)),
                   nontail=dict(n=int((~t["tail_target"]).sum()), mean=round(float(t[~t["tail_target"]].rt_ret.mean()), 2),
                                median=round(float(t[~t["tail_target"]].rt_ret.median()), 2),
                                win=round(float((t[~t["tail_target"]].rt_ret > 0).mean() * 100), 1))))
json.dump(payload, open(f"{OUT}/sortino_clean_backtest.json", "w"), indent=1, default=str)
t.to_csv(f"{OUT}/sortino_clean_entries.csv", index=False)
try:
    from core.models import BacktestResult as BR
    from django.utils import timezone as tz
    BR.objects.update_or_create(kind="sortino_clean_backtest", defaults=dict(computed_at=tz.now(), payload=payload))
    print("\nsaved BacktestResult[sortino_clean_backtest]")
except Exception as e:
    print("BR save skipped:", e)
print(f"wrote {OUT}/sortino_clean_backtest.json + sortino_clean_entries.csv")
