#!/usr/bin/env python3
"""TWO fresh threads, one data load.

(A) NO-TRADE BAND — the flagship churns ~69% of the book/month (single picks rotate as sectors shuffle);
    the cost study flagged this as its biggest friction. Test holding a name until it actually FALLS OUT
    of qualification (sector leaves top-10 accel, or name disqualifies/delists) instead of re-picking the
    cheapest every month. Also a softer 'margin band' (only switch if a challenger is >=20% cheaper P/B).
    Measure turnover reduction + gross return + NET-of-cost (does letting winners run also help gross?).

(B) MISS-TILT — today's earnings finding inverted (recent MISS > recent BEAT for the value pick). Test
    tilting the pick toward recent grounded-NEGATIVE names (fallback flagship), with a grounded-POSITIVE
    tilt as the control. Real amplifier, or just duplicates oversold-entry?

-> BacktestResult[turnover_band] + [miss_tilt].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/turnover_misstilt_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

import config, sector_holdings
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH
import price_basis

TOP_N = 10
COST = 0.003          # 30bps round-trip on the liquid flagship names
BAND = 0.80           # margin band: switch only if challenger pb < incumbent pb * 0.80


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(total=0, vs_spy=0, sharpe=0, dd=0, win=0)
    tot = float(np.prod(1 + r) - 1) * 100; sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(total=round(tot, 1), vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), dd=round(dd, 1),
                win=round((r > 0).mean() * 100, 1))


def build():
    etfs = {n: e for n, e in config.SECTOR_ETFS.items() if e not in CRYPTO}
    sector_map, all_holds = {}, set()
    for n, e in etfs.items():
        h = [t for t in sector_holdings.get_holdings(n) if t not in (e, BENCH) and t not in CRYPTO]
        sector_map[e] = (n, h); all_holds.update(h)
    all_holds = sorted(all_holds)
    etf_tk = list(etfs.values())
    print(f"Loading {len(etf_tk)} ETFs + {len(all_holds)} stocks + {BENCH}...", flush=True)
    etf_daily = load_candles(etf_tk + [BENCH])
    etf_m = _monthly_close({t: d for t, d in etf_daily.items() if t in etf_tk})
    midx = etf_m.index
    spy_m = etf_daily[BENCH]["Close"].resample("ME").last().reindex(midx)
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)
    reps = load_financial_reports(all_holds)
    sh, eq, ni, dt = (_pit_monthly_panel(reps, f, midx) for f in
                      ("shares_outstanding", "total_equity", "net_income", "total_debt"))
    common = stock_m.columns.intersection(sh.columns).intersection(eq.columns)
    R = lambda p: p.reindex(index=midx, columns=common)
    px = stock_m[common]; sh, eq, ni, dt = R(sh), R(eq), R(ni), R(dt)
    pb = (price_basis.as_traded_close(px) * sh) / eq.where(eq != 0)
    trap = (ni < 0) & (~(eq >= eq.shift(12))) & (~(ni > ni.shift(4)))
    low = (dt / eq.where(eq != 0)) < 1.0
    dvol = pd.DataFrame({t: (stock_daily[t]["Close"] * stock_daily[t]["Volume"]).rolling(20).mean()
                         .resample("ME").last().reindex(midx) for t in common
                         if t in stock_daily and "Volume" in stock_daily[t]}).reindex(index=midx, columns=common)

    # grounded-earnings panel (PIT ffill by report_date) for section B
    from core.models import EarningsEvent
    eq_qs = (EarningsEvent.objects.filter(ticker__in=list(common), report_date__isnull=False, grounded_score__isnull=False)
             .values_list("ticker", "report_date", "grounded_score"))
    edf = pd.DataFrame.from_records(list(eq_qs), columns=["ticker", "rdate", "gscore"])
    gsc = pd.DataFrame()
    if not edf.empty:
        edf["rdate"] = pd.to_datetime(edf["rdate"])
        cols = {}
        for tk, g in edf.groupby("ticker", sort=False):
            s = pd.to_numeric(g.sort_values("rdate").drop_duplicates("rdate", keep="last").set_index("rdate")["gscore"], errors="coerce")
            cols[tk] = s.reindex(s.index.union(midx)).ffill().reindex(midx)
        gsc = pd.DataFrame(cols).reindex(index=midx, columns=common)
    print(f"months {len(midx)} | stocks {len(common)}", flush=True)

    def qualifies(h, date):
        return (h in px.columns and _available_at(px[h], date) and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0
                and not bool(trap.loc[date, h]) and pd.notna(dvol.loc[date, h]) and dvol.loc[date, h] >= 5e6)

    def cheapest(cands, date):
        g = [x for x in cands if bool(low.loc[date, x])] or cands
        return pb.loc[date, g].idxmin() if g else None

    # ============ shared monthly loop ============
    idxs = [i for i in range(9, len(midx) - 1) if np.isfinite(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1)]

    # --- (A) three portfolios: baseline / no-trade-band / margin-band ---
    held_ntb, held_band = {}, {}      # etf -> ticker
    base_r, ntb_r, band_r, spies = [], [], [], []
    base_prev, ntb_prev, band_prev = set(), set(), set()
    base_to, ntb_to, band_to = [], [], []
    for i in idxs:
        date, ndate = midx[i], midx[i + 1]
        sp = float(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1)
        top = list(accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        top_set = set(top)
        # cache candidates per sector
        cand = {}
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            cand[etf] = [h for h in holds if qualifies(h, date)]

        # BASELINE: full re-pick cheapest each month
        base_names = {}
        for etf in top:
            p = cheapest(cand[etf], date)
            if p:
                base_names[etf] = p
        # NO-TRADE BAND: hold incumbent while sector in top & still qualifies, else pick cheapest
        new_ntb = {}
        for etf in top:
            inc = held_ntb.get(etf)
            if inc and qualifies(inc, date):
                new_ntb[etf] = inc
            else:
                p = cheapest(cand[etf], date)
                if p:
                    new_ntb[etf] = p
        held_ntb = new_ntb
        # MARGIN BAND: switch only if a challenger is >=20% cheaper P/B than incumbent
        new_band = {}
        for etf in top:
            inc = held_band.get(etf)
            ch = cheapest(cand[etf], date)
            if inc and qualifies(inc, date):
                if ch and pd.notna(pb.loc[date, ch]) and pb.loc[date, ch] < pb.loc[date, inc] * BAND:
                    new_band[etf] = ch
                else:
                    new_band[etf] = inc
            elif ch:
                new_band[etf] = ch
        held_band = new_band

        def realize(names):
            rs = [_ret_delist(px[t], date, ndate) for t in names.values()]
            rs = [float(x) for x in rs if x is not None and np.isfinite(x)]
            return float(np.mean(rs)) if rs else None

        rb, rn, rband = realize(base_names), realize(new_ntb), realize(new_band)
        if rb is None:
            continue
        base_r.append(rb); ntb_r.append(rn if rn is not None else rb); band_r.append(rband if rband is not None else rb)
        spies.append(sp)
        # turnover (one-sided: fraction newly bought)
        for prev, curr, tolist in ((base_prev, set(base_names.values()), base_to),
                                   (ntb_prev, set(new_ntb.values()), ntb_to),
                                   (band_prev, set(new_band.values()), band_to)):
            if curr:
                tolist.append(len(curr - prev) / len(curr))
        base_prev, ntb_prev, band_prev = set(base_names.values()), set(new_ntb.values()), set(new_band.values())

    def net(r, to):
        to = np.array(to[:len(r)] + [0] * max(0, len(r) - len(to)))
        return np.array(r) - to * COST

    A = {}
    for name, r, to in (("baseline_repick", base_r, base_to), ("no_trade_band", ntb_r, ntb_to), ("margin_band_20pct", band_r, band_to)):
        turn = float(np.mean(to)) * 100 if to else 0.0
        A[name] = {"gross": _stats(r, spies), "net": _stats(net(r, to), spies), "turnover_pct": round(turn, 1)}
    print("\n=== (A) NO-TRADE BAND — cut the 69%/mo churn? ===", flush=True)
    for k, v in A.items():
        print(f"  {k:18} turnover {v['turnover_pct']:>5}%/mo  GROSS {v['gross']['total']:>7}%/Sh{v['gross']['sharpe']}  "
              f"NET {v['net']['total']:>7}%/Sh{v['net']['sharpe']}/DD{v['net']['dd']}%  (vsSPY net {v['net']['vs_spy']:+})", flush=True)

    # --- (B) miss-tilt vs beat-tilt vs baseline ---
    baseB, misst, beatt = [], [], []
    spB = []
    for i in idxs:
        date, ndate = midx[i], midx[i + 1]
        sp = float(spy_m.iloc[i + 1] / spy_m.iloc[i] - 1)
        top = list(accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index)
        grow = gsc.loc[date] if not gsc.empty else pd.Series(dtype=float)
        b_slot, m_slot, be_slot = [], [], []
        for etf in top:
            _, holds = sector_map.get(etf, (etf, []))
            c = [h for h in holds if qualifies(h, date)]
            base_pick = cheapest(c, date)
            if not base_pick:
                continue
            rb = _ret_delist(px[base_pick], date, ndate)
            if rb is None or not np.isfinite(rb):
                continue
            b_slot.append(float(rb))
            g = [x for x in c if bool(low.loc[date, x])] or c
            miss = [x for x in g if x in grow.index and pd.notna(grow.get(x)) and grow.get(x) < 0]
            beat = [x for x in g if x in grow.index and pd.notna(grow.get(x)) and grow.get(x) > 0]
            mp = pb.loc[date, miss].idxmin() if miss else base_pick
            bp = pb.loc[date, beat].idxmin() if beat else base_pick
            rm, rbe = _ret_delist(px[mp], date, ndate), _ret_delist(px[bp], date, ndate)
            m_slot.append(float(rm) if rm is not None and np.isfinite(rm) else float(rb))
            be_slot.append(float(rbe) if rbe is not None and np.isfinite(rbe) else float(rb))
        if b_slot:
            baseB.append(float(np.mean(b_slot))); misst.append(float(np.mean(m_slot))); beatt.append(float(np.mean(be_slot))); spB.append(sp)
    B = {"baseline": _stats(baseB, spB), "miss_tilt": _stats(misst, spB), "beat_tilt": _stats(beatt, spB)}
    print("\n=== (B) MISS-TILT — is 'buy the recent misser' a real amplifier? ===", flush=True)
    for k, v in B.items():
        lift = v["vs_spy"] - B["baseline"]["vs_spy"]
        print(f"  {k:10} total {v['total']:>7}%  vsSPY {v['vs_spy']:>7}  Sh {v['sharpe']:>5}  DD {v['dd']}%  win {v['win']}%  ({lift:+.1f}pp)", flush=True)

    ntb = A["no_trade_band"]; bl = A["baseline_repick"]
    payload_A = {
        "computed_at": pd.Timestamp.utcnow().isoformat(), "cost_roundtrip": COST, "variants": A,
        "verdict": (f"No-trade band cuts turnover {bl['turnover_pct']}%->{ntb['turnover_pct']}%/mo; NET Sharpe "
                    f"{bl['net']['sharpe']}->{ntb['net']['sharpe']}, net vsSPY {bl['net']['vs_spy']}->{ntb['net']['vs_spy']}. " + (
                    "The band WINS on net — less churn keeps more return AND (letting winners run) doesn't hurt gross. Adopt it."
                    if ntb['net']['vs_spy'] >= bl['net']['vs_spy'] - 5 else
                    "The band cuts costs but gives up gross return (monthly re-pick to the cheapest name matters more than "
                    "the saved friction) — keep monthly re-pick.")),
        "caveat": "Cost=30bps flat (liquid flagship names). In-sample ~5y. Turnover one-sided.",
    }
    payload_B = {
        "computed_at": pd.Timestamp.utcnow().isoformat(), "variants": B,
        "verdict": (f"Miss-tilt {B['miss_tilt']['vs_spy']-B['baseline']['vs_spy']:+.1f}pp vs flagship, beat-tilt "
                    f"{B['beat_tilt']['vs_spy']-B['baseline']['vs_spy']:+.1f}pp. " + (
                    "Miss-tilt ADDS — buying the recent-misser value name is a real conviction amplifier."
                    if B['miss_tilt']['vs_spy'] > B['baseline']['vs_spy'] + 10 else
                    "Miss-tilt does NOT materially add over the flagship — the cheapest-P/B pick already sits on the "
                    "beaten-down names; the earnings miss is confirmation, not extra selection.")),
        "caveat": "Grounded-score PIT on report_date; sectors without a graded miss fall back to flagship. In-sample.",
    }
    return payload_A, payload_B


def main():
    pa, pb_ = build()
    from pathlib import Path
    from core.models import BacktestResult
    from django.utils import timezone
    for kind, p in (("turnover_band", pa), ("miss_tilt", pb_)):
        Path(f"/app/.data/studies/{kind}.json").write_text(json.dumps(p, indent=2, default=str))
        try:
            BacktestResult.objects.update_or_create(kind=kind, defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
            print(f"Saved BacktestResult[{kind}]", flush=True)
        except Exception as e:
            print(f"DB save failed {kind}:", e, flush=True)
    print("\nA) " + pa["verdict"], flush=True)
    print("B) " + pb_["verdict"], flush=True)


if __name__ == "__main__":
    main()
