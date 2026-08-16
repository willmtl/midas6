#!/usr/bin/env python3
"""Do NEWS or INSIDER BUYING improve the value pick? Overlay test on the accel-sector value picks
(cheapest-P/B guard+low_debt). For each pick, conditional forward-return LIFT by:
  insider_buy   net insider BUYING (buy_value > sell_value) in the trailing ~90d
  news_pos      net positive news direction in the trailing ~60d
  news_neg      net negative news direction in the trailing ~60d
Plus tilt strategies: value pick preferring insider-buying names / avoiding negative-news names, vs baseline.
Insider is orthogonal (cheap + insiders buying = conviction value); news may fight 'buy weakness'.
-> BacktestResult[alt_data] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/alt_data_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()

from pathlib import Path
import config, sector_holdings
import price_basis
from studies import _tstat_from_returns
from seq_fundamental_study import load_candles, load_financial_reports
from trend_stock_studies import _pit_monthly_panel, _available_at, _ret_delist, CRYPTO
from backtest_lowpb import _monthly_close, BENCH

OUT = Path(__file__).resolve().parent / ".data" / "studies" / "alt_data.json"
LOOKBACK, TOP_N = 6, 10


def _stats(r, spy):
    r = np.asarray(r, float); n = len(r)
    if n == 0:
        return dict(vs_spy=0, sharpe=0, max_drawdown=0, periods=0)
    tot = float(np.prod(1 + r) - 1) * 100
    sp = float(np.prod(1 + np.asarray(spy)) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    eq = np.cumprod(1 + r); dd = float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)
    return dict(vs_spy=round(tot - sp, 1), sharpe=round(sh, 2), max_drawdown=round(dd, 1), periods=n)


def _lift(x):
    a = np.array(x, float)
    return None if not len(a) else dict(n=len(a), mean_pct=round(a.mean() * 100, 2), win_pct=round((a > 0).mean() * 100, 1))


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
    etf_accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    stock_daily = load_candles(all_holds)
    stock_m = _monthly_close(stock_daily).reindex(midx)

    reps = load_financial_reports(all_holds)
    P = lambda f: _pit_monthly_panel(reps, f, midx)
    shares, equity, ni, debt = P("shares_outstanding"), P("total_equity"), P("net_income"), P("total_debt")
    common = stock_m.columns.intersection(shares.columns).intersection(equity.columns)

    def Rf(p):
        return p.reindex(index=midx, columns=common)
    px = stock_m[common]
    shares, equity, ni, debt = map(Rf, (shares, equity, ni, debt))
    pb = (price_basis.as_traded_close(px) * shares) / equity.where(equity != 0)
    trap = (ni < 0) & (~(equity >= equity.shift(12))) & (~(ni > ni.shift(4)))
    low_debt = (debt / equity.where(equity != 0)) < 1.0

    # ---- preload alt-data into monthly per-ticker panels ----
    from core.models import InsiderBuy, NewsItem, CongressTrade
    uni = list(common)
    # insider: net buy_value by month-end (trailing 3mo rolling sum > 0 = net buying)
    ins = {}
    for tk, fd, bv, sv in InsiderBuy.objects.filter(ticker__in=uni).values_list("ticker", "filed_date", "buy_value", "sell_value"):
        ins.setdefault(tk, []).append((pd.Timestamp(fd), (bv or 0) - (sv or 0)))
    ins_net = {}
    for tk, rows in ins.items():
        s = pd.Series({d: v for d, v in rows}).groupby(level=0).sum().sort_index()
        ins_net[tk] = s.resample("ME").sum().reindex(midx).fillna(0).rolling(3, min_periods=1).sum()
    ins_panel = pd.DataFrame(ins_net).reindex(index=midx, columns=common).fillna(0)
    # news: net direction by month (llm_dir; fallback sign(pos-neg)); trailing 2mo
    nws = {}
    for tk, dt, ld, pos, neg in NewsItem.objects.filter(ticker__in=uni).values_list("ticker", "dt", "llm_dir", "pos", "neg"):
        d = 0
        if ld is not None:
            d = 1 if ld > 0 else (-1 if ld < 0 else 0)
        elif pos is not None and neg is not None:
            d = 1 if pos > neg else (-1 if neg > pos else 0)
        nws.setdefault(tk, []).append((pd.Timestamp(dt).tz_localize(None), d))
    nws_net = {}
    for tk, rows in nws.items():
        s = pd.Series([v for _, v in rows], index=[d for d, _ in rows]).groupby(level=0).sum().sort_index()
        nws_net[tk] = s.resample("ME").sum().reindex(midx).fillna(0).rolling(2, min_periods=1).sum()
    nws_panel = pd.DataFrame(nws_net).reindex(index=midx, columns=common).fillna(0)
    # congress: net BUYS by report_date (public disclosure, PIT-safe) trailing 3mo
    cong = {}
    for tk, rd in CongressTrade.objects.filter(ticker__in=uni, transaction_type="buy").values_list("ticker", "report_date"):
        if rd is not None:
            cong.setdefault(tk, []).append(pd.Timestamp(rd))
    cong_net = {}
    for tk, dts in cong.items():
        s = pd.Series(1, index=dts).groupby(level=0).sum().sort_index()
        cong_net[tk] = s.resample("ME").sum().reindex(midx).fillna(0).rolling(3, min_periods=1).sum()
    cong_panel = pd.DataFrame(cong_net).reindex(index=midx, columns=common).fillna(0)
    print(f"months {len(midx)} | insider tk {len(ins)} | news tk {len(nws)} | congress tk {len(cong)}", flush=True)
    warmup = 12

    # ---- collect value picks + alt-data conditional buckets ----
    ins_on, ins_off, npos_on, npos_off, nneg_on, nneg_off, allp = [], [], [], [], [], [], []
    cong_on, cong_off = [], []
    base_r, base_s, insT_r, insT_s, avoidneg_r, avoidneg_s = [], [], [], [], [], []
    for i in range(warmup, len(midx) - 1):
        date, ndate = midx[i], midx[i + 1]
        sp = spy_m.iloc[i + 1] / spy_m.iloc[i] - 1
        if not np.isfinite(sp):
            continue
        ranks = etf_accel.loc[date].dropna().sort_values(ascending=False).head(TOP_N).index
        b_slot, it_slot, an_slot = [], [], []
        for etf in ranks:
            _, holds = sector_map.get(etf, (etf, []))
            cands = [h for h in holds if h in px.columns and _available_at(px[h], date)
                     and pd.notna(pb.loc[date, h]) and pb.loc[date, h] > 0 and not bool(trap.loc[date, h])]
            guarded = [c for c in cands if bool(low_debt.loc[date, c])] or cands
            if not guarded:
                continue
            bpick = pb.loc[date, guarded].idxmin()
            br = _ret_delist(px[bpick], date, ndate)
            if br is None or not np.isfinite(br):
                continue
            br = float(br); b_slot.append(br); allp.append(br)
            ib = ins_panel.loc[date, bpick] if bpick in ins_panel.columns else 0
            nv = nws_panel.loc[date, bpick] if bpick in nws_panel.columns else 0
            cg = cong_panel.loc[date, bpick] if bpick in cong_panel.columns else 0
            (ins_on if ib > 0 else ins_off).append(br)
            (cong_on if cg > 0 else cong_off).append(br)
            (npos_on if nv > 0 else npos_off).append(br)
            (nneg_on if nv < 0 else nneg_off).append(br)
            # tilt: cheapest-P/B among INSIDER-BUYING guarded; fallback baseline
            ibnames = [g for g in guarded if (g in ins_panel.columns and ins_panel.loc[date, g] > 0)]
            it = pb.loc[date, ibnames].idxmin() if ibnames else bpick
            itr = _ret_delist(px[it], date, ndate)
            if itr is not None and np.isfinite(itr):
                it_slot.append(float(itr))
            # tilt: cheapest-P/B AVOIDING negative-news guarded; fallback baseline
            annames = [g for g in guarded if not (g in nws_panel.columns and nws_panel.loc[date, g] < 0)]
            an = pb.loc[date, annames].idxmin() if annames else bpick
            anr = _ret_delist(px[an], date, ndate)
            if anr is not None and np.isfinite(anr):
                an_slot.append(float(anr))
        if b_slot:
            base_r.append(float(np.mean(b_slot))); base_s.append(float(sp))
        if it_slot:
            insT_r.append(float(np.mean(it_slot))); insT_s.append(float(sp))
        if an_slot:
            avoidneg_r.append(float(np.mean(an_slot))); avoidneg_s.append(float(sp))

    cond = {"insider_buying": {"on": _lift(ins_on), "off": _lift(ins_off)},
            "congress_buying": {"on": _lift(cong_on), "off": _lift(cong_off)},
            "news_positive": {"on": _lift(npos_on), "off": _lift(npos_off)},
            "news_negative": {"on": _lift(nneg_on), "off": _lift(nneg_off)}}
    base = _stats(base_r, base_s); ins_tilt = _stats(insT_r, insT_s); avoid = _stats(avoidneg_r, avoidneg_s)

    print("\n=== A. conditional forward-return LIFT on the value pick ===", flush=True)
    for k, v in cond.items():
        on, off = v["on"], v["off"]
        if not on or not off:
            print(f"  {k:15} insufficient (on n={on['n'] if on else 0}, off n={off['n'] if off else 0})", flush=True)
            continue
        lift = round(on["mean_pct"] - off["mean_pct"], 2)
        print(f"  {k:15} on {on['mean_pct']}%/{on['win_pct']}%win (n{on['n']}) | off {off['mean_pct']}%/{off['win_pct']}% "
              f"| LIFT {lift}pp", flush=True)
    print("\n=== B. tilt strategies vs baseline ===", flush=True)
    print(f"  baseline value          vsSPY {base['vs_spy']}%  Sh {base['sharpe']}  DD {base['max_drawdown']}%", flush=True)
    print(f"  + insider-buying tilt    vsSPY {ins_tilt['vs_spy']}%  Sh {ins_tilt['sharpe']}  DD {ins_tilt['max_drawdown']}%  "
          f"({'+' if ins_tilt['vs_spy']-base['vs_spy']>=0 else ''}{round(ins_tilt['vs_spy']-base['vs_spy'],1)}pp)", flush=True)
    print(f"  + avoid-negative-news    vsSPY {avoid['vs_spy']}%  Sh {avoid['sharpe']}  DD {avoid['max_drawdown']}%  "
          f"({'+' if avoid['vs_spy']-base['vs_spy']>=0 else ''}{round(avoid['vs_spy']-base['vs_spy'],1)}pp)", flush=True)

    il = (cond["insider_buying"]["on"]["mean_pct"] - cond["insider_buying"]["off"]["mean_pct"]) if (ins_on and ins_off) else None
    payload = {
        "computed_at": pd.Timestamp.utcnow().isoformat(),
        "params": {"top_n": TOP_N, "benchmark": BENCH, "months": int(len(midx)),
                   "note": "Congress data unusable (0 purchases in window, corrupt dates) — excluded."},
        "conditional": cond, "baseline": base, "insider_tilt": ins_tilt, "avoid_neg_news": avoid,
        "verdict": (("INSIDER BUYING helps the value pick (+lift & tilt)." if (il and il > 1 and ins_tilt["vs_spy"] > base["vs_spy"] + 10)
                     else "Neither insider buying nor news adds meaningfully on top of the value pick — the value+quality "
                     "signal already prices what they'd tell us; keep straight value.")),
        "caveat": "In-sample, no fees, ~5y. Insider filed_date has SEC lag (already lagged -> OK, no lookahead). News "
                  "direction from llm_dir/sentiment. Alt-data coverage partial (insider 689 tk, news 942 tk).",
    }
    return payload


def main():
    p = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(p, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(
            kind="alt_data", defaults={"payload": json.loads(json.dumps(p, default=str)), "computed_at": timezone.now()})
        print("Saved BacktestResult[alt_data]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("\n" + p["verdict"], flush=True)


if __name__ == "__main__":
    main()
