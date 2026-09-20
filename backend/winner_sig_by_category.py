#!/usr/bin/env python3
"""IS THE UNPREDICTABLE WINNER ONLY IN GROWTH ('ARK') SLEEVES? (user). Re-run the winner-signature (avg percentile-
rank of the ex-post best-forward-return constituent, per candidate feature; 0.5 = random/unpickable) but SPLIT BY
SLEEVE CATEGORY: GROWTH (innovation/hypergrowth) vs VALUE/CYCLICAL (banks/industrials/retail/energy majors) vs
COMMODITY/MINER (gold/silver/oil/uranium/copper). Same method as growth_winners.py, same accel-top-10 gate, US/CA
$5M-liquid pools. If the winner is ~0.5 in ALL categories -> the single winner is unpickable EVERYWHERE (not an ARK
thing); the flagship wins by targeting the UPPER HALF via cheapest-P/B in VALUE sleeves ([[selection-alpha-verified]]),
NOT by picking the winner. Saves BacktestResult[winner_sig_by_category].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/winner_sig_by_category.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import config, sector_holdings
from seq_fundamental_study import load_candles

MIN_DVOL = 5e6; TOP_ACCEL = 10
FEATS = ["mom3", "mom6", "mom12", "rev1", "rsi14", "rs3", "dvol"]
CATS = {
    "GROWTH": ['Semiconductors', 'Biotech', 'Genomics', 'Cybersecurity', 'AI & Robotics', 'Cloud Computing', 'Space',
               'Clean Energy', 'Solar', 'Fintech', 'Electric Vehicles', 'Internet', 'Software', 'Lithium & Battery',
               'Nanotechnology', 'Cannabis', 'Hydrogen', 'Psychedelics', 'E-Commerce', 'Social Media', 'Gaming & Esports'],
    "VALUE_CYCLICAL": ['Energy', 'Financials', 'Regional Banks', 'Homebuilders', 'Transports', 'Airlines', 'Steel',
                       'Materials', 'Retail', 'Insurance', 'MLPs & Pipelines', 'Industrials', 'Real Estate',
                       'Mortgage REITs', 'Consumer Discretionary', 'Consumer Staples', 'Aerospace & Defense'],
    "COMMODITY_MINER": ['Gold', 'Silver', 'Platinum', 'Oil', 'Natural Gas', 'Uranium', 'Copper Miners',
                        'Rare Earth & Critical Minerals', 'Agriculture', 'Timber & Forestry'],
}


def is_usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    sleeves = [s for cat in CATS.values() for s in cat if s in config.SECTOR_ETFS]
    etf_daily = load_candles(sorted({config.SECTOR_ETFS[s] for s in sleeves}) + ["SPY"] + list(config.SECTOR_ETFS.values()))
    etf_m = pd.DataFrame({n: etf_daily[e]["Close"].resample("ME").last() for n, e in config.SECTOR_ETFS.items() if e in etf_daily and etf_daily[e] is not None})
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    midx = etf_m.index[etf_m.index >= "2016-01-01"]
    topaccel = {d: set(accel.loc[d].dropna().nlargest(TOP_ACCEL).index) if d in accel.index and accel.loc[d].notna().any() else set() for d in midx}

    cons = {s: [t for t in sector_holdings.get_holdings(s) if is_usca(t)] for s in sleeves}
    allc = sorted({t for v in cons.values() for t in v})
    cand = {}
    for i in range(0, len(allc), 40):
        cand.update(load_candles(allc[i:i + 40]))
    close_m, dvol_m, rsi_m = {}, {}, {}
    for tk, df in cand.items():
        if df is None or df.empty:
            continue
        close_m[tk] = df["Close"].resample("ME").last().reindex(midx)
        dvol_m[tk] = (df["Close"] * df["Volume"]).resample("ME").mean().reindex(midx)
        rsi_m[tk] = rsi(df["Close"]).resample("ME").last().reindex(midx)
    close_m = pd.DataFrame(close_m); dvol_m = pd.DataFrame(dvol_m); rsi_m = pd.DataFrame(rsi_m)
    fret = close_m.shift(-1) / close_m - 1.0
    mom3 = close_m / close_m.shift(3) - 1; mom6 = close_m / close_m.shift(6) - 1; mom12 = close_m / close_m.shift(12) - 1
    rev1 = close_m / close_m.shift(1) - 1
    etf_ret3 = etf_m.pct_change(3)

    def feat_val(feat, d, t, g):
        if feat == "mom3": return mom3.loc[d, t]
        if feat == "mom6": return mom6.loc[d, t]
        if feat == "mom12": return mom12.loc[d, t]
        if feat == "rev1": return rev1.loc[d, t]
        if feat == "rsi14": return rsi_m.loc[d, t]
        if feat == "dvol": return dvol_m.loc[d, t]
        if feat == "rs3":
            er = etf_ret3.loc[d, g] if g in etf_ret3.columns else np.nan
            return mom3.loc[d, t] - er if pd.notna(mom3.loc[d, t]) and pd.notna(er) else np.nan
        return np.nan

    res = {"params": {"top_accel": TOP_ACCEL, "min_dvol": MIN_DVOL}, "categories": {}}
    print(f"{'category':>16} {'events':>7} " + " ".join(f"{f:>6}" for f in FEATS) + f"  {'winRet':>7} {'sleeveEW':>8}", flush=True)
    for cat, names in CATS.items():
        names = [s for s in names if s in config.SECTOR_ETFS]
        rankacc = {f: [] for f in FEATS}; nev = 0; winrets = []; sew_exc = []
        spy_m = etf_daily["SPY"]["Close"].resample("ME").last().reindex(midx)
        spy_fwd = spy_m.shift(-1) / spy_m - 1.0
        for d in midx[:-1]:
            for g in names:
                if g not in topaccel.get(d, set()):
                    continue
                pool = [t for t in cons[g] if t in close_m.columns and pd.notna(fret.loc[d, t])
                        and (dvol_m.loc[d, t] if pd.notna(dvol_m.loc[d, t]) else 0) >= MIN_DVOL]
                if len(pool) < 3:
                    continue
                nev += 1
                rr = [(t, float(fret.loc[d, t])) for t in pool]
                win, wret = max(rr, key=lambda x: x[1]); winrets.append(wret)
                sew = float(np.nanmean([r for _, r in rr]))
                if pd.notna(spy_fwd.loc[d]):
                    sew_exc.append(sew - float(spy_fwd.loc[d]))
                for f in FEATS:
                    vals = [(t, feat_val(f, d, t, g)) for t in pool if pd.notna(feat_val(f, d, t, g))]
                    if len(vals) < 3 or all(v == vals[0][1] for _, v in vals):
                        continue
                    order = sorted(vals, key=lambda x: x[1]); nm = [t for t, _ in order]
                    if win in nm:
                        rankacc[f].append(nm.index(win) / (len(nm) - 1))
        sig = {f: round(float(np.mean(rankacc[f])), 3) if rankacc[f] else None for f in FEATS}
        res["categories"][cat] = {"events": nev, "winner_signature": sig,
                                  "avg_winner_ret_pct": round(float(np.mean(winrets)) * 100, 1) if winrets else None,
                                  "sleeve_ew_exc_spy_pct": round(float(np.mean(sew_exc)) * 100, 3) if sew_exc else None}
        print(f"{cat:>16} {nev:>7} " + " ".join(f"{sig[f]:>6.3f}" if sig[f] is not None else f"{'-':>6}" for f in FEATS)
              + f"  {np.mean(winrets)*100:>+6.0f}% {res['categories'][cat]['sleeve_ew_exc_spy_pct']:>+7.3f}%", flush=True)
    print("\n(0.5 = winner's rank is RANDOM in that feature = unpickable ex-ante; >0.6 or <0.4 = a real signature)", flush=True)

    open("/app/.data/studies/winner_sig_by_category.json", "w").write(json.dumps(res, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="winner_sig_by_category", defaults={"payload": res, "computed_at": timezone.now()})
        print("Saved BacktestResult[winner_sig_by_category]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
