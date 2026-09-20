#!/usr/bin/env python3
"""WHO ACTUALLY WON each growth-sleeve period, and did the winner have a SIGNATURE? (user: "in the period ARK was
the winner, what was the top performer each period?"). For every (month, growth sleeve-in-top-10-accel) event, find
the ex-post best-forward-return constituent (the winner), and record the winner's cross-sectional PERCENTILE RANK
within that event's pool for each candidate feature (mom3/6/12, rev1, rsi14, dvol, rs3). If winners cluster near
rank 1.0 (or 0.0) in some feature, that feature predicts the winner ex-ante; if they scatter ~0.5 (random), the
winner is UNPICKABLE and reverse-engineering is hindsight. Prints the biggest winners + the aggregate signature.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/growth_winners.py"""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import config, sector_holdings
from seq_fundamental_study import load_candles

MIN_DVOL = 5e6; TOP_ACCEL = 10
GROWTH = ['Semiconductors', 'Biotech', 'Genomics', 'Cybersecurity', 'AI & Robotics', 'Cloud Computing', 'Space',
          'Clean Energy', 'Solar', 'Fintech', 'Electric Vehicles', 'Internet', 'Software', 'Lithium & Battery',
          'Nanotechnology', 'Cannabis', 'Hydrogen', 'Psychedelics', 'E-Commerce', 'Social Media',
          'Gaming & Esports', 'Uranium', 'Digital Infrastructure']
FEATS = ["mom3", "mom6", "mom12", "rev1", "rsi14", "rs3", "dvol"]


def is_usca(tk):
    return ("." not in tk) or tk.rsplit(".", 1)[1] in ("TO", "V")


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    growth = [g for g in GROWTH if g in config.SECTOR_ETFS]
    etf_daily = load_candles(sorted({config.SECTOR_ETFS[g] for g in growth}) + ["SPY"] + [config.SECTOR_ETFS[n] for n in config.SECTOR_ETFS])
    etf_m = pd.DataFrame({n: etf_daily[e]["Close"].resample("ME").last() for n, e in config.SECTOR_ETFS.items() if e in etf_daily and etf_daily[e] is not None})
    accel = etf_m.pct_change(3) - etf_m.pct_change(3).shift(3)
    midx = etf_m.index[etf_m.index >= "2016-01-01"]
    topaccel = {d: set(accel.loc[d].dropna().nlargest(TOP_ACCEL).index) if d in accel.index and accel.loc[d].notna().any() else set() for d in midx}

    cons = {g: [t for t in sector_holdings.get_holdings(g) if is_usca(t)] for g in growth}
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

    rows = []
    rankacc = {f: [] for f in FEATS}
    for d in midx[:-1]:
        for g in growth:
            if g not in topaccel.get(d, set()):
                continue
            pool = [t for t in cons[g] if t in close_m.columns and pd.notna(fret.loc[d, t])
                    and (dvol_m.loc[d, t] if pd.notna(dvol_m.loc[d, t]) else 0) >= MIN_DVOL]
            if len(pool) < 3:
                continue
            rr = [(t, float(fret.loc[d, t])) for t in pool]
            win, wret = max(rr, key=lambda x: x[1])
            # winner percentile rank per feature (1.0 = winner had the highest value in the pool)
            wr = {}
            for f in FEATS:
                vals = [(t, feat_val(f, d, t, g)) for t in pool if pd.notna(feat_val(f, d, t, g))]
                if len(vals) < 3 or all(v == vals[0][1] for _, v in vals):
                    continue
                order = sorted(vals, key=lambda x: x[1])           # ascending
                names = [t for t, _ in order]
                if win in names:
                    pct = names.index(win) / (len(names) - 1)       # 0=lowest,1=highest
                    wr[f] = round(pct, 3); rankacc[f].append(pct)
            rows.append({"date": str(d.date()), "sleeve": g, "winner": win, "ret_pct": round(wret * 100, 1),
                         "n_pool": len(pool), "win_rank": wr})

    print(f"{len(rows)} growth-sleeve-in-play events\n", flush=True)
    print("=== 25 BIGGEST WINNERS (ex-post) + where the winner ranked in each feature (1.0=highest in pool) ===", flush=True)
    print(f"{'date':>10} {'sleeve':>16} {'winner':>7} {'ret%':>6} {'mom6':>5} {'rev1':>5} {'rsi14':>5} {'dvol':>5} {'rs3':>5}", flush=True)
    for r in sorted(rows, key=lambda x: -x["ret_pct"])[:25]:
        wr = r["win_rank"]
        print(f"{r['date']:>10} {r['sleeve'][:16]:>16} {r['winner']:>7} {r['ret_pct']:>+6.0f} "
              f"{wr.get('mom6','  -'):>5} {wr.get('rev1','  -'):>5} {wr.get('rsi14','  -'):>5} {wr.get('dvol','  -'):>5} {wr.get('rs3','  -'):>5}", flush=True)

    print(f"\n=== WINNER SIGNATURE — avg percentile-rank of the winner per feature (0.5 = RANDOM/unpredictable) ===", flush=True)
    summary = {}
    for f in FEATS:
        a = np.array(rankacc[f], float)
        if len(a):
            summary[f] = {"mean_pct": round(float(a.mean()), 3), "n": int(len(a)),
                          "top_quintile_share": round(float((a >= 0.8).mean()) * 100, 1)}
            print(f"  {f:>7}: winner avg-rank {a.mean():.3f}  (n={len(a)})  winner-in-top-20%: {(a>=0.8).mean()*100:.0f}%  (random=20%)", flush=True)

    out = {"n_events": len(rows), "winner_signature": summary, "events": rows}
    open("/app/.data/studies/growth_winners.json", "w").write(json.dumps(out, indent=2, default=float))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="growth_winners", defaults={"payload": out, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[growth_winners]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
