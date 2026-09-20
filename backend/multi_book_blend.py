#!/usr/bin/env python3
"""#3 MULTI-BOOK PORTFOLIO (user: lower drawdown without sacrificing flagship return). Combine the 3 low-correlation
engines — flagship (small-cap value), book2 (analyst-revision large-cap), book3 (FDA ORIG catalyst) — into one
monthly-rebalanced portfolio and quantify the return/DD tradeoff. Diversification math should lower AGGREGATE DD;
question is how much return it costs (flagship CAGR ~120% dwarfs the others, so any weight off flagship cuts return).
Reports correlations + a weight frontier (100% flagship .. EW .. tilts) + inverse-vol. Missing book-months = 0 (sleeve
in cash when not deployed). Saves BacktestResult[multi_book_blend].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/multi_book_blend.py"""
import os, json, math
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
from book_blend_study import book2_series      # validated PIT tgt_rev_3m large-cap book -> monthly Series


def ser_from(pairs):
    return pd.Series({pd.Timestamp(d): float(v) for d, v in (pairs or [])}).sort_index()


def stat(r):
    r = r.dropna()
    if len(r) < 12:
        return {"n": len(r)}
    eq = (1 + r).prod(); yrs = len(r) / 12.0
    return {"n_mo": len(r), "total_pct": round(float((eq - 1) * 100)),
            "cagr_pct": round(float((eq ** (1 / yrs) - 1) * 100), 1) if eq > 0 else -100.0,
            "sharpe": round(float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else 0.0, 2),
            "maxdd_pct": round(float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min() * 100), 1)}


def main():
    flag = ser_from(json.load(open("/app/.data/studies/flagship_history.json")).get("monthly_net"))
    book2, _ = book2_series()
    fda = ser_from(json.load(open("/app/.data/studies/fda_book.json")).get("orig", {}).get("_monthly"))

    idx = flag.index                                  # rebalance grid = flagship months
    F = flag.reindex(idx)
    B = book2.reindex(idx).fillna(0.0)                # book2 in cash when outside its window
    D = fda.reindex(idx).fillna(0.0)                  # FDA sleeve in cash when no catalyst that month
    df = pd.DataFrame({"flagship": F, "book2": B, "fda": D}).dropna(subset=["flagship"])
    print(f"common window {df.index.min().date()}..{df.index.max().date()} ({len(df)}mo)", flush=True)
    print("standalone:", flush=True)
    for c in ["flagship", "book2", "fda"]:
        s = stat(df[c]); print(f"  {c:9} CAGR {s.get('cagr_pct')}%  Sh {s.get('sharpe')}  DD {s.get('maxdd_pct')}%  total {s.get('total_pct'):,}%", flush=True)
    corr = df.corr()
    print("\ncorrelations:", flush=True)
    print(corr.round(2).to_string(), flush=True)

    def blend(wf, wb, wd):
        w = np.array([wf, wb, wd]); w = w / w.sum()
        return df["flagship"] * w[0] + df["book2"] * w[1] + df["fda"] * w[2]

    # inverse-vol (risk parity) weights
    vols = {c: df[c].std() for c in ["flagship", "book2", "fda"]}
    iv = {c: 1 / vols[c] for c in vols}; sv = sum(iv.values()); ivw = {c: iv[c] / sv for c in iv}

    arms = {
        "100% flagship": (1, 0, 0),
        "90/5/5": (0.90, 0.05, 0.05), "80/10/10": (0.80, 0.10, 0.10), "70/15/15": (0.70, 0.15, 0.15),
        "60/20/20": (0.60, 0.20, 0.20), "equal 1/3": (1, 1, 1),
        f"inv-vol {ivw['flagship']:.2f}/{ivw['book2']:.2f}/{ivw['fda']:.2f}": (ivw["flagship"], ivw["book2"], ivw["fda"]),
    }
    res = {"window": f"{df.index.min().date()}..{df.index.max().date()}", "corr": corr.round(3).to_dict(),
           "standalone": {c: stat(df[c]) for c in ["flagship", "book2", "fda"]}, "blends": {}}
    print(f"\n{'blend (F/B2/FDA)':>26} {'CAGR':>7} {'Sh':>5} {'DD':>7} {'total%':>13}  {'dCAGR':>7} {'dDD':>6}", flush=True)
    base = stat(df["flagship"])
    for lab, w in arms.items():
        s = stat(blend(*w)); res["blends"][lab] = {**s, "weights": w}
        print(f"{lab:>26} {s.get('cagr_pct'):>6.1f}% {s.get('sharpe'):>5.2f} {s.get('maxdd_pct'):>6.1f}% {s.get('total_pct'):>13,}  "
              f"{s.get('cagr_pct')-base['cagr_pct']:>+6.1f} {s.get('maxdd_pct')-base['maxdd_pct']:>+5.1f}", flush=True)

    json.dump(res, open("/app/.data/studies/multi_book_blend.json", "w"), indent=2, default=str)
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="multi_book_blend", defaults={"payload": res, "computed_at": timezone.now()})
        print("\nSaved BacktestResult[multi_book_blend]", flush=True)
    except Exception as e:
        print("save skipped:", e, flush=True)


if __name__ == "__main__":
    main()
