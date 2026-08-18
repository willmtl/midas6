#!/usr/bin/env python3
"""VOL-SHOCK POP-EXIT overlay on the flagship. User idea: a normally LOW-VOLATILITY holding that suddenly
POPS ~10% is an extreme move for it (a vol shock) — and vol-shock UP moves don't continue (they revert, see
vol-shock-study) — so is it best to EXIT into that pop rather than hold to month-end?

Reuses the flagship picks (flagship_history.json: ticker + buy month-end 'date' + next 'ndate' + realized
monthly 'ret'). For each hold, load the DAILY path over the hold month and the name's TRAILING daily vol
(60d before buy). Two exit triggers, applied (a) to ALL holds and (b) to LOW-VOL holds only (below-median
trailing vol):
  - pop_10 : first day the cumulative return from buy hits +10% -> exit there (bank the pop), cash to月-end
  - pop_ksd: first day a SINGLE-DAY return exceeds k×(trailing daily vol) -> exit at that day's close
Compares the overlay portfolio (mean pick return/month, popped picks replaced by their exit return) vs the
baseline (hold to month-end). -> BacktestResult[vol_pop_exit] + JSON.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/vol_pop_exit_study.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django
django.setup()
from pathlib import Path
from django.db import connection
from studies import _tstat_from_returns

TRACE = Path("/app/.data/studies/flagship_history.json")
OUT = Path("/app/.data/studies/vol_pop_exit.json")


def _compound(rows):
    """rows = list of monthly basket returns (fraction). Returns total % and Sharpe."""
    r = np.asarray(rows, float)
    if len(r) == 0:
        return 0.0, 0.0
    tot = float(np.prod(1 + r) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 1e-9 else 0.0
    return round(tot, 1), round(sh, 2)


def main():
    with connection.cursor() as c:
        c.execute("SET max_parallel_workers_per_gather=0")
    from core.models import Candle, BacktestResult
    from django.utils import timezone

    d = json.load(open(TRACE))
    months = d["months"]
    picks = [(m["date"], m["ndate"], p["ticker"], p["ret"], p["weight"])
             for m in months for p in m["picks"] if p.get("ret") is not None]
    tickers = sorted({p[2] for p in picks})
    print(f"flagship holds: {len(picks)} across {len(months)} months | {len(tickers)} tickers", flush=True)

    # daily close series per ticker
    cand = {}
    for tk in tickers:
        q = list(Candle.objects.filter(ticker=tk, interval="1d").order_by("date").values_list("date", "close"))
        if len(q) > 80:
            cand[tk] = pd.Series({pd.Timestamp(dt): float(c) for dt, c in q}).sort_index()

    CUM_THR = [0.10, 0.15, 0.20, 0.25]     # cumulative pop thresholds
    DAY_THR = [0.10, 0.15, 0.20]           # SINGLE-DAY close-to-close jump thresholds (incl overnight gap)
    rec = []
    for bd, nd, tk, mret, w in picks:
        s = cand.get(tk)
        if s is None:
            continue
        bd, nd = pd.Timestamp(bd), pd.Timestamp(nd)
        hold = s[(s.index > bd) & (s.index <= nd)]
        past = s[s.index <= bd]
        if len(hold) < 2 or len(past) < 40:
            continue
        buy_px = float(past.iloc[-1])
        if buy_px <= 0:
            continue
        tvol = float(past.pct_change().iloc[-60:].std())     # trailing daily vol (for low-vol classification)
        cum = hold / buy_px - 1
        dret = hold.pct_change()                              # close-to-close daily returns (include overnight gap)
        row = dict(bd=str(bd.date()), tk=tk, w=w, tvol=tvol, base=mret)
        for thr in CUM_THR:                                   # exit at first day cumulative >= thr
            hit = cum[cum >= thr]
            row[f"cum_{int(thr*100)}"] = float(hit.iloc[0]) if len(hit) else mret
            row[f"hit_cum_{int(thr*100)}"] = len(hit) > 0
        for thr in DAY_THR:                                   # exit at close of first day with a >= thr single-day jump
            hd = hold[dret >= thr]
            row[f"day_{int(thr*100)}"] = float(hd.iloc[0] / buy_px - 1) if len(hd) else mret
            row[f"hit_day_{int(thr*100)}"] = len(hd) > 0
        rec.append(row)
    df = pd.DataFrame(rec)
    lowvol_cut = df["tvol"].median()
    df["lowvol"] = df["tvol"] <= lowvol_cut
    print(f"holds: {len(df)} | low-vol cutoff {lowvol_cut*100:.2f}%/day", flush=True)

    def portfolio(col, subset=None):
        rows = []
        for bd, g in df.groupby("bd"):
            use_exit = pd.Series(True, index=g.index) if subset is None else g[subset]
            r = np.where(use_exit, g[col], g["base"])
            rows.append(float(np.average(r, weights=g["w"].values)))
        return _compound(rows)

    base_tot, base_sh = portfolio("base")
    results = {"baseline": {"total": base_tot, "sharpe": base_sh}}
    print(f"\n  {'policy (exit into the pop)':40}{'hit%':>6}{'total':>10}{'Sharpe':>8}{'vs base':>9}   revert-diag(low-vol)", flush=True)
    print(f"  {'baseline: hold to month-end':40}{'':>6}{base_tot:>9.1f}%{base_sh:>8}{0.0:>+9.1f}", flush=True)
    specs = ([(f"cum +{int(t*100)}%", f"cum_{int(t*100)}", f"hit_cum_{int(t*100)}") for t in CUM_THR] +
             [(f"1-day +{int(t*100)}% (close-close)", f"day_{int(t*100)}", f"hit_day_{int(t*100)}") for t in DAY_THR])
    for lab, col, hitcol in specs:
        for scope, sub in [("ALL", None), ("LOW-VOL", "lowvol")]:
            t, sh = portfolio(col, sub)
            key = f"{lab} — {scope}"
            results[key] = {"total": t, "sharpe": sh, "vs_base": round(t - base_tot, 1)}
            # revert diagnostic on the low-vol popped subset: exit-ret vs hold-to-month-end
            diag = ""
            if scope == "LOW-VOL":
                pl = df[df["lowvol"] & df[hitcol]]
                if len(pl):
                    ex, ho = pl[col].mean() * 100, pl["base"].mean() * 100
                    diag = (f"n={len(pl):>3}  exit {ex:+5.1f}% vs hold {ho:+5.1f}%  "
                            f"{'REVERTS→exit' if ex > ho + 0.5 else ('CONTINUES→hold' if ho > ex + 0.5 else 'wash')}")
            print(f"  {key:40}{100*df[hitcol].mean():>5.0f}%{t:>9.1f}%{sh:>8}{t-base_tot:>+9.1f}   {diag}", flush=True)
    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_holds": int(len(df)),
               "lowvol_cut_daily_pct": round(lowvol_cut * 100, 3), "results": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    try:
        BacktestResult.objects.update_or_create(kind="vol_pop_exit",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("\nSaved BacktestResult[vol_pop_exit]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print("DONE_VOLPOP", flush=True)


if __name__ == "__main__":
    main()
