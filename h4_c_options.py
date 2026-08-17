#!/usr/bin/env python3
"""Do OPTIONS signals amplify the H4-on-C dip-buy? — bucket the bounce by option-market state at entry.

Analyst-upside was a strong amplifier of the dip-buy (monotone). Options data (OptionSnapshot: atm_iv,
put/call vol & OI, iv_skew, gex, 2022-09..2026, 829 tickers) is another lens on the SAME dip: a capitulatory
put-buying spike / IV blow-off / negative gamma at the low may mark a better (or worse) bounce. This joins
each C oversold-dip entry to the option snapshot as-of that date and buckets the 3-bar return by each option
feature to see which predicts the reversion. -> BacktestResult[h4_c_options].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_options.py
"""
import os, json, bisect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
FEATURES = ["pc_vol", "pc_oi", "atm_iv", "iv_skew", "gex"]


def _t(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 3:
        return None
    sd = a.std(ddof=1)
    return round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else None


def _agg(r):
    r = np.asarray(r, float)
    return {"n": len(r), "avg": round(float(r.mean()), 3) if len(r) else None,
            "win": round(float((r > 0).mean() * 100), 1) if len(r) else None, "t": _t(r)}


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    import intraday_data as ID
    from h4_on_signals_study import candidate_windows
    from core.models import OptionSnapshot

    allowedC, _ = candidate_windows("C")
    names = sorted(allowedC)
    # options as-of store: {tk: (sorted_dates, {feat: [vals]})}
    opt = {}
    for r in (OptionSnapshot.objects.filter(ticker__in=names)
              .values("ticker", "date", *FEATURES).order_by("ticker", "date")):
        tk = r["ticker"]
        rec = opt.setdefault(tk, ([], {f: [] for f in FEATURES}))
        rec[0].append(r["date"])
        for f in FEATURES:
            rec[1][f].append(r[f])

    def asof(tk, d):
        rec = opt.get(tk)
        if not rec:
            return None
        i = bisect.bisect_right(rec[0], d) - 1
        if i < 0:
            return None
        return {f: rec[1][f][i] for f in FEATURES}

    entries = []       # (r3, {feat})
    n_names = covered = 0
    for tk in names:
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        n_names += 1
        c = df["Close"].values
        ts = df.index
        n = len(c)
        ad = allowedC[tk]
        if tk in opt:
            covered += 1
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        for i in sorted(H._episode_starts([j for j in range(n) if fire[j]], gap=H.GAP)):
            if i + H.GAP >= n or c[i] <= 0 or ts[i].date() not in ad or i + 3 >= n:
                continue
            o = asof(tk, ts[i].date())
            if o is None:
                continue
            r3 = (c[i + 3] - c[i]) / c[i] * 100
            entries.append((r3, o))

    rets = np.array([e[0] for e in entries])
    out = {}
    for f in FEATURES:
        vals = np.array([e[1].get(f) if e[1].get(f) is not None else np.nan for e in entries], float)
        ok = np.isfinite(vals)
        if ok.sum() < 100:
            out[f] = {"note": "insufficient coverage", "n": int(ok.sum())}
            continue
        qs = np.quantile(vals[ok], [0.2, 0.4, 0.6, 0.8])
        buckets = {}
        lab = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
        qi = np.searchsorted(qs, vals, side="right")
        for q in range(5):
            m = ok & (qi == q)
            buckets[lab[q]] = {"range": [round(float(vals[m].min()), 3), round(float(vals[m].max()), 3)] if m.any() else None,
                               **_agg(rets[m])}
        out[f] = {"buckets": buckets, "overall": _agg(rets[ok])}

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(),
               "n_names": n_names, "n_opt_covered": covered, "n_entries": len(entries),
               "features": FEATURES, "by_feature": out,
               "note": ("C oversold-dip 3-bar return bucketed by option-market state at entry (OptionSnapshot "
                        "as-of). pc_vol/pc_oi = put/call ratios (high=bearish/fearful positioning), atm_iv = "
                        "implied vol, iv_skew = put-vs-call skew, gex = dealer gamma. Tests which option signal "
                        "predicts the bounce. Options history 2022-09+; gross of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_options.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_options",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_options]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== OPTIONS AMPLIFIER on C dip-buy ({covered}/{n_names} names w/ options, {len(entries)} entries) ===", flush=True)
    for f in FEATURES:
        d = out.get(f, {})
        if "buckets" not in d:
            print(f"  {f}: {d.get('note')}", flush=True); continue
        print(f"  {f}  (overall {d['overall']['avg']}%):", flush=True)
        for k, v in d["buckets"].items():
            print(f"     {k:8} avg {(v['avg'] or 0):>+.3f}%  win {v['win']}%  t {v['t']}  n {v['n']}  rng {v['range']}", flush=True)


if __name__ == "__main__":
    main()
