#!/usr/bin/env python3
"""Positioning amplifiers for the C dip-buy: SHORT INTEREST + put/call OPEN-INTEREST — bucket the bounce.

A short-term oversold bounce is a squeeze setup, so positioning matters: high short interest = fuel for a
squeeze; put/call OPEN interest = standing options positioning. This buckets the C dip-buy's 3-bar return by
short_pct_float, short_ratio (days-to-cover), and pc_oi (put/call OI), each BOTH WAYS (low..high quintiles),
joined as-of the entry date. Reports coverage honestly (pc_oi history is thin). -> BacktestResult[h4_c_short].
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_short.py
"""
import os, json, bisect, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]


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


def _asof_store(rows, key_fields):
    """rows -> {tk: (sorted_dates, {field:[vals]})} for as-of lookup."""
    st = {}
    for r in rows:
        rec = st.setdefault(r["ticker"], ([], {f: [] for f in key_fields}))
        rec[0].append(r["date"])
        for f in key_fields:
            rec[1][f].append(r[f])
    return st


def _asof(st, tk, d, fields):
    rec = st.get(tk)
    if not rec:
        return None
    i = bisect.bisect_right(rec[0], d) - 1
    return {f: rec[1][f][i] for f in fields} if i >= 0 else None


def _bucket(entries, feat):
    vals = np.array([e[1].get(feat) if e[1].get(feat) is not None else np.nan for e in entries], float)
    rets = np.array([e[0] for e in entries])
    ok = np.isfinite(vals)
    if ok.sum() < 100:
        return {"coverage": int(ok.sum()), "note": "insufficient history to backtest"}
    qs = np.quantile(vals[ok], [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(qs, vals, side="right")
    lab = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    out = {"coverage": int(ok.sum())}
    for q in range(5):
        m = ok & (qi == q)
        out[lab[q]] = {"range": [round(float(vals[m].min()), 2), round(float(vals[m].max()), 2)] if m.any() else None, **_agg(rets[m])}
    return out


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    import intraday_data as ID
    from h4_on_signals_study import candidate_windows
    from core.models import Fundamental, OptionSnapshot

    import datetime as _dt
    allowedC, _ = candidate_windows("C")
    names = sorted(allowedC)
    # DATED short interest from the Polygon backfill archive (.data/short_interest.jsonl); ~8-business-day
    # publication lag baked into the as-of so it's point-in-time (short interest is public ~8d after settlement).
    LAG = _dt.timedelta(days=11)
    si = {}
    try:
        with open("/app/.data/short_interest.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                d = r.get("date")
                if not d or r.get("days_to_cover") is None:
                    continue
                rec = si.setdefault(r["ticker"], ([], {"days_to_cover": [], "short_interest": []}))
                rec[0].append(_dt.date.fromisoformat(d))
                rec[1]["days_to_cover"].append(r["days_to_cover"])
                rec[1]["short_interest"].append(r["short_interest"])
    except FileNotFoundError:
        print("WARN: short_interest.jsonl not found — run fetch_short_interest.py first", flush=True)

    def si_asof(tk, d):
        rec = si.get(tk)
        if not rec:
            return None
        i = bisect.bisect_right(rec[0], d - LAG) - 1        # most recent settlement public by date d
        if i < 0:
            return None
        dtc = rec[1]["days_to_cover"][i]
        prev = rec[1]["short_interest"][i - 1] if i >= 1 else None
        cur = rec[1]["short_interest"][i]
        chg = ((cur / prev - 1) * 100) if (prev and cur is not None and prev > 0) else None
        return {"days_to_cover": dtc, "si_change_pct": chg}

    # put/call OPEN INTEREST — reconstructed from ThetaData (.data/option_oi.jsonl, exact per-dip); fall back
    # to the sparse live OptionSnapshot only if the archive is missing.
    pcoi_theta = {}
    if os.path.exists("/app/.data/option_oi.jsonl"):
        for line in open("/app/.data/option_oi.jsonl"):
            try:
                r = json.loads(line)
                if r.get("pc_oi") is not None:
                    pcoi_theta[(r["ticker"], r["date"])] = r["pc_oi"]
            except Exception:
                pass
    opt = _asof_store(OptionSnapshot.objects.filter(ticker__in=names).exclude(pc_oi=None)
                      .values("ticker", "date", "pc_oi").order_by("ticker", "date"), ["pc_oi"])

    entries = []
    n_names = 0
    for tk in names:
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        n_names += 1
        c = df["Close"].values
        ts = df.index
        n = len(c)
        ad = allowedC[tk]
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        for i in sorted(H._episode_starts([j for j in range(n) if fire[j]], gap=H.GAP)):
            if i + 3 >= n or c[i] <= 0 or ts[i].date() not in ad:
                continue
            feats = {}
            f1 = si_asof(tk, ts[i].date())
            if f1:
                feats.update(f1)
            pk = pcoi_theta.get((tk, ts[i].date().isoformat()))
            if pk is not None:
                feats["pc_oi"] = pk                          # ThetaData reconstructed put/call OI (exact)
            else:
                f2 = _asof(opt, tk, ts[i].date(), ["pc_oi"])
                if f2:
                    feats.update(f2)
            r3 = (c[i + 3] - c[i]) / c[i] * 100
            entries.append((r3, feats))

    FEATS = ["days_to_cover", "si_change_pct", "pc_oi"]
    out = {f: _bucket(entries, f) for f in FEATS}
    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "n_names": n_names, "n_entries": len(entries),
               "by_feature": out,
               "note": ("C oversold-dip 3-bar return bucketed by positioning: days_to_cover + si_change_pct "
                        "(DATED short interest from Polygon backfill, as-of with 11d publication lag) and pc_oi "
                        "(put/call OPEN interest, OptionSnapshot — thin history). Each both ways (Q1 low..Q5 high). "
                        "Hypothesis: high days-to-cover / rising short interest into a dip = squeeze fuel. Gross of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_short.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_short",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_short]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)
    print(f"\n=== POSITIONING AMPLIFIERS on C dip-buy ({len(entries)} entries) ===", flush=True)
    for f in FEATS:
        d = out[f]
        if "Q1 low" not in d:
            print(f"  {f}: coverage {d.get('coverage')} — {d.get('note')}", flush=True); continue
        print(f"  {f}  (coverage {d['coverage']}):", flush=True)
        for k in ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]:
            v = d[k]
            print(f"     {k:8} avg {(v['avg'] or 0):>+.3f}%  win {v['win']}%  t {v['t']}  n {v['n']}  rng {v['range']}", flush=True)


if __name__ == "__main__":
    main()
