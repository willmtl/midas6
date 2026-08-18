#!/usr/bin/env python3
"""Reconstruct dealer GAMMA EXPOSURE (gex) per C dip from ThetaData (direct gRPC, VALUE tier). For each dip,
pull per-strike gamma (option_history_greeks_eod, 2nd order) AND open interest across the nearest expirations,
then gex = SUM over contracts of gamma * OI * 100 * spot^2 * 0.01, signed +calls / -puts (dealer-short-put
convention). -> .data/option_gex.jsonl {ticker, date, gex, gex_per_1pct, spot, n_exp}. Resumable.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_option_gex_theta.py
"""
import os, json, time, datetime as dt, warnings
warnings.filterwarnings("ignore")

OUT = "/app/.data/option_gex.jsonl"
MAN = "/app/.data/oi_manifest.json"
MAX_DTE, N_EXP = 90, 6


def _dates(exp_df):
    try:
        col = exp_df.to_pandas()["expiration"] if hasattr(exp_df, "to_pandas") else exp_df["expiration"]
    except Exception:
        return []
    out = []
    for v in col:
        try:
            out.append(dt.date.fromisoformat(str(v)[:10]))
        except Exception:
            pass
    return sorted(set(out))


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    from thetadata import ThetaClient
    c = ThetaClient(api_key=os.environ["THETADATA_API_KEY"])
    man = json.load(open(MAN))["manifest"]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["ticker"], r["date"]))
            except Exception:
                pass
    print(f"gex backfill: {sum(len(v) for v in man.values())} dips, done {len(done)}", flush=True)
    exp_cache = {}
    nrow = ncall = nerr = 0
    with open(OUT, "a") as f:
        for tk in sorted(man):
            try:
                if tk not in exp_cache:
                    exp_cache[tk] = _dates(c.option_list_expirations(tk))
                exps = exp_cache[tk]
            except Exception:
                continue
            for ds in man[tk]:
                if (tk, ds) in done:
                    continue
                d = dt.date.fromisoformat(ds)
                use = [e for e in exps if 0 <= (e - d).days <= MAX_DTE][:N_EXP]
                gex = 0.0; spot = None; nx = 0
                for e in use:
                    try:
                        g = c.option_history_greeks_eod(tk, e, start_date=d, end_date=d, strike="*", right="both")
                        ncall += 1
                        gp = g.to_pandas() if hasattr(g, "to_pandas") else g
                        if not len(gp):
                            continue
                        oi = c.option_history_open_interest(tk, e, date=d, strike="*", right="both")
                        ncall += 1
                        op = oi.to_pandas() if hasattr(oi, "to_pandas") else oi
                        # join gamma (gp) with OI (op) on strike+right
                        gcol = next((x for x in gp.columns if x.lower() == "gamma"), None)
                        scol = next((x for x in gp.columns if x.lower() in ("underlying_price", "spot", "underlying")), None)
                        if gcol is None or not len(op):
                            continue
                        if scol is not None and spot is None:
                            try:
                                spot = float(gp[scol].dropna().iloc[-1])
                            except Exception:
                                pass
                        m = gp[["strike", "right", gcol]].merge(op[["strike", "right", "open_interest"]], on=["strike", "right"], how="inner")
                        if not len(m):
                            continue
                        sgn = m["right"].map(lambda r: 1.0 if str(r).upper().startswith("C") else -1.0)
                        gex += float((m[gcol] * m["open_interest"] * 100.0 * sgn).sum())
                        nx += 1
                    except Exception as ex:
                        if "429" in str(ex) or "RESOURCE" in str(ex).upper():
                            time.sleep(2)
                        nerr += 1
                    time.sleep(0.02)
                if nx:
                    sp = spot or 0.0
                    gex_1pct = gex * (sp ** 2) * 0.01 if sp else None    # $ gamma per 1% move
                    row = {"ticker": tk, "date": ds, "gex_raw": round(gex, 2),
                           "gex_per_1pct": round(gex_1pct, 2) if gex_1pct is not None else None,
                           "spot": sp, "n_exp": nx}
                    f.write(json.dumps(row) + "\n"); f.flush(); nrow += 1
                if nrow and nrow % 100 == 0:
                    print(f"  {nrow} dips, {ncall} calls, {nerr} err (at {tk} {ds})", flush=True)
    print(f"DONE: {nrow} rows, {ncall} calls, {nerr} err -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
