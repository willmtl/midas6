#!/usr/bin/env python3
"""Reconstruct historical put/call OPEN INTEREST (pc_oi) per C dip from ThetaData (direct v3 cloud, VALUE
tier — no terminal, no Pro flat-files). For each (ticker, dip-date) in .data/oi_manifest.json, pull the
nearest expirations' OI across all strikes (strike='*', both rights), sum put vs call -> pc_oi. Resumable.
-> .data/option_oi.jsonl  {ticker, date, pc_oi, call_oi, put_oi, n_exp}.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/fetch_option_oi_theta.py [--gex]
"""
import os, json, time, datetime as dt, warnings
warnings.filterwarnings("ignore")

OUT = "/app/.data/option_oi.jsonl"
MAN = "/app/.data/oi_manifest.json"
MAX_DTE = 90          # only near-term expirations (hold the OI/gamma that matters)
N_EXP = 6             # nearest N expirations per dip (caps call volume)


def _dates(exp_df):
    """expiration list -> sorted list of datetime.date."""
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
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--gex", action="store_true"); args = ap.parse_args()
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
    print(f"manifest {sum(len(v) for v in man.values())} dips; already done {len(done)}", flush=True)

    exp_cache = {}
    n_rows = n_calls = n_err = 0
    with open(OUT, "a") as f:
        for tk in sorted(man):
            try:
                if tk not in exp_cache:
                    exp_cache[tk] = _dates(c.option_list_expirations(tk))
                exps = exp_cache[tk]
            except Exception as e:
                print(f"  {tk}: no expirations ({str(e)[:60]})", flush=True); continue
            if not exps:
                continue
            for ds in man[tk]:
                if (tk, ds) in done:
                    continue
                d = dt.date.fromisoformat(ds)
                use = [e for e in exps if 0 <= (e - d).days <= MAX_DTE][:N_EXP]
                call_oi = put_oi = 0
                nx = 0
                for e in use:
                    for attempt in range(3):
                        try:
                            df = c.option_history_open_interest(tk, e, date=d, strike="*", right="both")
                            n_calls += 1
                            pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
                            if len(pdf):
                                g = pdf.groupby("right")["open_interest"].sum()
                                call_oi += int(g.get("CALL", 0)); put_oi += int(g.get("PUT", 0))
                                nx += 1
                            break
                        except Exception as ex:
                            if "429" in str(ex) or "RESOURCE" in str(ex).upper():
                                time.sleep(2); continue
                            n_err += 1; break
                    time.sleep(0.02)
                if call_oi > 0 or put_oi > 0:
                    row = {"ticker": tk, "date": ds, "call_oi": call_oi, "put_oi": put_oi,
                           "pc_oi": round(put_oi / call_oi, 4) if call_oi else None, "n_exp": nx}
                    f.write(json.dumps(row) + "\n"); f.flush(); n_rows += 1
                if n_rows and n_rows % 100 == 0:
                    print(f"  {n_rows} dips written, {n_calls} calls, {n_err} err (at {tk} {ds})", flush=True)
    print(f"DONE: {n_rows} dip rows, {n_calls} calls, {n_err} errors -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
