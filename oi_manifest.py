#!/usr/bin/env python3
"""Build the option-OI fetch MANIFEST: the exact (ticker, date) pairs where the C dip-buy fires, so the
ThetaData reconstruction only pulls each dip's option chain once (~1-2k calls) instead of every day of
history. -> .data/oi_manifest.json  {ticker: [dates...]}.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/oi_manifest.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import h4_study as H

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    import intraday_data as ID
    from h4_on_signals_study import candidate_windows
    allowedC, _ = candidate_windows("C")
    man = {}
    n = 0
    for tk in sorted(allowedC):
        df = ID.get_4h(tk, 5, False)
        if df is None or len(df) < 120:
            continue
        c = df["Close"].values
        ts = df.index
        ad = allowedC[tk]
        fire = np.zeros(len(c), dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        dates = sorted({ts[i].date().isoformat() for i in H._episode_starts([j for j in range(len(c)) if fire[j]], gap=H.GAP)
                        if ts[i].date() in ad})
        if dates:
            man[tk] = dates
            n += len(dates)
    out = Path("/app/.data/oi_manifest.json")
    out.write_text(json.dumps({"n_names": len(man), "n_dips": n, "manifest": man}, indent=2))
    # also emit distinct dates per name range for sanity
    alld = sorted({d for ds in man.values() for d in ds})
    print(f"manifest: {len(man)} names, {n} dip dates, span {alld[0]}..{alld[-1]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
