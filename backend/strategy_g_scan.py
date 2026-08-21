#!/usr/bin/env python3
"""Live Strategy G monthly picks -> .data/strategy_g_picks.json (current month-end G-core holdings).
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/strategy_g_scan.py"""
import json
from pathlib import Path
import pandas as pd
import strategy_g as G

uni, tgts = G.build_universe()
P = G.build_panels(uni, tgts)
d, picks = G.latest_picks(P)
out = dict(computed_at=pd.Timestamp.utcnow().isoformat(), date=d, variant="core", picks=picks)
Path("/app/.data/strategy_g_picks.json").write_text(json.dumps(out, indent=2, default=str))
print(f"wrote {len(picks)} picks for {d}", flush=True)
