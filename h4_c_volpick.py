#!/usr/bin/env python3
"""H4 dip-buy on the HIGHEST-VOLATILITY stock inside each C sector (vol-pick, not value-pick).

The synthesis: keep C's sector ROTATION (which sectors are accelerating — gives ~10 names, one per
sector = the diversification that held C's drawdown at -20%), but swap the STOCK pick from cheapest-P/B
to HIGHEST trailing volatility. Marries C's diversification with the volatility edge — aiming for high-vol
return without the broad-universe -57% drawdown. Sectors taken from the saved rotation_history (the
accelerating-sector selection each month); within each, pick the max-vol holding with 4h + $5M dvol.
Run through the cash-aware engine (baseline / steep4x+hedge / vol-parity+hedge) vs C value-pick.
-> BacktestResult[h4_c_volpick]. Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/h4_c_volpick.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import h4_study as H
import h4_c_enhance as E
from h4_c_upside import load_targets, upside_asof, bucket_upside

SIGS = ["mr_rsi_os", "mr_newlow60", "mr_ndown"]
MIN_DVOL, VOLWIN = 5e6, 30


def _cached(years=5):
    """{ticker: (df, vol_series%, dvol_series$)} for every cached-4h name."""
    import intraday_data as ID
    out = {}
    for p in sorted((ID.DATA / "4h").glob("*.parquet")):
        tk = p.stem
        df = ID.get_4h(tk, years, False)
        if df is None or len(df) < 120:
            continue
        close = df["Close"].values
        ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1
        vol = pd.Series(ret, index=df.index).rolling(VOLWIN).std() * np.sqrt(2 * 252) * 100
        dv = (df["Close"] * df["Volume"]).rolling(20).mean() if "Volume" in df else pd.Series(np.inf, index=df.index)
        out[tk] = (df, vol, dv)
    return out


def build_windows(frames):
    """From rotation_history's monthly accelerating sectors, pick the max-vol holding per sector per month."""
    import sector_holdings
    from core.models import BacktestResult
    months = BacktestResult.objects.get(kind="rotation_history").payload["months"]
    dates = [pd.Timestamp(m["date"]).tz_localize("UTC") for m in months]   # 4h index is tz-aware UTC
    hold_cache = {}
    windows = {}
    n_picks = 0
    for i, m in enumerate(months):
        date = dates[i]
        ndate = dates[i + 1] if i + 1 < len(months) else date + pd.Timedelta(days=31)
        sectors = {p["sector"] for p in m["picks"] if p.get("sector")}
        for sec in sectors:
            if sec not in hold_cache:
                hold_cache[sec] = sector_holdings.get_holdings(sec)
            cands = []
            for h in hold_cache[sec]:
                fr = frames.get(h)
                if fr is None:
                    continue
                _, vser, dser = fr
                v = vser.asof(date); dv = dser.asof(date)
                if pd.notna(v) and pd.notna(dv) and dv >= MIN_DVOL and 0 < v <= 300:
                    cands.append((h, float(v)))
            if not cands:
                continue
            pick = max(cands, key=lambda x: x[1])[0]      # HIGHEST volatility in the sector
            windows.setdefault(pick, []).append((date, ndate))
            n_picks += 1
    return windows, n_picks


def collect(windows, frames, store, sectors):
    trades = []
    for tk, spans in windows.items():
        fr = frames.get(tk)
        if fr is None:
            continue
        df, vser, _ = fr
        close = df["Close"].values
        ts = df.index
        n = len(close)
        fire = np.zeros(n, dtype=bool)
        for s in SIGS:
            e, _ = H.SIGNALS[s]["fn"](df); fire |= np.asarray(e, dtype=bool)
        idxs = sorted(H._episode_starts([i for i in range(n) if fire[i]], gap=H.GAP))
        for i in idxs:
            if i + 1 >= n or close[i] <= 0:
                continue
            d = ts[i]
            if not any(s <= d < e for s, e in spans):
                continue
            up = upside_asof(store, tk, d.date(), float(close[i]))
            sched = [(ts[i + b], float(close[i + b] / close[i + b - 1] - 1))
                     for b in range(1, E.MAXHOLD + 1) if i + b < n]
            if not sched:
                continue
            v = vser.iloc[i]
            trades.append({"entry_ts": d, "bucket": bucket_upside(up), "upside": up,
                           "sector": sectors.get(tk, tk), "sched": sched,
                           "vol": float(v) if pd.notna(v) else 60.0})
    return trades


def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    import django; django.setup()
    from pathlib import Path
    frames = _cached(5)
    windows, n_picks = build_windows(frames)
    store = load_targets()
    sectors = E._sector_map()
    daily, spybar = E._spy(5)
    trades = collect(windows, frames, store, sectors)
    print(f"vol-picks: {n_picks} sector-months, {len(windows)} distinct names, {len(trades)} dip trades", flush=True)

    CFGS = {
        "baseline (steep2x+gate)":   {"weight": "steep_2x", "hold": E.HOLD_FIXED, "gate": True},
        "steep4x + hedge50":         {"weight": "steep_4x", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5},
        "vol_parity + hedge50":      {"weight": "vol_parity", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5},
        "vol_parity x conv + hedge": {"weight": "vol_parity_x_conv", "hold": E.HOLD_FIXED, "gate": False, "hedge_frac": 0.5},
    }
    rows = []
    for name, cfg in CFGS.items():
        m = E.simulate(trades, daily, spybar, cfg)
        rows.append({"config": name, **m})
        print(f"  {name:28} total {m['total_return_pct']:>8}%  DD {m['max_dd_pct']:>7} @{m.get('dd_start')}->{m.get('dd_end')}  Sh {m['sharpe']:>5}  [data {m.get('data_start')}]", flush=True)

    payload = {"computed_at": pd.Timestamp.utcnow().isoformat(), "rows": rows,
               "n_sector_month_picks": n_picks, "n_names": len(windows), "n_trades": len(trades),
               "note": ("Highest-vol stock inside each C accelerating sector (vol-pick, not cheapest-P/B). "
                        "Sectors from rotation_history's monthly selection; pick = max trailing-vol holding "
                        "with 4h + $5M dvol. Cash-aware engine. Compares sizing schemes. Gross of fees.")}
    Path("/app/.data/studies").mkdir(parents=True, exist_ok=True)
    Path("/app/.data/studies/h4_c_volpick.json").write_text(json.dumps(payload, indent=2, default=str))
    try:
        from core.models import BacktestResult
        from django.utils import timezone
        BacktestResult.objects.update_or_create(kind="h4_c_volpick",
            defaults={"payload": json.loads(json.dumps(payload, default=str)), "computed_at": timezone.now()})
        print("saved BacktestResult[h4_c_volpick]", flush=True)
    except Exception as e:
        print("DB save failed:", e, flush=True)


if __name__ == "__main__":
    print("=== H4 dip-buy: HIGHEST-VOL stock in C sectors ===", flush=True)
    main()
