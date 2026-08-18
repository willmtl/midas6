"""
FINVIZ VERSION — taxonomy config (SEPARATE from the ETF version's config.py / sector_holdings.py).

The "previous version" rotates 91 hand-picked sector/thematic ETFs (config.SECTOR_ETFS) and picks a
cheap value stock from each ETF's ~20 hardcoded holdings. THIS version rotates the full Finviz
taxonomy instead: 11 sectors -> 149 industries -> every US/CA-listed member (scraped by
scrape_finviz_universe.py -> .data/finviz_universe.json, 11,574 names).

Loaded lazily so the file is import-safe even before the scrape has run. Nothing here mutates or
imports the ETF-version config — the two universes evolve independently and are compared head-to-head.
"""
import os
import json

_UNIVERSE = None
_PATHS = [
    "/app/.data/finviz_universe.json",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data", "finviz_universe.json"),
]

# the 11 Finviz sectors (Morningstar/Yahoo scheme), in the canonical order
FINVIZ_SECTORS = [
    "Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial", "Healthcare", "Industrials", "Real Estate", "Technology", "Utilities",
]


def _load():
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = {}
        for p in _PATHS:
            try:
                with open(p) as f:
                    _UNIVERSE = json.load(f)
                break
            except Exception:
                continue
    return _UNIVERSE


def universe() -> dict:
    """The full scraped universe dict (sectors / by_ticker / totals)."""
    return _load()


def industries() -> dict:
    """{ 'Sector > Industry': [tickers...] } across all 11 sectors — the rotation units of this version."""
    u = _load()
    out = {}
    for sec, inds in u.get("sectors", {}).items():
        for ind, info in inds.items():
            out[f"{sec} > {ind}"] = [x["t"] for x in info["tickers"]]
    return out


def industry_members(usca_only=True) -> dict:
    """{ 'Sector > Industry': [tickers] }, optionally filtered to USA/Canada listings (matches the
    flagship's US/CA thesis). Country comes from the Finviz row."""
    u = _load()
    out = {}
    keep = {"USA", "Canada"}
    for sec, inds in u.get("sectors", {}).items():
        for ind, info in inds.items():
            ts = [x["t"] for x in info["tickers"] if (not usca_only or x.get("country") in keep)]
            if ts:
                out[f"{sec} > {ind}"] = ts
    return out


def ticker_meta() -> dict:
    """{ TICKER: {sector, industry, company, country} } — flat lookup."""
    return _load().get("by_ticker", {})


def stats():
    u = _load()
    return u.get("totals", {})
