#!/usr/bin/env python3
"""FDA CATALYST — data step 2. Map openFDA sponsor_name -> our tradeable ticker. Name index = EODHD US symbol list
(active) UNION DelistedCompany (delisted, for survivorship). Conservative normalized matching (strip legal/geo
suffixes only; keep distinctive tokens like PHARMACEUTICAL/THERAPEUTICS). Restrict matches to tickers we have candles
for. Writes /app/.data/fda_ticker_map.json {sponsor_name: ticker} + reports match rate AND — the number that matters —
how many post-2015 approval EVENTS get a tradeable ticker. Unmatched sponsors are logged, not hidden.
Run: MSYS_NO_PATHCONV=1 docker exec rotation-backend-1 python -u /app/build_fda_ticker_map.py"""
import os, json, re
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
from api.tasks import _eodhd_get
from core.models import DelistedCompany, Candle

APPROVALS = "/app/.data/fda_approvals.json"
OUT = "/app/.data/fda_ticker_map.json"
SUFFIX = {"INC", "INCORPORATED", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "LLC", "CO", "COMPANY",
          "LP", "LLP", "SA", "AG", "NV", "GMBH", "AB", "ASA", "SE", "HOLDING", "HOLDINGS", "GROUP",
          "USA", "US", "AMERICA", "NORTH", "AMERICAN", "THE"}


def norm(name):
    if not name:
        return ""
    s = str(name).upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = [t for t in s.split() if t and t not in SUFFIX]
    return " ".join(toks)


def main():
    events = json.load(open(APPROVALS))
    sponsors = {e["sponsor_name"] for e in events if e.get("sponsor_name")}
    have = set(Candle.objects.filter(interval="1d").values_list("ticker", flat=True).distinct())

    # name -> ticker index (prefer tradeable + common stock). EODHD active first, delisted second.
    idx = {}

    def add(name, tk, tradeable, is_cs):
        k = norm(name)
        if not k or not tk:
            return
        prev = idx.get(k)
        score = (2 if tradeable else 0) + (1 if is_cs else 0)
        if prev is None or score > prev[1]:
            idx[k] = (tk, score)

    us = _eodhd_get("exchange-symbol-list/US") or []
    for r in us:
        tk = r.get("Code"); nm = r.get("Name"); typ = r.get("Type")
        if tk and nm:
            add(nm, tk, tk in have, typ == "Common Stock")
    for tk, nm in DelistedCompany.objects.exclude(name="").values_list("ticker", "name"):
        add(nm, tk, tk in have, True)

    print(f"name index: {len(idx)} normalized names | {len(sponsors)} unique sponsors | {len(have)} tickers w/ candles", flush=True)

    # bucket keys by first token to make prefix matching fast (openFDA truncates: "BIOMARIN PHARM" ⊂ "BIOMARIN PHARMACEUTICAL")
    from collections import defaultdict
    by_first = defaultdict(list)
    for k in idx:
        ft = k.split(" ", 1)[0] if k else ""
        if ft:
            by_first[ft].append(k)

    def prefix_match(S):
        if len(S) < 6:
            return None
        ft = S.split(" ", 1)[0]
        keys = by_first.get(ft, [])
        # index name starts with the (possibly truncated) sponsor, OR sponsor starts with a full index name
        cands = {idx[k][0] for k in keys if k.startswith(S) or (S.startswith(k) and len(k) >= 6)}
        if len(cands) == 1:
            return next(iter(cands))
        if len(cands) > 1:                       # collision -> prefer the unique tradeable one
            trad = {tk for tk in cands if tk in have}
            if len(trad) == 1:
                return next(iter(trad))
        return None

    smap = {}
    unmatched = []
    for sp in sponsors:
        k = norm(sp)
        hit = idx.get(k)
        tk = hit[0] if hit else prefix_match(k)
        if tk:
            smap[sp] = tk
        else:
            unmatched.append(sp)
    # keep only sponsors whose ticker is tradeable (has candles)
    tradeable_map = {sp: tk for sp, tk in smap.items() if tk in have}

    json.dump(tradeable_map, open(OUT, "w"), indent=1)
    # event-level coverage (post-2015, where we have prices)
    ev15 = [e for e in events if e["action_date"] >= "2015-01-01"]
    ev_mapped = [e for e in ev15 if tradeable_map.get(e["sponsor_name"])]
    print(f"\nsponsors matched (any): {len(smap)}/{len(sponsors)} | tradeable (has candles): {len(tradeable_map)}", flush=True)
    print(f"EVENTS >=2015: {len(ev15)} | with tradeable ticker: {len(ev_mapped)} "
          f"({100*len(ev_mapped)/max(1,len(ev15)):.0f}%)", flush=True)
    from collections import Counter
    byk = Counter(e["kind"] for e in ev_mapped)
    print(f"  mapped events by kind: {dict(byk)}", flush=True)
    print(f"WROTE {OUT}", flush=True)
    # sample matches + a few unmatched biopharma-looking sponsors for eyeballing
    print("\nsample matches:", flush=True)
    for sp, tk in list(tradeable_map.items())[:10]:
        print(f"  {sp[:40]:40} -> {tk}", flush=True)
    bio_un = [s for s in unmatched if any(w in s.upper() for w in ("PHARMA", "THERAP", "BIO", "SCIENCE", "GENE"))][:12]
    print(f"\nunmatched biopharma-looking sponsors ({len(bio_un)} shown): {bio_un}", flush=True)


if __name__ == "__main__":
    main()
