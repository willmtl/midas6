"""Head-to-head: local qwen labels (local_*) vs Anthropic Haiku labels (llm_*) on the overlap set.
Throwaway validation script. The wired signal keys on DIRECTION + IMPACT (impact>=2, rating>0), so those
are the metrics that matter; category taxonomies differ (Haiku is signed) so it's reported loosely."""
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from core.models import NewsItem
from collections import Counter

# signed Haiku cat -> unsigned base, to line up with local's unsigned taxonomy
BASE = {"earnings_beat": "earnings", "earnings_miss": "earnings", "guidance_up": "guidance",
        "guidance_down": "guidance", "upgrade": "analyst", "downgrade": "analyst",
        "capital": "capital", "offering": "capital", "buyback": "capital"}
def base(c):
    c = (c or "").lower()
    if c in BASE: return BASE[c]
    if c in ("offering", "buyback"): return "capital"
    return c

qs = NewsItem.objects.filter(local_rating__isnull=False, llm_rating__isnull=False).values(
    "local_dir", "llm_dir", "local_impact", "llm_impact", "local_rating", "llm_rating",
    "local_horizon", "llm_horizon", "cat_llm", "llm_cat")
rows = list(qs)
n = len(rows)
print(f"paired rows: {n}\n")

def pct(x): return f"{100*x/n:.1f}%" if n else "-"

# DIRECTION
dir_exact = sum(1 for r in rows if r["local_dir"] == r["llm_dir"])
# sign agreement among rows where BOTH took a non-zero stance
both_dir = [r for r in rows if r["local_dir"] and r["llm_dir"]]
dir_sign = sum(1 for r in both_dir if (r["local_dir"] > 0) == (r["llm_dir"] > 0))
# how often they DISAGREE on sign (one bull, one bear) — the dangerous case
opp = sum(1 for r in rows if r["local_dir"] and r["llm_dir"] and (r["local_dir"] > 0) != (r["llm_dir"] > 0))
print("== DIRECTION ==")
print(f"  exact (-1/0/+1)   : {dir_exact}/{n} = {pct(dir_exact)}")
print(f"  sign agree (both nonzero, n={len(both_dir)}): {dir_sign}/{len(both_dir)} = "
      f"{100*dir_sign/len(both_dir):.1f}%" if both_dir else "  n/a")
print(f"  OPPOSITE sign (bull vs bear): {opp}/{n} = {pct(opp)}  <- dangerous cases")
cm = Counter((r["llm_dir"], r["local_dir"]) for r in rows)
print("  confusion (haiku_dir -> local_dir):")
for hd in (-1, 0, 1):
    print(f"    haiku {hd:+d}: " + "  ".join(f"local{ld:+d}={cm.get((hd,ld),0)}" for ld in (-1,0,1)))

# IMPACT
imp_exact = sum(1 for r in rows if r["local_impact"] == r["llm_impact"])
imp_w1 = sum(1 for r in rows if abs((r["local_impact"] or 0) - (r["llm_impact"] or 0)) <= 1)
mad = sum(abs((r["local_impact"] or 0) - (r["llm_impact"] or 0)) for r in rows) / n
# material-set agreement: does each side call it impact>=2 the same way?
mat_agree = sum(1 for r in rows if (r["local_impact"] >= 2) == (r["llm_impact"] >= 2))
local_mat = sum(1 for r in rows if r["local_impact"] >= 2)
haiku_mat = sum(1 for r in rows if r["llm_impact"] >= 2)
print("\n== IMPACT ==")
print(f"  exact (0-3)       : {imp_exact}/{n} = {pct(imp_exact)}")
print(f"  within +/-1       : {imp_w1}/{n} = {pct(imp_w1)}")
print(f"  mean abs diff     : {mad:.2f}")
print(f"  material(>=2) agree: {mat_agree}/{n} = {pct(mat_agree)}   (local flags {local_mat}, haiku {haiku_mat})")

# RATING (the wired quantity)
r_exact = sum(1 for r in rows if r["local_rating"] == r["llm_rating"])
r_sign = sum(1 for r in rows if (r["local_rating"] > 0) == (r["llm_rating"] > 0) and (r["local_rating"] < 0) == (r["llm_rating"] < 0))
r_w1 = sum(1 for r in rows if abs(r["local_rating"] - r["llm_rating"]) <= 1)
# the SPECIFIC wired predicate: bullish high-impact (impact>=2 & rating>0)
lo_bull = set(i for i, r in enumerate(rows) if r["local_impact"] >= 2 and r["local_rating"] > 0)
ha_bull = set(i for i, r in enumerate(rows) if r["llm_impact"] >= 2 and r["llm_rating"] > 0)
inter = len(lo_bull & ha_bull); uni = len(lo_bull | ha_bull)
print("\n== RATING (dir*impact, the wired quantity) ==")
print(f"  exact (-3..+3)    : {r_exact}/{n} = {pct(r_exact)}")
print(f"  sign class agree  : {r_sign}/{n} = {pct(r_sign)}")
print(f"  within +/-1       : {r_w1}/{n} = {pct(r_w1)}")
print(f"  WIRED predicate (impact>=2 & rating>0 = bullish-hi-impact):")
print(f"    local flags {len(lo_bull)}, haiku flags {len(ha_bull)}, agree-on-both {inter}, "
      f"Jaccard {100*inter/uni:.0f}%" if uni else "    none")

# HORIZON
hz = [r for r in rows if r["local_horizon"] and r["llm_horizon"]]
hz_exact = sum(1 for r in hz if r["local_horizon"] == r["llm_horizon"])
print("\n== HORIZON ==")
print(f"  exact (n={len(hz)}) : {hz_exact}/{len(hz)} = {100*hz_exact/len(hz):.1f}%" if hz else "  n/a")

# CATEGORY (loose, base-mapped)
cat_rows = [r for r in rows if r["cat_llm"] and r["llm_cat"]]
cat_exact = sum(1 for r in cat_rows if base(r["cat_llm"]) == base(r["llm_cat"]))
print("\n== CATEGORY (base-mapped, taxonomies differ) ==")
print(f"  base agree (n={len(cat_rows)}): {cat_exact}/{len(cat_rows)} = "
      f"{100*cat_exact/len(cat_rows):.1f}%" if cat_rows else "  n/a")
