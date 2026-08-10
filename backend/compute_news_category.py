"""
News EVENT-CATEGORY classifier — title-only heuristic, no LLM, no candle data.

Sets NewsItem.cat_auto to a human-readable news TYPE that exists for EVERY item (not just the
~13-month LLM-classified slice), so the dashboard can show a Category column and filter across the
whole corpus (user: "we need a category column and add a few categories like earnings report,
partnership etc").

Ordered detectors — FIRST match wins, so specific/high-signal types are checked before generic ones
(e.g. M&A and partnership before 'contract', clinical before 'product'). Anything unmatched -> other.
The order in CATEGORIES is the priority order.

Writes NewsItem.cat_auto (str). Idempotent; re-runnable. Title-only -> whole corpus in seconds.
Run: docker compose exec -T backend python -u compute_news_category.py [--limit N] [--only-null]
"""
import django, os, sys, re
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings"); django.setup()
from core.models import NewsItem

# (label, regex) in PRIORITY order — first match wins.
CATEGORIES = [
    ("ma", re.compile(
        r"\bto acquire\b|\bacquisition\b|\bacquires?\b|\bmerger\b|\bmerges?\b|\bbuyout\b|\btakeover\b"
        r"|\bto be acquired\b|\bagree(s|d)? to (acquire|buy|merge)\b|\btender offer\b|\bgoes private\b"
        r"|\bto combine with\b|\bto merge with\b", re.I)),
    ("partnership", re.compile(
        r"\bpartnership\b|\bpartners with\b|\bpartner(s|ed)? with\b|\bcollaborat(e|es|ion|ing)\b"
        r"|\bteams? up\b|\bjoint venture\b|\bstrategic alliance\b|\balliance with\b|\bto collaborate\b"
        r"|\bsigns? (a )?(deal|agreement|mou) with\b|\bmemorandum of understanding\b", re.I)),
    ("clinical", re.compile(
        r"\bfda\b|\bphase\s*(1|2|3|i{1,3})\b|\bclinical\b|\btrial\b|\btopline\b|\bnda\b|\bbla\b"
        r"|\b510\(k\)\b|\bide\b|\bapproval\b|\bapproves?\b|\bclearance\b|\bbreakthrough therapy\b"
        r"|\bprimary endpoint\b|\bpivotal\b|\borphan drug\b|\bemergency use\b", re.I)),
    ("analyst", re.compile(
        r"\bupgrade(s|d)?\b|\bdowngrade(s|d)?\b|\bprice target\b|\binitiate(s|d)? coverage\b"
        r"|\breiterate(s|d)?\b|\bmaintains?\b.*\b(buy|sell|hold|neutral)\b|\boverweight\b|\bunderweight\b"
        r"|\boutperform\b|\bunderperform\b|\bhikes? (price )?target\b|\bcuts? (price )?target\b"
        r"|\braises? (price )?target\b|\banalysts?\b.*\b(rating|target)\b|\bbuy rating\b|\bsell rating\b", re.I)),
    ("offering", re.compile(
        r"\b(public|secondary|registered direct|private) offering\b|\bprices? .*offering\b"
        r"|\bshelf (offering|registration)\b|\bconvertible (notes|senior notes|debt)\b|\bprivate placement\b"
        r"|\bat[- ]the[- ]market\b|\batm (offering|program)\b|\bdilut(e|ion|ive)\b|\bunderwritten\b"
        r"|\bcommon stock (offering|units)\b|\bproposed offering\b", re.I)),
    ("buyback", re.compile(
        r"\bbuyback\b|\brepurchase\b|\bshare repurchase\b|\brepurchase program\b|\bto repurchase\b", re.I)),
    ("dividend", re.compile(
        r"\bdividend\b|\bdeclares? .*dividend\b|\bdistribution\b|\bex-dividend\b|\bspecial dividend\b"
        r"|\bhikes? .*dividend\b|\braises? .*dividend\b", re.I)),
    ("guidance", re.compile(
        r"\bguidance\b|\boutlook\b|\bforecast(s|ed)?\b|\bsees? (fy|q[1-4]|full[- ]year|revenue|eps)\b"
        r"|\braises? .*(guidance|outlook|forecast)\b|\bcuts? .*(guidance|outlook|forecast)\b"
        r"|\blowers? .*(guidance|outlook)\b|\breaffirms? .*(guidance|outlook)\b", re.I)),
    ("earnings", re.compile(
        r"\bearnings\b|\bq[1-4]\s*(20\d\d|fy|results|earnings)\b|\breports? .*(results|earnings)\b"
        r"|\bquarterly results\b|\beps\b|\bbeats?\b.*\b(estimates?|expectations?|street)\b"
        r"|\bmisses?\b.*\b(estimates?|expectations?|street)\b|\brevenue of\b|\btops? (estimates?|views?)\b"
        r"|\bfirst[- ]quarter\b|\bsecond[- ]quarter\b|\bthird[- ]quarter\b|\bfourth[- ]quarter\b"
        r"|\bfull[- ]year results\b|\bpreliminary results\b", re.I)),
    ("contract", re.compile(
        r"\bcontract\b|\bawarded\b|\bwins?\b.*\b(deal|order|contract|bid)\b|\border worth\b"
        r"|\bselected by\b|\bto supply\b|\bsupply agreement\b|\bpurchase order\b|\bbags? .*order\b"
        r"|\bsecures? .*(contract|order|deal)\b|\bbook(s|ed)? .*order\b|\btask order\b", re.I)),
    ("product", re.compile(
        r"\blaunch(es|ed|ing)?\b|\bunveil(s|ed)?\b|\bintroduce(s|d)?\b|\brolls? out\b|\brelease(s|d)?\b"
        r"|\bdebut(s|ed)?\b|\bannounce(s|d)? .*(product|platform|service|feature|version|model)\b"
        r"|\bnew (product|platform|service|chip|model|app|feature|version)\b|\bavailab(le|ility)\b"
        r"|\bnow available\b|\bgenerally available\b", re.I)),
    ("legal", re.compile(
        r"\blawsuit\b|\bsues?\b|\bsued\b|\bsettlement\b|\bsettles?\b|\binvestigation\b|\bprobe\b"
        r"|\bsec charges?\b|\bfraud\b|\bclass action\b|\bsubpoena\b|\bantitrust\b|\bpatent (suit|dispute)\b"
        r"|\bfiles? suit\b|\bcourt\b|\bjudge\b|\bverdict\b|\bfined?\b|\bpenalt(y|ies)\b|\brecall\b", re.I)),
    ("insider", re.compile(
        r"\binsider (buy|sell|purchase|selling|buying|transaction)\b|\bform 4\b|\b13d\b|\b13g\b"
        r"|\b10b5-1\b|\bceo (buys?|sells?|purchases?)\b|\bdirector (buys?|sells?)\b|\bstake in\b"
        r"|\bacquires? .*stake\b|\bincreases? .*stake\b|\binstitutional (buying|selling)\b", re.I)),
    ("mgmt", re.compile(
        r"\bappoints?\b|\bnames?\b.*\b(ceo|cfo|coo|cto|president|chair|director)\b|\bresigns?\b"
        r"|\bsteps down\b|\bnew (ceo|cfo|coo|cto|president|chair)\b|\bhires?\b|\bboard of directors\b"
        r"|\bexecutive (change|appointment|hire)\b|\bmanagement change\b|\bpromoted?\b.*\b(ceo|cfo)\b"
        r"|\bdeparture\b|\bretires?\b", re.I)),
    ("macro", re.compile(
        r"\bfederal reserve\b|\bthe fed\b|\brate (cut|hike|decision)\b|\binterest rates?\b|\btariffs?\b"
        r"|\binflation\b|\bjobs report\b|\bgdp\b|\bcpi\b|\bppi\b|\bfomc\b|\bunemployment\b", re.I)),
]


def classify(title):
    t = title or ""
    for label, rx in CATEGORIES:
        if rx.search(t):
            return label
    return "other"


def main(limit=None, only_null=False):
    qs = NewsItem.objects.all()
    if only_null:
        qs = qs.filter(cat_auto="")
    rows = list(qs.values_list("id", "title"))
    if limit:
        rows = rows[:limit]
    print(f"scanning {len(rows)} items (only_null={only_null})", flush=True)

    from collections import Counter
    counts = Counter()
    batch, updated = [], 0
    for _id, title in rows:
        c = classify(title)
        counts[c] += 1
        batch.append(NewsItem(id=_id, cat_auto=c))
        updated += 1
        if len(batch) >= 5000:
            NewsItem.objects.bulk_update(batch, ["cat_auto"], batch_size=2000)
            batch = []
            print(f"  {updated}/{len(rows)} scanned", flush=True)
    if batch:
        NewsItem.objects.bulk_update(batch, ["cat_auto"], batch_size=2000)

    print(f"DONE scanned={updated}", flush=True)
    for cat, n in counts.most_common():
        print(f"  {cat:14s} {n:8d} ({100*n/updated:.1f}%)", flush=True)
    return {"updated": updated, "counts": dict(counts)}


if __name__ == "__main__":
    lim = None; onull = False
    for i, a in enumerate(sys.argv):
        if a == "--limit" and i + 1 < len(sys.argv):
            lim = int(sys.argv[i + 1])
        if a == "--only-null":
            onull = True
    main(limit=lim, only_null=onull)
