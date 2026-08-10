import urllib.request, urllib.parse, json
BASE="https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
def q(sql):
    u=BASE+"?"+urllib.parse.urlencode({"q":sql})
    req=urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8","replace"))
d=q("DESCRIBE option_chain")
print("option_chain columns:")
for r in d["rows"]: print("  ",r.get("Field"),r.get("Type"))
dr=q("SELECT min(date) mn, max(date) mx FROM option_chain")
print("date range:", dr["rows"])
# sample: put vs call OPEN INTEREST for AAPL on the latest available date
ld=q("SELECT max(date) d FROM option_chain WHERE act_symbol='AAPL'")["rows"][0]["d"]
agg=q(f"SELECT call_put, SUM(open_interest) oi, SUM(vol) v FROM option_chain WHERE act_symbol='AAPL' AND date='{ld}' GROUP BY call_put")
print(f"AAPL {ld} put/call OI:", agg["rows"])
