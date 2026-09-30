#!/usr/bin/env python
"""Build a self-contained local dashboard of all POTENTIAL Sortino-RSI entries from the cross scan.
Reads sortino_cross_scan.csv, embeds it as JSON, writes a single interactive HTML (no external deps)."""
import json, pandas as pd, os

CSV = "/app/.data/studies/sortino_cross_scan.csv"
OUT = "/app/.data/served/sortino_potential.html"
OUT_WATCH = "/app/.data/served/sortino_watchlist.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = pd.read_csv(CSV)
d = d.where(pd.notna(d), None)
recs = d.to_dict("records")

# ---- exchange mapping: EODHD suffix -> (readable market, TradingView exchange prefix) ----
EXCH = {
    "AS": ("Euronext Amsterdam", "EURONEXT"), "AX": ("ASX · Australia", "ASX"),
    "CO": ("Nasdaq Copenhagen", "OMXCOP"),   "DE": ("XETRA · Germany", "XETR"),
    "HK": ("HKEX · Hong Kong", "HKEX"),       "JO": ("JSE · Johannesburg", "JSE"),
    "MC": ("BME · Madrid", "BME"),            "MX": ("BMV · Mexico", "BMV"),
    "PA": ("Euronext Paris", "EURONEXT"),     "SA": ("B3 · Brazil", "BMFBOVESPA"),
    "SS": ("SSE · Shanghai", "SSE"),          "SW": ("SIX · Switzerland", "SIX"),
    "SZ": ("SZSE · Shenzhen", "SZSE"),        "TW": ("TWSE · Taiwan", "TWSE"),
    "TWO": ("TPEx · Taiwan", "TPEX"),         "WA": ("GPW · Warsaw", "GPW"),
}
try:
    TV_NAMES = json.load(open("/app/.data/served/tv_names.json", encoding="utf-8"))
except Exception:
    TV_NAMES = {}
for r in recs:
    tk = r["ticker"]
    if "." in tk:
        base, sfx = tk.rsplit(".", 1)
        market, xcode = EXCH.get(sfx, (sfx, sfx))
        r["exch"], r["xcode"], r["tv"] = market, xcode, f"{xcode}:{base}"
    else:
        r["exch"], r["xcode"], r["tv"] = "US", "US", tk        # US: bare symbol (TradingView resolves)
    r["company"] = TV_NAMES.get(r["tv"], "")
asof = recs[0]["as_of"] if recs else "n/a"
n_us = int((d.market == "US").sum()); n_fx = int((d.market == "FX").sum())
n_buy = int(d.full_buy.sum()); n_tail = int(d.tail_target.sum())
DATA = json.dumps(recs, separators=(",", ":"))

# ---- TradingView-importable watchlist (comma-separated EXCHANGE:SYMBOL, with ### section headers) ----
tail_r = [r for r in recs if r["tail_target"]]
buy_r = [r for r in recs if r["full_buy"] and not r["tail_target"]]
co_r = [r for r in recs if not r["full_buy"]]
parts = []
if tail_r: parts += ["###TAIL-TARGETS"] + [r["tv"] for r in tail_r]
if buy_r:  parts += ["###FULL BUYS"] + [r["tv"] for r in buy_r]
if co_r:   parts += ["###CROSS-ONLY"] + [r["tv"] for r in co_r]
open(OUT_WATCH, "w", encoding="utf-8").write(",".join(parts))

HTML = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sortino-RSI · Potential Entries</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--bd:#2a3240;--tx:#e6edf3;--mut:#8b949e;
--grn:#3fb950;--red:#f85149;--amb:#d29922;--blue:#58a6ff;--tail:#a371f7;--chip:#21262d;}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--panel:#fff;--panel2:#f0f3f6;--bd:#d0d7de;
--tx:#1f2328;--mut:#656d76;--chip:#eaeef2;}}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--tx);
font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:22px 16px 60px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin:0 0 18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}
.card .v{font-size:26px;font-weight:700}.card .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.card.tail .v{color:var(--tail)}.card.buy .v{color:var(--grn)}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}
input[type=search],select{background:var(--panel);color:var(--tx);border:1px solid var(--bd);
border-radius:8px;padding:8px 10px;font-size:13px;outline:none}input[type=search]{min-width:210px}
.seg{display:inline-flex;border:1px solid var(--bd);border-radius:8px;overflow:hidden}
.seg button{background:var(--panel);color:var(--mut);border:0;padding:8px 12px;cursor:pointer;font-size:13px}
.seg button.on{background:var(--blue);color:#fff}
.rng{display:flex;align-items:center;gap:6px;color:var(--mut);font-size:12px;background:var(--panel);
border:1px solid var(--bd);border-radius:8px;padding:5px 10px}.rng output{color:var(--tx);font-weight:600}
.dl{background:var(--panel);color:var(--blue);border:1px solid var(--bd);border-radius:8px;padding:8px 12px;
font-size:13px;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:4px}
.dl:hover{border-color:var(--blue)}
.pill[title]{cursor:help}
.co-name{max-width:200px;overflow:hidden;text-overflow:ellipsis;color:var(--tx)}
.tblwrap{overflow-x:auto;border:1px solid var(--bd);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:940px}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--bd)}
th{position:sticky;top:0;background:var(--panel2);cursor:pointer;user-select:none;font-weight:600;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.3px}
th:hover{color:var(--tx)}th.a::after{content:" \\2191"}th.d::after{content:" \\2193"}
td.l,th.l{text-align:left}tbody tr:hover{background:var(--panel2)}
tr.tail{background:rgba(163,113,247,.09)}tr.tail:hover{background:rgba(163,113,247,.16)}
.tk{font-weight:700}.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600}
.pill.tail{background:rgba(163,113,247,.18);color:var(--tail)}
.pill.buy{background:rgba(63,185,80,.16);color:var(--grn)}
.pill.co{background:var(--chip);color:var(--mut)}
.pos{color:var(--grn)}.neg{color:var(--red)}.mut{color:var(--mut)}.up{color:var(--grn)}
.note{color:var(--mut);font-size:12px;margin-top:16px;line-height:1.6}
.note b{color:var(--tx)}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 16px;font-size:12px;color:var(--mut)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.count{color:var(--mut);font-size:12px;margin:8px 2px}
</style></head><body><div class="wrap">
<h1>Sortino-RSI · Potential Entries</h1>
<p class="sub">Pine v6 port — RSI(14) of the rolling Sortino ratio crossing up its EMA, still holding. As of <b id="asof"></b> · $1B+ tradeable, $vol&ge;$5M · cross within last ~month.</p>
<div class="cards">
 <div class="card"><div class="v" id="c-tot"></div><div class="l">Recent crosses</div></div>
 <div class="card buy"><div class="v" id="c-buy"></div><div class="l">Full Pine BUYs (DD&gt;0)</div></div>
 <div class="card tail"><div class="v" id="c-tail"></div><div class="l">Tail-targets</div></div>
 <div class="card"><div class="v" id="c-mkt"></div><div class="l">US / Foreign</div></div>
</div>
<div class="legend">
 <span><span class="dot" style="background:var(--tail)"></span>Tail-target (Ulcer&ge;5, 0&lt;DD&le;2.5 — carries the big trades)</span>
 <span><span class="dot" style="background:var(--grn)"></span>Full Pine BUY (DD&gt;0)</span>
 <span><span class="dot" style="background:var(--mut)"></span>Cross-only (trigger fired, DD not yet positive)</span>
</div>
<div class="controls">
 <input type="search" id="q" placeholder="Search ticker / sleeve...">
 <span class="seg" id="mkt"><button data-v="all" class="on">All</button><button data-v="US">US</button><button data-v="FX">Foreign</button></span>
 <span class="seg" id="flag"><button data-v="all" class="on">All</button><button data-v="tail">Tail</button><button data-v="buy">BUYs</button><button data-v="co">Cross-only</button></span>
 <select id="cap"><option value="all">All caps</option><option>mega</option><option>large</option><option>mid</option><option>small</option></select>
 <label class="rng">bars&le; <input type="range" id="bars" min="0" max="21" value="21"><output id="barsv">21</output></label>
 <label class="rng">Ulcer&ge; <input type="range" id="ulc" min="0" max="20" value="0" step="0.5"><output id="ulcv">0</output></label>
 <a class="dl" href="sortino_watchlist.txt" download>&#8595; TradingView list</a>
 <button class="dl" id="copytv">&#8862; Copy symbols</button>
</div>
<div class="count" id="count"></div>
<div class="tblwrap"><table id="t"><thead><tr>
 <th class="l" data-k="ticker">Ticker</th><th class="l" data-k="company">Company</th><th class="l" data-k="xcode">Market</th><th class="l" data-k="sector">Sleeve</th>
 <th class="l" data-k="cap">Cap</th><th data-k="mcap_b">Mcap $B</th><th data-k="bars_since_cross">Bars</th>
 <th data-k="cross_date">Crossed</th><th data-k="srsi">S-RSI</th><th data-k="srsi_ema">EMA</th>
 <th data-k="dd">DD</th><th data-k="ulcer">Ulcer</th><th data-k="ohlc4_rsi">O4-RSI</th>
 <th data-k="dvol_m">$M vol</th><th class="l" data-k="_flag">Flag</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<div class="note">
 <b>How to read it.</b> The trigger is the Sortino-RSI up-cross (S-RSI &gt; EMA). It becomes a full BUY when the DD downside score turns positive; a <b>tail-target</b> also clears the validated filter (Ulcer&ge;5, 0&lt;DD&le;2.5, non-mega) — the config shown to carry the big trades.
 Deeper entries (low S-RSI just crossing up) historically pay more than extended ones (S-RSI already ~90). <b>Ulcer</b> is the tail lever: higher = fatter upside tail. &bull;
 <b>Caveats:</b> "Sleeve" is the ETF bucket a name sits in, not GICS. Foreign tickers carry an exchange suffix. This is a research board, not advice.
</div>
</div>
<script>
const DATA=__DATA__;const ASOF="__ASOF__";
document.getElementById('asof').textContent=ASOF;
document.getElementById('c-tot').textContent=DATA.length;
document.getElementById('c-buy').textContent=DATA.filter(r=>r.full_buy).length;
document.getElementById('c-tail').textContent=DATA.filter(r=>r.tail_target).length;
document.getElementById('c-mkt').textContent=DATA.filter(r=>r.market=='US').length+' / '+DATA.filter(r=>r.market=='FX').length;
const flagOf=r=>r.tail_target?'tail':(r.full_buy?'buy':'co');
function flagReason(r,f){const s=(+r.srsi).toFixed(0),e=(+r.srsi_ema).toFixed(0),dd=(+r.dd).toFixed(2),u=(+r.ulcer).toFixed(1),b=r.bars_since_cross;
 if(f=='co')return `Cross fired ${b} bar(s) ago (Sortino-RSI ${s} > EMA ${e}), but DD=${dd} is not > 0 yet — the downside score must turn positive to confirm a full BUY.`;
 if(f=='buy'){let x=[];if(+r.ulcer<5)x.push(`Ulcer ${u} < 5`);if(+r.dd>2.5)x.push(`DD ${dd} > 2.5`);
  return `Full Pine BUY: Sortino-RSI crossed above its EMA ${b} bar(s) ago and DD=${dd} > 0. Not a tail-target because ${x.join(' and ')||'it just misses'}.`;}
 return `Tail-target: Ulcer ${u} >= 5 and 0 < DD ${dd} <= 2.5, crossed ${b} bar(s) ago — matches the validated filter that has carried the big trades.`;}
let VIEW=[];
let st={q:'',mkt:'all',flag:'all',cap:'all',bars:21,ulc:0,sort:'srsi',dir:1};
const num=v=>v==null||v===''?NaN:+v;
function fnum(v,d=1){return v==null||v===''||isNaN(+v)?'<span class=mut>–</span>':(+v).toFixed(d);}
function render(){
 let rows=DATA.filter(r=>{
  if(st.mkt!='all'&&r.market!=st.mkt)return false;
  if(st.flag!='all'&&flagOf(r)!=st.flag)return false;
  if(st.cap!='all'&&r.cap!=st.cap)return false;
  if(num(r.bars_since_cross)>st.bars)return false;
  if((num(r.ulcer)||0)<st.ulc)return false;
  if(st.q){const s=(r.ticker+' '+r.sector+' '+(r.company||'')).toLowerCase();if(!s.includes(st.q))return false;}
  return true;});
 rows.sort((a,b)=>{let x=a[st.sort],y=b[st.sort];
  if(['ticker','market','sector','cap','cross_date'].includes(st.sort)){x=(x||'')+'';y=(y||'')+'';return x<y?-st.dir:x>y?st.dir:0;}
  x=num(x);y=num(y);if(isNaN(x))x=-1e9;if(isNaN(y))y=-1e9;return(x-y)*st.dir;});
 VIEW=rows;
 const tb=document.getElementById('tb');
 tb.innerHTML=rows.map(r=>{const f=flagOf(r);
  const ddc=num(r.dd)>0?'pos':num(r.dd)<0?'neg':'mut';
  const arr=num(r.srsi)>=num(r.srsi_ema)?'<span class=up>&#9650;</span>':'';
  const lbl=f=='tail'?'TAIL':f=='buy'?'BUY':'cross';
  const pill=`<span class="pill ${f}" title="${flagReason(r,f)}">${lbl}</span>`;
  return `<tr class="${f=='tail'?'tail':''}">
   <td class="l tk">${r.ticker}</td><td class="l co-name" title="${r.company||''}">${r.company||''}</td><td class="l mut" title="${r.exch||''}">${r.xcode||''}</td><td class="l mut">${r.sector||''}</td>
   <td class="l mut">${r.cap||''}</td><td>${fnum(r.mcap_b,1)}</td><td>${r.bars_since_cross}</td>
   <td class="mut">${r.cross_date}</td><td>${fnum(r.srsi,0)} ${arr}</td><td class="mut">${fnum(r.srsi_ema,0)}</td>
   <td class="${ddc}">${fnum(r.dd,2)}</td><td>${fnum(r.ulcer,1)}</td><td>${fnum(r.ohlc4_rsi,0)}</td>
   <td class="mut">${fnum(r.dvol_m,0)}</td><td class="l">${pill}</td></tr>`;}).join('');
 document.getElementById('count').textContent=rows.length+' shown';
 document.querySelectorAll('th').forEach(th=>{th.classList.remove('a','d');
  if(th.dataset.k==st.sort)th.classList.add(st.dir>0?'a':'d');});
}
document.getElementById('copytv').onclick=e=>{const syms=VIEW.map(r=>r.tv).join(',');
 navigator.clipboard.writeText(syms).then(()=>{const b=e.target;const o=b.textContent;b.textContent='\\u2713 '+VIEW.length+' copied';setTimeout(()=>b.textContent=o,1600);});};
document.getElementById('q').oninput=e=>{st.q=e.target.value.toLowerCase().trim();render();};
document.getElementById('cap').onchange=e=>{st.cap=e.target.value;render();};
document.getElementById('bars').oninput=e=>{st.bars=+e.target.value;document.getElementById('barsv').textContent=e.target.value;render();};
document.getElementById('ulc').oninput=e=>{st.ulc=+e.target.value;document.getElementById('ulcv').textContent=e.target.value;render();};
for(const grp of ['mkt','flag'])document.getElementById(grp).onclick=e=>{if(e.target.tagName!='BUTTON')return;
 st[grp]=e.target.dataset.v;[...e.target.parentElement.children].forEach(b=>b.classList.toggle('on',b==e.target));render();};
document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
 if(st.sort==k)st.dir*=-1;else{st.sort=k;st.dir=['ticker','market','sector','cap','cross_date'].includes(k)?1:-1;}render();});
render();
</script></body></html>"""

HTML = HTML.replace("__DATA__", DATA).replace("__ASOF__", asof)
open(OUT, "w", encoding="utf-8").write(HTML)
print(f"wrote {OUT}  ({len(recs)} rows, {len(HTML)//1024}KB)  asof={asof}  US={n_us} FX={n_fx} buy={n_buy} tail={n_tail}")
