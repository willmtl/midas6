#!/usr/bin/env python
"""Multi-stock TradingView-style page: price + BUY/SELL markers on top, and the Pine indicator panes below
(Sortino-RSI + EMA, DD Downside Score, Ulcer Index, OHLC4-RSI + EMA). Dropdown to switch stocks.
Reuses the scanner's exact indicator + state-machine code. Writes /app/.data/served/sortino_charts.html."""
import os, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
import django; django.setup()
import numpy as np, pandas as pd
import sortino_dd_scanner as S
from seq_fundamental_study import load_candles

S.ULCER_MIN = float(os.environ.get("ULCER_MIN", 5.0))   # Pine ulcer gate for the chart's BUY/SELL markers (user: try >5)
NBARS = int(os.environ.get("NBARS", 460))   # ~22 months displayed
TICKERS = (os.environ.get("TICKERS") or
           "RKLB,NIO,TSLA,ENPH,ARWR,TGTX,AEHR,CVNA,NBIS,MP,ALAB,CENX,NTLA,PLUG,"
           "UPST,RXRX,ETSY,AFRM,SAIA,TER,ONTO,VALE,CDNS,PODD").split(",")
OUT = "/app/.data/served/sortino_charts.html"

cd = load_candles(TICKERS)
out = {}
for tk in TICKERS:
    d = cd.get(tk)
    if d is None or "Close" not in d:
        continue
    close = pd.DataFrame({tk: d["Close"]}).sort_index(); close = close[close > 0]
    idx = close.index
    o = d["Open"].reindex(idx); h = d["High"].reindex(idx); l = d["Low"].reindex(idx)
    ohlc4 = pd.DataFrame({tk: (o + h + l + close[tk]) / 4.0})
    ind = S.compute_indicators(close, ohlc4)
    cnp = close[tk].to_numpy(); valid = close[tk].notna().to_numpy()
    srsi = ind["srsi"][tk].to_numpy(); srsi_ema = ind["srsi_ema"][tk].to_numpy()
    nrsi = ind["nrsi"][tk].to_numpy(); nrsi_ema = ind["nrsi_ema"][tk].to_numpy()
    dd = ind["dd_score"][tk].to_numpy(); ulcer = ind["ulcer"][tk].to_numpy()
    warm = max(S.SORT_WIN + S.SORT_SMOOTH + S.SORT_RSI + S.SORT_EMA, S.DD_DD, 200)
    trips = S.run_state_machine(0, cnp, srsi, srsi_ema, nrsi, nrsi_ema, dd, ulcer, valid, warm)
    a = max(0, len(idx) - NBARS)
    def ser(arr):
        return [None if not np.isfinite(x) else round(float(x), 4) for x in arr[a:]]
    dates = [str(x.date()) for x in idx[a:]]
    buys = [[str(idx[ei].date()), round(float(cnp[ei]), 2)] for (ei, xi, r, b) in trips if ei >= a]
    sells = [[str(idx[xi].date()), round(float(cnp[xi]), 2), r] for (ei, xi, r, b) in trips if xi > 0 and xi >= a]
    out[tk] = dict(dates=dates, close=ser(cnp), srsi=ser(srsi), srsi_ema=ser(srsi_ema),
                   nrsi=ser(nrsi), nrsi_ema=ser(nrsi_ema), dd=ser(dd), ulcer=ser(ulcer),
                   buys=buys, sells=sells)
    print(f"{tk}: {len(dates)} bars, {len(buys)} buys, {len(sells)} sells", flush=True)

DATA = json.dumps(out, separators=(",", ":"))
TKS = json.dumps([t for t in TICKERS if t in out])

html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sortino-DD signals &amp; indicators</title>
<style>
:root{--bg:#0e1016;--card:#161923;--ink:#e8ebf2;--mut:#8b93a7;--line:#242a38;--grid:#1c2230;
 --price:#d7dce8;--buy:#3ecb84;--sell:#e5605e;--hard:#f0a63c;--srsi:#6ea8fe;--srsiema:#a98bfb;
 --dd:#4db6ac;--ulcer:#c678dd;--nrsi:#6ea8fe;--nrsiema:#f0a63c;--zero:#4a5265}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d26;--mut:#5b6473;--line:#e3e7ef;--grid:#eef1f6;--price:#2a2f3a}}
:root[data-theme=light]{--bg:#f6f7f9;--card:#fff;--ink:#1a1d26;--mut:#5b6473;--line:#e3e7ef;--grid:#eef1f6;--price:#2a2f3a}
:root[data-theme=dark]{--bg:#0e1016;--card:#161923;--ink:#e8ebf2;--mut:#8b93a7;--line:#242a38;--grid:#1c2230;--price:#d7dce8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:22px 18px 60px}
h1{font-size:20px;margin:0 0 3px}.sub{color:var(--mut);font-size:13px;margin:0 0 16px}
.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:15px;font-weight:600}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut)}
.legend b{color:var(--ink);font-weight:600}.sw{display:inline-block;width:12px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 10px 2px;margin-bottom:8px}
.plab{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em;padding:4px 4px 0}
canvas{width:100%;display:block}
.tt{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:12px;font-variant-numeric:tabular-nums;opacity:0;box-shadow:0 6px 24px rgba(0,0,0,.3);z-index:9;white-space:nowrap}
</style></head><body><div class="wrap">
<h1>Sortino-DD &mdash; signals &amp; indicators</h1>
<p class="sub">BUY = Sortino-RSI crossed up its EMA within 10 bars + DD&gt;0 + Ulcer&gt;10. SELL = Sortino down-cross confirmed by OHLC4-RSI &amp; DD&lt;0 (orange = hard DD&ge;3). ~22 months shown.</p>
<div class="bar">
  <select id="pick"></select>
  <div class="legend">
    <span><i class="sw" style="background:var(--buy)"></i>BUY</span>
    <span><i class="sw" style="background:var(--sell)"></i>SELL</span>
    <span><i class="sw" style="background:var(--hard)"></i>hard DD&ge;3</span>
  </div>
</div>
<div class="card"><div class="plab">Price &amp; signals</div><canvas id="c_price" height="230"></canvas></div>
<div class="card"><div class="plab">Sortino-RSI(14) &middot; <span style="color:var(--srsi)">RSI</span> vs <span style="color:var(--srsiema)">EMA</span></div><canvas id="c_srsi" height="120"></canvas></div>
<div class="card"><div class="plab">DD Downside Score &middot; <span style="color:var(--zero)">0</span> / <span style="color:var(--hard)">hard 3</span></div><canvas id="c_dd" height="110"></canvas></div>
<div class="card"><div class="plab">Ulcer Index &middot; buy gate 10</div><canvas id="c_ulcer" height="100"></canvas></div>
<div class="card"><div class="plab">OHLC4-RSI(14) &middot; <span style="color:var(--nrsi)">RSI</span> vs <span style="color:var(--nrsiema)">EMA</span> (exit confirm)</div><canvas id="c_nrsi" height="120"></canvas></div>
</div><div class="tt" id="tt"></div>
<script>
const DATA=__DATA__, TKS=__TKS__;
const gc=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pick=document.getElementById('pick');
pick.innerHTML=TKS.map(t=>`<option>${t}</option>`).join('');
let cur=TKS[0], HX=null;
const DPR=Math.max(1,window.devicePixelRatio||1);
function setup(id){const cv=document.getElementById(id),w=cv.clientWidth;cv.width=w*DPR;const h=cv.height;cv.height=h*DPR;const g=cv.getContext('2d');g.setTransform(DPR,0,0,DPR,0,0);return {g,w,h,cv};}
const PADL=52,PADR=10,PADT=8,PADB=16;
function xfor(i,n,w){return PADL+(n>1?i/(n-1):0)*(w-PADL-PADR);}
function yfor(v,lo,hi,h){return PADT+(1-(v-lo)/((hi-lo)||1))*(h-PADT-PADB);}
function years(dates){const o=[];let last='';dates.forEach((d,i)=>{const y=d.slice(0,4)+'-'+d.slice(5,7);const yy=d.slice(0,4);if(yy!==last){o.push([i,yy]);last=yy;}});return o;}
function drawLine(g,arr,dates,lo,hi,w,h,col,wid){g.strokeStyle=col;g.lineWidth=wid||1.4;g.beginPath();let started=false;arr.forEach((v,i)=>{if(v==null){started=false;return;}const x=xfor(i,arr.length,w),y=yfor(v,lo,hi,h);if(!started){g.moveTo(x,y);started=true;}else g.lineTo(x,y);});g.stroke();}
function grid(g,dates,lo,hi,w,h,fmt){g.strokeStyle=gc('--grid');g.fillStyle=gc('--mut');g.font='10px sans-serif';g.lineWidth=1;
 const ticks=4;for(let k=0;k<=ticks;k++){const v=lo+(hi-lo)*k/ticks;const y=yfor(v,lo,hi,h);g.beginPath();g.moveTo(PADL,y);g.lineTo(w-PADR,y);g.stroke();g.fillText(fmt(v),4,y+3);}
 years(dates).forEach(([i,y])=>{const x=xfor(i,dates.length,w);g.strokeStyle=gc('--grid');g.beginPath();g.moveTo(x,PADT);g.lineTo(x,h-PADB);g.stroke();g.fillStyle=gc('--mut');g.fillText(y,x-12,h-4);});}
function rng(a){let lo=Infinity,hi=-Infinity;for(const g of a)for(const v of g){if(v==null)continue;lo=Math.min(lo,v);hi=Math.max(hi,v);}if(!isFinite(lo)){lo=0;hi=1;}const pad=(hi-lo)*.06||1;return [lo-pad,hi+pad];}
let GEO={};
function draw(){const d=DATA[cur];if(!d)return;const dts=d.dates;
 // price
 let o=setup('c_price');let[lo,hi]=rng([d.close]);grid(o.g,dts,lo,hi,o.w,o.h,v=>'$'+v.toFixed(v<10?2:0));
 drawLine(o.g,d.close,dts,lo,hi,o.w,o.h,gc('--price'),1.6);
 GEO={lo,hi,w:o.w,h:o.h,n:dts.length};const di={};dts.forEach((x,i)=>di[x]=i);
 const mk=(arr,col,up,hard)=>{arr.forEach(m=>{const i=di[m[0]];if(i==null)return;const x=xfor(i,dts.length,o.w),y=yfor(m[1],lo,hi,o.h);const c=(hard&&m[2]==='hard_dd')?gc('--hard'):col;o.g.fillStyle=c;o.g.beginPath();const s=5;if(up){o.g.moveTo(x,y+9);o.g.lineTo(x-s,y+9+s);o.g.lineTo(x+s,y+9+s);}else{o.g.moveTo(x,y-9);o.g.lineTo(x-s,y-9-s);o.g.lineTo(x+s,y-9-s);}o.g.closePath();o.g.fill();});};
 mk(d.buys,gc('--buy'),true,false);mk(d.sells,gc('--sell'),false,true);
 // sortino-rsi
 o=setup('c_srsi');[lo,hi]=rng([d.srsi,d.srsi_ema]);grid(o.g,dts,lo,hi,o.w,o.h,v=>v.toFixed(0));
 drawLine(o.g,d.srsi_ema,dts,lo,hi,o.w,o.h,gc('--srsiema'),1.3);drawLine(o.g,d.srsi,dts,lo,hi,o.w,o.h,gc('--srsi'),1.6);
 // dd
 o=setup('c_dd');[lo,hi]=rng([d.dd,[0,3]]);grid(o.g,dts,lo,hi,o.w,o.h,v=>v.toFixed(1));
 o.g.strokeStyle=gc('--zero');o.g.setLineDash([3,3]);o.g.beginPath();let yz=yfor(0,lo,hi,o.h);o.g.moveTo(PADL,yz);o.g.lineTo(o.w-PADR,yz);o.g.stroke();
 o.g.strokeStyle=gc('--hard');let y3=yfor(3,lo,hi,o.h);o.g.beginPath();o.g.moveTo(PADL,y3);o.g.lineTo(o.w-PADR,y3);o.g.stroke();o.g.setLineDash([]);
 drawLine(o.g,d.dd,dts,lo,hi,o.w,o.h,gc('--dd'),1.5);
 // ulcer
 o=setup('c_ulcer');[lo,hi]=rng([d.ulcer,[10,10]]);grid(o.g,dts,lo,hi,o.w,o.h,v=>v.toFixed(0));
 o.g.strokeStyle=gc('--mut');o.g.setLineDash([3,3]);let y10=yfor(10,lo,hi,o.h);o.g.beginPath();o.g.moveTo(PADL,y10);o.g.lineTo(o.w-PADR,y10);o.g.stroke();o.g.setLineDash([]);
 drawLine(o.g,d.ulcer,dts,lo,hi,o.w,o.h,gc('--ulcer'),1.5);
 // ohlc4-rsi
 o=setup('c_nrsi');[lo,hi]=rng([d.nrsi,d.nrsi_ema]);grid(o.g,dts,lo,hi,o.w,o.h,v=>v.toFixed(0));
 o.g.strokeStyle=gc('--zero');o.g.setLineDash([3,3]);let y50=yfor(50,lo,hi,o.h);o.g.beginPath();o.g.moveTo(PADL,y50);o.g.lineTo(o.w-PADR,y50);o.g.stroke();o.g.setLineDash([]);
 drawLine(o.g,d.nrsi_ema,dts,lo,hi,o.w,o.h,gc('--nrsiema'),1.3);drawLine(o.g,d.nrsi,dts,lo,hi,o.w,o.h,gc('--nrsi'),1.6);
}
pick.onchange=()=>{cur=pick.value;draw();};
new ResizeObserver(draw).observe(document.getElementById('c_price'));
new MutationObserver(draw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
// hover readout on price
const tt=document.getElementById('tt');const pc=document.getElementById('c_price');
pc.onmousemove=e=>{const d=DATA[cur];if(!GEO.n)return;const r=pc.getBoundingClientRect();let i=Math.round((e.clientX-r.left-PADL)/((GEO.w-PADL-PADR))*(GEO.n-1));i=Math.max(0,Math.min(GEO.n-1,i));
 tt.innerHTML=`<b>${d.dates[i]}</b> &middot; $${(d.close[i]||0).toFixed(2)}<br>sRSI ${fmt(d.srsi[i])}/${fmt(d.srsi_ema[i])} &middot; DD ${fmt(d.dd[i])} &middot; ulcer ${fmt(d.ulcer[i])} &middot; oRSI ${fmt(d.nrsi[i])}`;
 tt.style.opacity=1;tt.style.left=Math.min(e.clientX+14,innerWidth-260)+'px';tt.style.top=(e.clientY+14)+'px';};
pc.onmouseleave=()=>tt.style.opacity=0;
function fmt(v){return v==null?'–':(+v).toFixed(1);}
draw();
</script></body></html>"""
html = html.replace("__DATA__", DATA).replace("__TKS__", TKS)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(html)
print(f"\nwrote {OUT}  {len(html):,} bytes  ({len(out)} stocks)")
