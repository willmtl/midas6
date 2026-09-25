#!/usr/bin/env python
"""Build a self-contained equity-curve + underwater(drawdown) chart for the Sortino-DD tail basket.
Reads /app/.data/studies/sortino_dd_scanner.json -> writes /app/.data/served/sortino_tail.html."""
import json, os

SRC = "/app/.data/studies/sortino_dd_scanner.json"
OUT = "/app/.data/served/sortino_tail.html"
d = json.load(open(SRC))
eq = d["portfolio_equity"]; port = d["portfolio"]; cfg = d.get("tuned_config", {})
data_json = json.dumps(eq)
stats_json = json.dumps(port)
span = f"{eq['ALL-signals EW'][0][0]} → {eq['ALL-signals EW'][-1][0]}"
cfgtxt = f"ulcer≥{cfg.get('ulcer_min')}, 0&lt;dd≤{cfg.get('dd_ceil')}, ex-mega, entry=signal close, exit=Pine (DD/OHLC4-RSI)"

html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sortino-DD Tail Basket &mdash; Equity &amp; Drawdown</title>
<style>
:root{--bg:#0f1117;--card:#171a23;--ink:#e9ecf3;--mut:#8b93a7;--line:#252b39;--grid:#1e2431;
  --tail:#f0b429;--max5:#f2683c;--spy:#7f8aa3;--qqq:#38c2b4;--red:#e5605e;--pos:#3ecb84}
@media (prefers-color-scheme:light){:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d26;--mut:#5b6473;--line:#e3e7ef;--grid:#eef1f6;--spy:#5b6473}}
:root[data-theme=dark]{--bg:#0f1117;--card:#171a23;--ink:#e9ecf3;--mut:#8b93a7;--line:#252b39;--grid:#1e2431;--spy:#7f8aa3}
:root[data-theme=light]{--bg:#f6f7f9;--card:#fff;--ink:#1a1d26;--mut:#5b6473;--line:#e3e7ef;--grid:#eef1f6;--spy:#5b6473}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1160px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:22px}
.tile{background:var(--card);padding:14px 16px}
.tile .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:3px}
.tile .v small{font-size:12px;color:var(--mut);font-weight:400}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px 8px;margin-bottom:18px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 2px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:6px 0 10px;font-size:12.5px}
.legend span{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.legend .off{opacity:.35}
canvas{width:100%;display:block}
.note{color:var(--mut);font-size:12px;margin-top:10px}
.tt{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;font-variant-numeric:tabular-nums;opacity:0;transition:opacity .08s;box-shadow:0 6px 24px rgba(0,0,0,.3);z-index:9}
.tt b{font-weight:600}
</style></head><body><div class="wrap">
<h1>Sortino-DD Tail Basket &mdash; Equity &amp; Drawdown</h1>
<p class="sub">__SPAN__ &middot; __CFG__ &middot; gross of costs, survivorship-aware</p>
<div class="tiles" id="tiles"></div>
<div class="card"><h2>Growth of $1 (log scale)</h2>
  <div class="legend" id="leg"></div>
  <canvas id="eqc" height="380"></canvas></div>
<div class="card"><h2>Drawdown / underwater &mdash; ALL-signals equal-weight</h2>
  <canvas id="ddc" height="200"></canvas>
  <p class="note">High-Ulcer names are high-beta &rarr; the basket piles into them in market crashes (concurrent positions spike ~46&rarr;587), so the deep water is 2020 &amp; 2022.</p></div>
</div>
<div class="tt" id="tt"></div>
<script>
const DATA=__DATA__, STATS=__STATS__;
const ORDER=["ALL-signals EW","max5 EW","QQQ","SPY"];
const COL={"ALL-signals EW":getcss("--tail"),"max5 EW":getcss("--max5"),"QQQ":getcss("--qqq"),"SPY":getcss("--spy")};
const NAME={"ALL-signals EW":"Tail basket (all, EW)","max5 EW":"Tail basket (max 5)","QQQ":"QQQ","SPY":"SPY"};
function getcss(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
const on={}; ORDER.forEach(k=>on[k]=true);
// stat tiles
function fin(series){const eqv=series[series.length-1][1]; let peak=-1,dd=0; for(const p of series){peak=Math.max(peak,p[1]); dd=Math.min(dd,(p[1]-peak)/peak);} return {tot:(eqv-1)*100, dd:dd*100};}
const tiles=document.getElementById('tiles');
function tile(k,label){const s=DATA[k]; if(!s)return ''; const f=fin(s); const st=STATS[k]||{};
  const cagr=st.cagr!=null?`<small> &middot; CAGR ${st.cagr>0?'+':''}${st.cagr}%</small>`:'';
  const shp=st.sharpe!=null?` <small>&middot; Sh ${st.sharpe}</small>`:'';
  return `<div class="tile"><div class="k">${label}</div><div class="v" style="color:${COL[k]}">${f.tot>0?'+':''}${Math.round(f.tot).toLocaleString()}%${cagr}${shp}</div>`+
  `<div class="k" style="margin-top:6px">max drawdown</div><div class="v" style="font-size:16px;color:var(--red)">${f.dd.toFixed(0)}%</div></div>`;}
tiles.innerHTML=ORDER.map(k=>tile(k,NAME[k])).join('');
// legend
const leg=document.getElementById('leg');
leg.innerHTML=ORDER.map(k=>`<span data-k="${k}"><i style="background:${COL[k]}"></i>${NAME[k]}</span>`).join('');
leg.querySelectorAll('span').forEach(s=>s.onclick=()=>{on[s.dataset.k]=!on[s.dataset.k]; s.classList.toggle('off',!on[s.dataset.k]); draw();});
// dates -> x index using the longest series
const base=DATA["ALL-signals EW"]; const dates=base.map(p=>p[0]);
const idxByDate={}; dates.forEach((d,i)=>idxByDate[d]=i);
function series_xy(k){const s=DATA[k]||[]; return s.map(p=>[idxByDate[p[0]]??null,p[1],p[2]]).filter(p=>p[0]!=null);}
const DPR=Math.max(1,window.devicePixelRatio||1);
const eqc=document.getElementById('eqc'), ddc=document.getElementById('ddc');
let EQ={}, GEO={};
function setup(cv){const w=cv.clientWidth; cv.width=w*DPR; const h=cv.height*DPR; const g=cv.getContext('2d'); g.setTransform(DPR,0,0,DPR,0,0); return {g,w,h:cv.height};}
function niceYears(){const out=[]; let last=''; dates.forEach((d,i)=>{const y=d.slice(0,4); if(y!==last){out.push([i,y]); last=y;}}); return out;}
function draw(){
  // ---- equity (log) ----
  let {g,w,h}=setup(eqc); g.clearRect(0,0,w,h);
  const padL=52,padR=14,padT=10,padB=22, W=w-padL-padR, H=h-padT-padB;
  let lo=Infinity,hi=-Infinity;
  ORDER.forEach(k=>{if(!on[k])return; series_xy(k).forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1]);});});
  if(!isFinite(lo)){return;}
  lo=Math.max(lo,0.2); const l0=Math.log10(lo), l1=Math.log10(hi);
  const X=i=>padL+ (dates.length>1? i/(dates.length-1):0)*W;
  const Y=v=>padT+H-((Math.log10(v)-l0)/(l1-l0))*H;
  const grid=getcss('--grid'),mut=getcss('--mut'),line=getcss('--line');
  // log gridlines at 1,2,5,10,20,50,100...
  g.strokeStyle=grid; g.fillStyle=mut; g.font='11px sans-serif'; g.lineWidth=1;
  const ticks=[]; for(let e=-1;e<=3;e++){[1,2,5].forEach(m=>{const v=m*Math.pow(10,e); if(v>=lo*0.9&&v<=hi*1.1)ticks.push(v);});}
  ticks.forEach(v=>{const y=Y(v); g.beginPath();g.moveTo(padL,y);g.lineTo(w-padR,y);g.stroke(); g.fillText((v>=1?v:v).toString()+'x',6,y+3);});
  niceYears().forEach(([i,y])=>{const x=X(i); g.strokeStyle=grid;g.beginPath();g.moveTo(x,padT);g.lineTo(x,padT+H);g.stroke(); g.fillStyle=mut;g.fillText(y,x-13,h-6);});
  ORDER.forEach(k=>{if(!on[k])return; const s=series_xy(k); g.strokeStyle=COL[k]; g.lineWidth=(k.indexOf('Tail')>=0||k.indexOf('EW')>=0)?2.1:1.5; g.beginPath();
    s.forEach((p,j)=>{const x=X(p[0]),y=Y(p[1]); j?g.lineTo(x,y):g.moveTo(x,y);}); g.stroke();
    const last=s[s.length-1]; g.fillStyle=COL[k]; g.beginPath(); g.arc(X(last[0]),Y(last[1]),3,0,7); g.fill();});
  EQ.eq={padL,padR,padT,padB,W,H,X,Y,w,h};
  // ---- drawdown ----
  let o=setup(ddc); g=o.g; w=o.w; h=o.h; g.clearRect(0,0,w,h);
  const dH=h-padT-padB; const s=series_xy("ALL-signals EW");
  let dlo=0; s.forEach(p=>dlo=Math.min(dlo,p[2]));
  const Yd=v=>padT+((v-0)/(dlo-0))*dH;
  g.strokeStyle=grid;g.fillStyle=mut;g.font='11px sans-serif';
  [0,-20,-40,-60].forEach(v=>{if(v<dlo-6)return;const y=Yd(v);g.beginPath();g.moveTo(padL,y);g.lineTo(w-padR,y);g.stroke();g.fillText(v+'%',6,y+3);});
  niceYears().forEach(([i,y])=>{const x=X(i);g.strokeStyle=grid;g.beginPath();g.moveTo(x,padT);g.lineTo(x,padT+dH);g.stroke();g.fillStyle=mut;g.fillText(y,x-13,h-6);});
  g.beginPath(); s.forEach((p,j)=>{const x=X(p[0]),y=Yd(p[2]); j?g.lineTo(x,y):g.moveTo(x,y);});
  g.lineTo(X(s[s.length-1][0]),Yd(0)); g.lineTo(X(s[0][0]),Yd(0)); g.closePath();
  g.fillStyle=getcss('--red')+'44'; g.fill(); g.strokeStyle=getcss('--red'); g.lineWidth=1.3; g.beginPath();
  s.forEach((p,j)=>{const x=X(p[0]),y=Yd(p[2]); j?g.lineTo(x,y):g.moveTo(x,y);}); g.stroke();
  EQ.dd={Yd,dH};
}
// hover
const tt=document.getElementById('tt');
function hover(ev){const r=eqc.getBoundingClientRect(); const geo=EQ.eq; if(!geo){return;}
  const mx=ev.clientX-r.left; let i=Math.round((mx-geo.padL)/geo.W*(dates.length-1)); i=Math.max(0,Math.min(dates.length-1,i));
  let rows=`<b>${dates[i]}</b>`;
  ORDER.forEach(k=>{if(!on[k])return; const p=(DATA[k]||[]).find(x=>x[0]===dates[i]); if(p)rows+=`<br><span style="color:${COL[k]}">&#9632;</span> ${NAME[k]}: <b>${p[1].toFixed(2)}x</b> <span style="color:var(--mut)">(${p[2].toFixed(0)}%)</span>`;});
  tt.innerHTML=rows; tt.style.opacity=1; tt.style.left=Math.min(ev.clientX+14,innerWidth-190)+'px'; tt.style.top=(ev.clientY+14)+'px';}
eqc.onmousemove=hover; eqc.onmouseleave=()=>tt.style.opacity=0;
new ResizeObserver(draw).observe(eqc);
draw();
const mo=new MutationObserver(draw); mo.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script></body></html>"""

html = (html.replace("__DATA__", data_json).replace("__STATS__", stats_json)
        .replace("__SPAN__", span).replace("__CFG__", cfgtxt))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT}  {len(html):,} bytes")
