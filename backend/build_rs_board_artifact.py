#!/usr/bin/env python3
"""Generate the self-contained Synthetic Relative-Strength board artifact (HTML content, no doctype/head/body) from
synthetic_rs_board.json. Diverging RSI heat (oversold cool -> 50 gray -> overbought warm), QQQ/SPY regime hero,
oversold/overbought ranked columns with sparklines, full 144-sector heatmap. Writes to the given out path."""
import json, sys

BOARD = "C:/workspace/rotation/.data/studies/synthetic_rs_board.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/User-X/AppData/Local/Temp/claude/C--workspace-rotation/3f909c31-a27e-47a9-9f04-c102ff8e78b5/scratchpad/rs_board.html"

d = json.load(open(BOARD))


def compact(s):
    ser = s["series"]["ratio"]
    spark = ser[::4]                       # ~65 pts from 260 daily
    return {"n": s["name"], "e": s["etf"], "rd": s["daily"]["rsi"], "rw": (s["weekly"] or {}).get("rsi"),
            "p": s["daily"]["pct52"], "v": s["daily"]["vs200ma"], "t": s["daily"]["trend"],
            "c20": s["daily"]["cross20_recent"], "c30": s["daily"]["cross30_recent"], "s": spark}


data = {"asof": d["asof"], "qqq": {"rd": d["qqq_spy"]["daily"]["rsi"], "rw": d["qqq_spy"]["weekly"]["rsi"],
                                   "p": d["qqq_spy"]["daily"]["pct52"], "v": d["qqq_spy"]["daily"]["vs200ma"],
                                   "t": d["qqq_spy"]["daily"]["trend"], "s": d["qqq_spy"]["series"]["ratio"][::4]},
        "sectors": [compact(s) for s in d["sectors"]]}
J = json.dumps(data, separators=(",", ":"))

HTML = """<style>
:root{
  --bg:#f7f8fa; --surface:#ffffff; --surface-2:#f0f2f6; --ink:#161a22; --ink-2:#59606e; --ink-3:#8b93a3;
  --line:#e3e7ee; --accent:#2f6bd6; --mono:ui-monospace,"SF Mono",Menlo,"Cascadia Mono",Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --pos:#1c8a5b; --neg:#c0392b;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0c0f16; --surface:#141924; --surface-2:#1b2230; --ink:#e7ecf5; --ink-2:#9aa4b6; --ink-3:#6b7488;
  --line:#232c3c; --accent:#5b8cf0; --pos:#3ec98a; --neg:#f2705f;
}}
:root[data-theme="light"]{--bg:#f7f8fa;--surface:#fff;--surface-2:#f0f2f6;--ink:#161a22;--ink-2:#59606e;--ink-3:#8b93a3;--line:#e3e7ee;--accent:#2f6bd6;--pos:#1c8a5b;--neg:#c0392b;}
:root[data-theme="dark"]{--bg:#0c0f16;--surface:#141924;--surface-2:#1b2230;--ink:#e7ecf5;--ink-2:#9aa4b6;--ink-3:#6b7488;--line:#232c3c;--accent:#5b8cf0;--pos:#3ec98a;--neg:#f2705f;}
*{box-sizing:border-box}
.wrap{background:var(--bg);color:var(--ink);font-family:var(--sans);padding:32px 26px 60px;min-height:100vh;line-height:1.5}
.inner{max-width:1180px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3)}
h1{font-size:clamp(26px,4vw,40px);font-weight:680;letter-spacing:-.02em;margin:.28em 0 .1em;text-wrap:balance}
.lede{color:var(--ink-2);max-width:64ch;font-size:15px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.grid-hero{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;margin:26px 0}
@media(max-width:760px){.grid-hero{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:18px 20px}
.regime .big{font-family:var(--mono);font-size:clamp(40px,7vw,64px);font-weight:600;letter-spacing:-.03em;line-height:1}
.regime .row{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.pill{font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);color:var(--ink-2)}
.kv{display:flex;gap:22px;margin-top:14px;flex-wrap:wrap}
.kv div{display:flex;flex-direction:column;gap:2px}
.kv .k{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3)}
.kv .val{font-family:var(--mono);font-size:16px;font-variant-numeric:tabular-nums}
.legend{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.ramp{display:flex;height:12px;border-radius:4px;overflow:hidden;flex:1;min-width:120px}
.ramp i{flex:1}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:8px 0 26px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
.col h2{font-size:13px;font-family:var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:0 0 10px;font-weight:600}
.rowitem{display:grid;grid-template-columns:1fr auto 64px 44px;align-items:center;gap:12px;padding:8px 10px;border-radius:9px}
.rowitem+.rowitem{margin-top:2px}
.rowitem:hover{background:var(--surface-2)}
.nm{font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tk{font-family:var(--mono);font-size:11px;color:var(--ink-3)}
.chip{font-family:var(--mono);font-size:12px;font-weight:600;text-align:center;padding:3px 0;border-radius:6px}
.spark{width:64px;height:22px;display:block}
h2.section{font-size:13px;font-family:var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin:6px 0 12px;font-weight:600}
.heat{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:6px}
.cell{border-radius:8px;padding:8px 9px;min-height:52px;display:flex;flex-direction:column;justify-content:space-between;cursor:default;position:relative}
.cell .ce{font-family:var(--mono);font-size:11px;font-weight:600;opacity:.92}
.cell .cr{font-family:var(--mono);font-size:16px;font-weight:600;font-variant-numeric:tabular-nums}
.cell .cn{font-size:9.5px;opacity:.8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dot{position:absolute;top:6px;right:7px;width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.9}
.foot{color:var(--ink-3);font-size:12.5px;margin-top:26px;max-width:78ch;border-top:1px solid var(--line);padding-top:16px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:8px 10px;font-family:var(--mono);font-size:11.5px;color:var(--ink);box-shadow:0 6px 24px rgba(0,0,0,.18);opacity:0;transition:opacity .1s;z-index:9;max-width:230px}
</style>

<div class="wrap"><div class="inner">
  <div class="eyebrow">Relative strength &middot; synthetic ratios &middot; as of <span id="asof"></span></div>
  <h1>Sector Synthetics vs SPY</h1>
  <p class="lede">Every flagship sleeve as a synthetic instrument &mdash; its price divided by SPY &mdash; read through
  RSI(14) on the ratio. Cool = lagging the market (oversold); warm = leading (overbought). QQQ/SPY sets the growth-vs-market regime.</p>

  <div class="grid-hero">
    <div class="card regime">
      <div class="eyebrow">QQQ / SPY &middot; growth-vs-market regime</div>
      <div class="row" style="margin-top:8px"><span class="big" id="q-rsi"></span>
        <span class="pill" id="q-state"></span><span class="pill" id="q-trend"></span></div>
      <svg class="spark" id="q-spark" style="width:100%;height:46px;margin-top:12px"></svg>
      <div class="kv">
        <div><span class="k">RSI weekly</span><span class="val" id="q-rw"></span></div>
        <div><span class="k">52-wk pctile</span><span class="val" id="q-p"></span></div>
        <div><span class="k">vs 200-day</span><span class="val" id="q-v"></span></div>
      </div>
    </div>
    <div class="card">
      <div class="eyebrow">RSI-on-ratio scale</div>
      <div class="legend" style="margin-top:14px"><span>0</span><div class="ramp" id="ramp"></div><span>100</span></div>
      <div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;color:var(--ink-3);margin-top:6px">
        <span>oversold vs SPY</span><span>neutral 50</span><span>overbought</span></div>
      <p style="color:var(--ink-2);font-size:12.5px;margin-top:16px">A monitoring board, not a buy list. Standalone
      RS-oversold carries <b>no</b> robust forward edge (survivorship-free tested). The real signal is the
      <b>value-gated double-oversold confluence</b> &mdash; cheap P/E or P/B <i>and</i> oversold vs sector &amp; market.</p>
    </div>
  </div>

  <div class="cols">
    <div class="col"><h2>&#9660; Most oversold vs SPY</h2><div id="over-sold"></div></div>
    <div class="col"><h2>&#9650; Most overbought vs SPY</h2><div id="over-bought"></div></div>
  </div>

  <h2 class="section">All 144 sleeves &middot; ranked oversold &rarr; overbought</h2>
  <div class="heat" id="heat"></div>

  <p class="foot">Synthetic ratio = sleeve ETF close &divide; SPY close. RSI(14) computed on the ratio line (Wilder).
  Data frozen at the last candle import; &ldquo;oversold&rdquo; means lagging SPY recently, not a predicted bounce.
  &#9679; marks a cross up through RSI&nbsp;30 on the ratio within the last 10 sessions.</p>
</div></div>

<div class="tip" id="tip"></div>

<script>
const D=__DATA__;
document.getElementById('asof').textContent=D.asof;
// diverging RdBu (reversed): low RSI = cool blue (oversold), 50 = gray, high = warm red (overbought). CVD-safe family.
const STOPS=[[0,'#2166ac'],[20,'#4393c3'],[30,'#92c5de'],[40,'#c9dced'],[50,'#b9bec8'],[60,'#f6c9b4'],[70,'#e8896b'],[80,'#c94741'],[100,'#8f1a22']];
function hx(h){h=h.replace('#','');return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];}
function heat(r){r=Math.max(0,Math.min(100,r));let a=STOPS[0],b=STOPS[STOPS.length-1];
  for(let i=0;i<STOPS.length-1;i++){if(r>=STOPS[i][0]&&r<=STOPS[i+1][0]){a=STOPS[i];b=STOPS[i+1];break;}}
  const t=(r-a[0])/((b[0]-a[0])||1),ca=hx(a[1]),cb=hx(b[1]);
  return `rgb(${Math.round(ca[0]+(cb[0]-ca[0])*t)},${Math.round(ca[1]+(cb[1]-ca[1])*t)},${Math.round(ca[2]+(cb[2]-ca[2])*t)})`;}
function ink(rgb){const m=rgb.match(/\\d+/g),L=(0.299*m[0]+0.587*m[1]+0.114*m[2]);return L>150?'#10141c':'#f4f7fc';}
// ramp legend
document.getElementById('ramp').innerHTML=Array.from({length:40},(_,i)=>`<i style="background:${heat(i/39*100)}"></i>`).join('');
function spark(el,arr,color){const w=el.clientWidth||64,h=el.clientHeight||22,mn=Math.min(...arr),mx=Math.max(...arr),rg=(mx-mn)||1;
  const pts=arr.map((v,i)=>`${(i/(arr.length-1)*(w-2)+1).toFixed(1)},${(h-1-(v-mn)/rg*(h-2)).toFixed(1)}`).join(' ');
  const up=arr[arr.length-1]>=arr[0];const c=color||(up?'var(--pos)':'var(--neg)');
  el.innerHTML=`<polyline fill="none" stroke="${c}" stroke-width="1.6" stroke-linjoin="round" stroke-linecap="round" points="${pts}"/>`+
    `<circle cx="${(w-2).toFixed(1)}" cy="${(h-1-(arr[arr.length-1]-mn)/rg*(h-2)).toFixed(1)}" r="2" fill="${c}"/>`;}
// hero
const q=D.qqq;document.getElementById('q-rsi').textContent=q.rd.toFixed(1);
document.getElementById('q-state').textContent=q.rd<30?'oversold':q.rd>70?'overbought':'neutral';
document.getElementById('q-trend').textContent='trend '+q.t;
document.getElementById('q-rw').textContent=(q.rw||0).toFixed(1);
document.getElementById('q-p').textContent=(q.p==null?'—':q.p+'%ile');
document.getElementById('q-v').textContent=(q.v>0?'+':'')+q.v+'%';
requestAnimationFrame(()=>spark(document.getElementById('q-spark'),q.s,'var(--accent)'));
// ranked columns
function rows(host,list){host.innerHTML='';list.forEach(s=>{const bg=heat(s.rd),fg=ink(bg);
  const el=document.createElement('div');el.className='rowitem';
  el.innerHTML=`<div class="nm">${s.n} <span class="tk">${s.e}</span></div>
    <svg class="spark sp"></svg>
    <div class="chip" style="background:${bg};color:${fg}">${s.rd.toFixed(0)}${s.c30?'&#9679;':''}</div>
    <div class="tk" style="text-align:right">${s.p==null?'—':s.p+'%'}</div>`;
  host.appendChild(el);spark(el.querySelector('.sp'),s.s);});}
const sorted=[...D.sectors].sort((a,b)=>a.rd-b.rd);
rows(document.getElementById('over-sold'),sorted.slice(0,12));
rows(document.getElementById('over-bought'),sorted.slice(-12).reverse());
// heatmap
const heatEl=document.getElementById('heat');
sorted.forEach(s=>{const bg=heat(s.rd),fg=ink(bg);const c=document.createElement('div');c.className='cell';
  c.style.background=bg;c.style.color=fg;
  c.innerHTML=`<div class="ce">${s.e}${s.c30?'<span class="dot"></span>':''}</div><div class="cr">${s.rd.toFixed(0)}</div><div class="cn">${s.n}</div>`;
  c.addEventListener('mousemove',e=>{const t=document.getElementById('tip');
    t.innerHTML=`<b>${s.n}</b> (${s.e})<br>RSI d ${s.rd.toFixed(1)} / w ${(s.rw||0).toFixed(1)}<br>52-wk ${s.p==null?'—':s.p+'%ile'} &middot; vs200 ${(s.v>0?'+':'')+s.v}%`;
    t.style.opacity=1;t.style.left=Math.min(e.clientX+14,innerWidth-240)+'px';t.style.top=(e.clientY+14)+'px';});
  c.addEventListener('mouseleave',()=>document.getElementById('tip').style.opacity=0);
  heatEl.appendChild(c);});
</script>"""

open(OUT, "w", encoding="utf-8").write(HTML.replace("__DATA__", J))
print("wrote", OUT, len(HTML) + len(J), "bytes")
