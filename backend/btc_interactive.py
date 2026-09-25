#!/usr/bin/env python
"""Interactive Plotly charts with BTC / SPY tabs. Reads viz_<ticker>.json (from btc_live_viz.py) and writes a
self-contained tabbed HTML to the served dir: crosshair across all panels, unified rollover, full entry/exit reasons."""
import os, json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TABS = (os.environ.get("TABS") or "BTC-USD,SPY").split(",")


def build_fig(d):
    TK = d.get("ticker", "?"); P = d["params"]; dates = d["dates"]
    close = d["close"]; rsi14 = d["rsi14"]; rsi20 = d["rsi20"]; sma14 = d["sma14"]
    sortino = d["sortino"]; sortino_s = d.get("sortino_s", sortino); spy_ss = d.get("spy_sortino_s", [None] * len(dates))
    strat_eq = d["strat_eq"]; bh_eq = d["bh_eq"]
    buy = d["buy"]; sell = d["sell"]; held = d["held"]; cross = d.get("cross", [False] * len(dates)); dcross = d.get("dcross", [False] * len(dates))
    vt = {t["entry"]: t for t in d.get("vtrades", [])}; vx = {t["exit"]: t for t in d.get("vtrades", [])}

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.32, 0.16, 0.14, 0.14, 0.24],
                        subplot_titles=(f"{TK} price ($)", "Growth of $1 — strategy vs buy&hold",
                                        f"RSI-{P['rsi20']} of CLOSE (green/red color)", f"ADRIDEM Sortino({P['win']}) — {TK} vs SPY(ref)",
                                        f"RSI-{P['rsi14']} of Sortino + SMA-{P['sma']}"))
    regions = []; s = None
    for i, h in enumerate(held):
        if h and s is None: s = i
        elif not h and s is not None: regions.append((s, i - 1)); s = None
    if s is not None: regions.append((s, len(held) - 1))
    for a, b in regions:
        for r in range(1, 6):
            fig.add_vrect(x0=dates[a], x1=dates[b], fillcolor="#37c2c2", opacity=0.10, line_width=0, row=r, col=1)
    fig.add_trace(go.Scatter(x=dates, y=close, name="Close", line=dict(color="#4c8dff", width=1.6),
                             hovertemplate="$%{y:,.0f}<extra>Close</extra>"), row=1, col=1)
    bi = [i for i in range(len(buy)) if buy[i]]; si = [i for i in range(len(sell)) if sell[i]]
    ci = [i for i in range(len(cross)) if cross[i]]; di = [i for i in range(len(dcross)) if dcross[i]]

    def buy_txt(i):
        t = vt.get(dates[i]); r = t.get("entry_reason", "") if t else ""
        return f"{r}<br>entry ${t['entry_px']:,.0f}" if t else r

    def exit_txt(i):
        t = vx.get(dates[i]); r = t.get("exit_reason", "") if t else ""
        return (f"{r}<br>exit ${t['exit_px']:,.0f} · held {t['hold_days']}d · <b>{t['ret_pct']:+.1f}%</b>") if t else r
    fig.add_trace(go.Scatter(x=[dates[i] for i in bi], y=[close[i] * 0.95 for i in bi], mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", color="#22c55e", size=13, line=dict(color="white", width=1)),
                             text=[buy_txt(i) for i in bi], hovertemplate="%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in si], y=[close[i] * 1.05 for i in si], mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", color="#ef4444", size=13, line=dict(color="white", width=1)),
                             text=[exit_txt(i) for i in si], hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=strat_eq, name="Strategy", line=dict(color="#a855f7", width=2),
                             hovertemplate="×%{y:.2f}<extra>Strategy</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bh_eq, name="Buy&Hold", line=dict(color="#94a3b8", width=1.4, dash="dash"),
                             hovertemplate="×%{y:.2f}<extra>Buy&Hold</extra>"), row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=rsi20, name=f"RSI{P['rsi20']} close", line=dict(color="#14b8a6", width=1.5),
                             hovertemplate="%{y:.1f}<extra>RSI" + str(P['rsi20']) + " close</extra>"), row=3, col=1)
    fig.add_hline(y=P["green"], line=dict(color="#22c55e", width=1, dash="dash"), row=3, col=1)
    fig.add_hline(y=P["red"], line=dict(color="#ef4444", width=1, dash="dash"), row=3, col=1)

    fig.add_trace(go.Scatter(x=dates, y=sortino, name=f"{TK} Sortino raw", line=dict(color="#d6b98c", width=1), opacity=0.5,
                             hovertemplate="%{y:.2f}<extra>Sortino raw</extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sortino_s, name=f"{TK} Sortino", line=dict(color="#f59e0b", width=1.8),
                             hovertemplate="%{y:.2f}<extra>" + TK + " Sortino</extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=dates, y=spy_ss, name="SPY Sortino (ref)", line=dict(color="#ef4444", width=1.6), connectgaps=True,
                             hovertemplate="%{y:.2f}<extra>SPY Sortino (ref)</extra>"), row=4, col=1)
    fig.add_hline(y=0, line=dict(color="#8892a6", width=1, dash="dot"), row=4, col=1)

    fig.add_trace(go.Scatter(x=dates, y=rsi14, name=f"RSI{P['rsi14']} Sortino", line=dict(color="#c084fc", width=1.8),
                             hovertemplate="%{y:.1f}<extra>RSI14 Sortino</extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=sma14, name=f"SMA{P['sma']}", line=dict(color="#94a3b8", width=1.3),
                             hovertemplate="%{y:.1f}<extra>SMA14</extra>"), row=5, col=1)
    fig.add_hline(y=P["gate"], line=dict(color="#64748b", width=1, dash="dash"), row=5, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in ci], y=[rsi14[i] for i in ci], mode="markers", name="cross up (arms buy)",
                             marker=dict(symbol="x", color="#38bdf8", size=9, line=dict(width=1.4)),
                             hovertemplate="cross UP %{x}<br>RSI14 %{y:.1f} (arms entry)<extra></extra>"), row=5, col=1)
    fig.add_trace(go.Scatter(x=[dates[i] for i in di], y=[rsi14[i] for i in di], mode="markers", name="cross down (sell)",
                             marker=dict(symbol="x", color="#fb923c", size=9, line=dict(width=1.4)),
                             hovertemplate="cross DOWN %{x}<br>RSI14 %{y:.1f} (SELL)<extra></extra>"), row=5, col=1)

    settings = (f"WIN={P.get('win')} SMOOTH={P.get('smooth')} RSI-Sortino={P.get('rsi14')} RSI-color={P.get('rsi20')} "
                f"SMA={P.get('sma')} GATE={P['gate']:.0f} GREEN={P['green']:.0f} RED={P['red']:.0f} + weekly-confirm")
    fig.update_layout(template="plotly_dark", height=1200, hovermode="x unified",
                      legend=dict(orientation="h", y=1.055, x=0, font=dict(size=10)),
                      margin=dict(l=60, r=30, t=96, b=40),
                      title=dict(text=f"Sortino-RSI Technique — {TK}<br><span style='font-size:12px;color:#9aa3b2'>{settings}</span>",
                                 x=0.01, y=0.985, font=dict(size=17)))
    fig.update_layout(spikedistance=-1, hoverdistance=-1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1.4, spikecolor="#aeb6c6",
                     spikedash="solid", type="date", rangeslider_visible=False)
    fig.update_yaxes(fixedrange=False)
    return fig


import glob
plotly_loaded = [False]                                                # include plotly.js exactly once, in the first rendered plot


def _inc():
    if plotly_loaded[0]:
        return False
    plotly_loaded[0] = True
    return True


# ---- SCENARIOS tab: overlaid equity curves + a button per scenario + stat cards
SCEN_ORDER = ["full-book", "allnames-max3-weekly", "top50-max10-weekly", "top50-max3-noweekly", "top50-max3-weekly"]
scen = [json.load(open(f)) for f in glob.glob("/app/.data/studies/scenarios/*.json")]
scen.sort(key=lambda s: SCEN_ORDER.index(s["name"]) if s["name"] in SCEN_ORDER else 99)
scen_html = ""
if scen:
    sfig = go.Figure()
    for sc in scen:
        xs = [d for d, _ in sc.get("monthly", [])]; eq = []; c = 1.0
        for _, v in sc.get("monthly", []):
            c *= (1 + v); eq.append(round(c, 4))
        sfig.add_trace(go.Scatter(x=xs, y=eq, name=sc["name"], mode="lines",
                                  hovertemplate="×%{y:.2f}<extra>" + sc["name"] + "</extra>"))
    sfig.update_layout(template="plotly_dark", height=560, hovermode="x unified", yaxis_type="log",
                       yaxis_title="growth of $1 (log)", legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
                       margin=dict(l=60, r=30, t=44, b=40), spikedistance=-1, hoverdistance=-1,
                       title=dict(text="Scenario equity curves — growth of $1 (log scale)", x=0.01, font=dict(size=15)))
    sfig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#aeb6c6", spikethickness=1.2, type="date")
    scen_fig = sfig.to_html(include_plotlyjs=_inc(), full_html=False, config=dict(displaylogo=False), div_id="scen_chart")
    btns = '<button class="sbtn active" onclick="scenSel(-1,this)">All</button>' + "".join(
        f'<button class="sbtn" onclick="scenSel({i},this)">{sc["name"]}</button>' for i, sc in enumerate(scen))

    def _card(sc):
        s = sc["summary"]
        rows = "".join(f'<div class=srow><span>{k}</span><b>{v}</b></div>' for k, v in [
            ("Total", f'{s["total"]:+,.0f}%'), ("CAGR", f'{s["cagr"]:.1f}%'), ("Sharpe", f'{s["sharpe"]:.2f}'),
            ("maxDD", f'{s["dd"]:.1f}%'), ("Trades", s["n_trades"]), ("Win", f'{s["win_rate"]:.0f}%'),
            ("Mean/trade", f'{s["mean_ret"]:+.2f}%')])
        return f'<div class=scard><h3>{sc["name"]}</h3>{rows}</div>'
    scen_html = f'<div class=sbtns>{btns}</div><div class=scards>{"".join(_card(sc) for sc in scen)}</div>{scen_fig}'

bodies = []
loaded = []
if scen_html:
    bodies.append(("Scenarios", "scenarios", scen_html))
for tk in TABS:
    safe = tk.replace("/", "_").replace(" ", "_")
    path = f"/app/.data/studies/viz_{safe}.json"
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    html = build_fig(d).to_html(include_plotlyjs=_inc(), full_html=False,
                                config=dict(scrollZoom=True, displaylogo=False), div_id=f"chart_{safe}")
    bodies.append((tk, safe, html)); loaded.append(safe)

tabs_btns = "".join(f'<button class="tab {"active" if i == 0 else ""}" onclick="showTab(\'{s}\')">{t}</button>'
                    for i, (t, s, _) in enumerate(bodies))
panels = "".join(f'<div class="panel" id="panel_{s}" style="display:{"block" if i == 0 else "none"}">{h}</div>'
                 for i, (t, s, h) in enumerate(bodies))

doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Sortino-RSI — {' / '.join(t for t,_,_ in bodies)}</title>
<style>body{{margin:0;background:#0e1016;color:#e8ebf2;font:14px -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1500px;margin:0 auto;padding:16px}}
.tabs{{display:flex;gap:8px;margin-bottom:10px}}
.tab{{background:#171a23;color:#8b93a7;border:1px solid #272d3b;border-radius:8px 8px 0 0;padding:9px 22px;font-size:14px;font-weight:600;cursor:pointer}}
.tab.active{{background:#1b1f2b;color:#e8ebf2;border-bottom-color:#1b1f2b}}
.sbtns{{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 16px}}
.sbtn{{background:#171a23;color:#c3c9d6;border:1px solid #2c3342;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;cursor:pointer}}
.sbtn.active{{background:#2a3550;color:#fff;border-color:#4c6ef5}}
.scards{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}}
.scard{{background:#161923;border:1px solid #272d3b;border-radius:12px;padding:14px 18px;min-width:150px}}
.scard.sel{{border-color:#4c6ef5;box-shadow:0 0 0 1px #4c6ef5}}
.scard h3{{margin:0 0 10px;font-size:13px;color:#8b93a7;font-weight:600}}
.srow{{display:flex;justify-content:space-between;gap:18px;font-variant-numeric:tabular-nums;padding:2px 0;font-size:13px}}
.srow span{{color:#8b93a7}} .srow b{{color:#e8ebf2}}
</style></head><body><div class=wrap>
<div class=tabs>{tabs_btns}</div>
{panels}
</div>
<script>
function showTab(s){{
  document.querySelectorAll('.panel').forEach(p=>p.style.display='none');
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel_'+s).style.display='block';
  event.target.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
function scenSel(i, btn){{
  document.querySelectorAll('.sbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  var cards=document.querySelectorAll('.scard'); var n=cards.length; var vis=[];
  for(var k=0;k<n;k++) vis.push(i<0?true:(k===i?true:'legendonly'));
  if(window.Plotly) Plotly.restyle('scen_chart', {{'visible': vis}});
  cards.forEach(function(c,k){{ c.classList.toggle('sel', i>=0 && k===i); }});
}}
</script></body></html>"""
open("/app/.data/served/index.html", "w", encoding="utf-8").write(doc)
print(f"wrote tabbed chart: {[t for t,_,_ in bodies]}  {len(doc):,} bytes")
