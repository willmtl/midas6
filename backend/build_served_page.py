#!/usr/bin/env python
"""Build the locally-served verification page: embed the Sortino-RSI PNG (base64) + rich stats + full trade table."""
import json, base64, os

VIZ = "/app/.data/studies/meta_sortino_viz.json"
PNG = "/app/.data/studies/meta_sortino.png"
OUT = "/app/.data/served/index.html"

d = json.load(open(VIZ))
TK = d.get("ticker", "META"); P = d["params"]; trades = d["trades"]
b64 = base64.b64encode(open(PNG, "rb").read()).decode()

# ---- stats (full history)
closed = [t for t in trades if not t.get("open")]
rets = [t["ret_pct"] for t in closed]
wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
n = len(closed); nw = len(wins)
avg_win = sum(wins) / len(wins) if wins else 0.0
avg_loss = sum(losses) / len(losses) if losses else 0.0
best = max(rets) if rets else 0.0; worst = min(rets) if rets else 0.0
avg_hold = sum(t["hold_days"] for t in closed) / n if n else 0.0
avg_r14 = sum(t.get("rsi14_entry", 0) for t in closed) / n if n else 0.0
compound = 1.0
for r in rets:
    compound *= (1 + r / 100.0)
compound_pct = (compound - 1) * 100
profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

# ---- rows (full history, with running cumulative)
rows = []; cum = 1.0
for t in trades:
    rp = t["ret_pct"]; cls = "pos" if rp > 0 else "neg"
    cum *= (1 + rp / 100.0); cumpct = (cum - 1) * 100
    openflag = ' <span class="open">open</span>' if t.get("open") else ""
    rows.append(
        f'<tr><td class="mut">#{t["n"]}</td>'
        f'<td>{t["entry"]}</td><td class="num">${t["entry_px"]:g}</td>'
        f'<td>{t["exit"]}{openflag}</td><td class="num">${t["exit_px"]:g}</td>'
        f'<td class="num">{t["hold_days"]}</td>'
        f'<td class="num mut">{t.get("rsi14_entry", 0):.0f}</td>'
        f'<td class="num {cls}">{rp:+.1f}%</td>'
        f'<td class="num">{cumpct:+.0f}%</td></tr>')
tbl = "\n".join(rows)


def stat(v, lbl, cls=""):
    return f'<div class="stat"><b class="{cls}">{v}</b><span>{lbl}</span></div>'


stats = "".join([
    stat(n, "round-trips"),
    stat(f"{100*nw/n:.0f}%" if n else "–", "win rate"),
    stat(f"{nw}/{n}", "winners"),
    stat(f"+{compound_pct:,.0f}%", "compounded", "pos" if compound_pct > 0 else "neg"),
    stat(f"+{avg_win:.1f}%", "avg win", "pos"),
    stat(f"{avg_loss:.1f}%", "avg loss", "neg"),
    stat(f"+{best:.1f}%", "best", "pos"),
    stat(f"{worst:.1f}%", "worst", "neg"),
    stat(f"{avg_hold:.0f}d", "avg hold"),
    stat(f"{profit_factor:.2f}", "profit factor"),
    stat(f"{avg_r14:.0f}", "avg RSI14 entry"),
])

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sortino-RSI Technique &mdash; {TK}</title>
<style>
:root{{--bg:#0e1016;--card:#161923;--card2:#1b1f2b;--ink:#e8ebf2;--mut:#8b93a7;--line:#272d3b;--grn:#3ecb84;--red:#e5605e;--acc:#6ea8fe;--pur:#a98bfb}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1500px;margin:0 auto;padding:34px 24px 70px}}
h1{{font-size:27px;margin:0 0 6px;letter-spacing:-.015em}}
.sub{{color:var(--mut);margin:0 0 24px;font-size:14px;max-width:900px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:26px}}
h2{{font-size:15px;margin:0 0 14px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;font-weight:600}}
img{{width:100%;height:auto;border-radius:10px;display:block}}
.rule{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 24px;font-size:13px}}
.chip{{background:var(--card2);border:1px solid var(--line);border-radius:999px;padding:7px 14px;color:var(--mut)}}
.chip b{{color:var(--ink);font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:2px;background:var(--line);border-radius:12px;overflow:hidden;border:1px solid var(--line)}}
.stat{{background:var(--card);padding:16px 18px;text-align:center}}
.stat b{{font-size:24px;font-variant-numeric:tabular-nums;display:block;line-height:1.1}}
.stat span{{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-top:5px;display:block}}
.tblwrap{{overflow-x:auto;max-height:640px;overflow-y:auto;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{text-align:left;padding:9px 16px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;position:sticky;top:0;background:var(--card2);z-index:1}}
td.num{{text-align:right}} td.mut{{color:var(--mut)}}
td.pos{{color:var(--grn);font-weight:600}} td.neg{{color:var(--red);font-weight:600}}
.open{{color:var(--acc);font-size:11px;border:1px solid var(--acc);border-radius:4px;padding:1px 5px;margin-left:4px}}
tbody tr:hover td{{background:#1a1e2a}}
.pos{{color:var(--grn)}} .neg{{color:var(--red)}}
</style></head><body><div class="wrap">
<h1>Sortino-RSI Technique &mdash; {TK}</h1>
<p class="sub">Buy when price is in an uptrend (RSI-20 of the HIGH &gt; {P['green']:.0f}) and downside-adjusted outperformance vs SPY
turns up from a dip: the RSI-14 of the (smoothed) rel-Sortino crossing up its SMA-{P['sma']} while below {P['gate']:.0f} <b>arms</b> the entry for {P.get('arm_win', 5)} bars, and the buy fires when price confirms green (RSI-20 on Close &gt; {P['green']:.0f}) within that window. Exit when the uptrend color goes red
(RSI-20 &lt; {P['red']:.0f}). Shaded bands = time in position.</p>
<div class="rule">
  <span class="chip">RSI-20 source = <b>CLOSE</b></span>
  <span class="chip"><b>Green</b> &gt;{P['green']:.0f} &middot; <b>Red</b> &lt;{P['red']:.0f}</span>
  <span class="chip">rel-Sortino(<b>{P['win']}</b>) vs SPY, smoothed SMA-<b>{P.get('smooth', 14)}</b></span>
  <span class="chip">&rarr; RSI(<b>{P['rsi14']}</b>) crosses up SMA-{P['sma']} while &lt;{P['gate']:.0f} = <b>arms</b></span>
  <span class="chip">green within <b>{P.get('arm_win', 5)}</b> bars = <b>buy</b></span>
  <span class="chip">exit = color turns red</span>
</div>

<div class="card"><h2>Full-history trade stats</h2><div class="grid">{stats}</div></div>

<div class="card"><h2>Signal &amp; indicators</h2>
<img src="data:image/png;base64,{b64}" alt="{TK} Sortino-RSI chart"></div>

<div class="card"><h2>Every round-trip ({len(trades)} trades)</h2>
<div class="tblwrap"><table>
<thead><tr><th>#</th><th>Entry</th><th>Entry $</th><th>Exit</th><th>Exit $</th><th>Hold</th><th>RSI14</th><th>Return</th><th>Cumulative</th></tr></thead>
<tbody>{tbl}</tbody></table></div></div>

</div></body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(html)
print(f"wrote {OUT}  {len(html):,} bytes  ticker={TK}  trades={len(trades)}  wins={nw}/{n}  compounded={compound_pct:+.0f}%")
