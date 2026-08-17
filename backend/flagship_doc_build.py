# -*- coding: utf-8 -*-
"""Render the flagship full-history document (self-contained HTML) from the enriched trace.
ALL per-purchase fundamentals shown are POINT-IN-TIME as of the trade date (45-day report lag). There is
deliberately NO forward P/E: our providers only expose today's estimate, which would be look-ahead.
Reads  /app/.data/studies/flagship_history_enriched.json
Writes /app/.data/studies/flagship_history.html
"""
import json, html
from pathlib import Path

IN = Path("/app/.data/studies/flagship_history_enriched.json")
OUT = Path("/app/.data/studies/flagship_history.html")
D = json.load(open(IN))
S, P = D["summary"], D["perf"]


def esc(x):
    return html.escape(str(x)) if x is not None else ""


def pct(x, plus=True):
    if x is None:
        return '<span class="mut">—</span>'
    v = x * 100
    cls = "pos" if v >= 0 else "neg"
    sign = "+" if (plus and v >= 0) else ""
    return f'<span class="{cls}">{sign}{v:.1f}%</span>'


def pctp(v, plus=True):   # v already in percent
    if v is None:
        return '<span class="mut">—</span>'
    cls = "pos" if v >= 0 else "neg"
    sign = "+" if (plus and v >= 0) else ""
    return f'<span class="{cls}">{sign}{v:.1f}%</span>'


def cap(x):
    if x is None:
        return '<span class="mut">—</span>'
    if x >= 1e9:
        return f"${x/1e9:.1f}B"
    if x >= 1e6:
        return f"${x/1e6:.0f}M"
    return f"${x:,.0f}"


def pbfmt(x):
    return f"{x:.2f}" if x is not None else '<span class="mut">—</span>'


def pefmt(x):
    if x is None:
        return '<span class="mut">—</span>'
    if x < 0:
        return f'<span class="neg" title="negative P/E = loss-making">{x:.1f}</span>'
    if x > 60:
        return f'<span class="warn" title="expensive multiple">{x:.0f}</span>'
    return f"{x:.1f}"


def roefmt(x):
    if x is None:
        return '<span class="mut">—</span>'
    cls = "pos" if x > 0 else "neg"
    return f'<span class="{cls}">{x*100:.0f}%</span>'


def defmt(x):
    if x is None:
        return '<span class="mut">—</span>'
    cls = "warn" if x > 2 else ""
    return f'<span class="{cls}">{x:.2f}</span>'


def gpafmt(x):
    if x is None:
        return '<span class="mut">—</span>'
    cls = "pos" if x >= 0.33 else ""
    return f'<span class="{cls}">{x:.2f}</span>'


def winfmt(x):
    if x is None:
        return '<span class="mut">—</span>'
    cls = "pos" if x >= 50 else "neg"
    return f'<span class="{cls}">{x:.0f}%</span>'


def tierbadge(mc):
    """Market-cap tier badge at purchase."""
    if mc is None:
        return '<span class="mut">—</span>'
    if mc < 5e8:
        t, c = "micro", "t-micro"
    elif mc < 2e9:
        t, c = "small", "t-small"
    elif mc < 1e10:
        t, c = "mid", "t-mid"
    else:
        t, c = "large", "t-large"
    return f'<span class="tier {c}">{t}</span>'


def tick(t, delisted=False):
    dl = ' <span class="dl" title="delisted / acquired during the hold — exited at last/deal price">†</span>' if delisted else ""
    return f'<span class="tk">{esc(t)}</span>{dl}'


# ---------- column tooltips ----------
TIP = {
    "Ticker": "Exchange symbol as held. † = acquired or delisted during a hold (exited at its last/deal price).",
    "Company": "Issuer name at the time it was bought.",
    "Sector(s)": "GICS sector sleeve(s) whose ETF accelerated into the momentum top-10 when this name was picked.",
    "Sector": "GICS sector sleeve whose ETF accelerated into the momentum top-10 when this name was picked.",
    "ETF": "The sector ETF whose momentum acceleration defines the sleeve.",
    "Held": "Number of distinct rebalance months this name was a holding.",
    "P/B": "Price-to-book at purchase = purchase-month close × point-in-time shares ÷ point-in-time book equity. The value metric the engine ranks on — the cheapest qualifying name wins.",
    "P/E": "Standard trailing P/E = Price ÷ trailing-12-month EPS (= market cap ÷ TTM net income), point-in-time at purchase. NEGATIVE = loss-making (negative TTM earnings). Blank only when fewer than 4 quarters of filings were public on the trade date.",
    "ROE": "Trailing-12-month return on equity at purchase (TTM net income ÷ book equity). Above 0 = profitable over the trailing year; negative = loss-making.",
    "D/E": "Debt-to-equity at purchase (point-in-time total debt ÷ book equity). Balance-sheet leverage.",
    "Mkt cap": "Market capitalization at purchase, in USD. The strategy tilts small — most picks are under $2B.",
    "Tier": "Market-cap category at purchase: micro (< $500M), small ($500M–$2B), mid ($2B–$10B), large (≥ $10B). The strategy tilts micro/small.",
    "Avg ret": "Average realized next-month return across every month this name was held.",
    "Win%": "Share of this name's holds that ended positive (green months ÷ total months held).",
    "Rev g": "Median trailing year-over-year revenue growth at the times it was bought (point-in-time).",
    "GP/A": "Gross profitability (Novy-Marx) at purchase: trailing gross profit ÷ total assets — a quality signal that works independently of P/B. Higher = more productive assets.",
    "Best": "Best single-month return this name ever posted as a holding.",
    "Worst": "Worst single-month return this name ever posted as a holding.",
    "Contrib.": "Summed A/D-conviction-weighted return contribution to the portfolio across all holds. This is the sort key — a few names carry the book.",
    "Span": "First → last rebalance month the name appears as a holding.",
    "Weight": "Basket weight: 2× when the name shows A/D accumulation into price weakness (conviction), else 1×.",
    "In top-10": "How many months this sleeve's ETF accelerated into the momentum top-10.",
    "Picked from": "How many of those top-10 months the engine found a qualifying value pick in the sleeve.",
    "Skipped": "Months the sleeve was in the top-10 but had no qualifying value stock (foreign/commodity/bond) — the slot was skipped, held as cash-equivalent.",
    "Avg pick ret": "Average realized next-month return of picks drawn from this sleeve.",
    "Month": "Rebalance date the pick was bought.",
    "Return": "Realized next-month return of the pick (purchase-month close to next rebalance).",
}


def th(label, cls=""):
    """Header cell with a hover/focus '?' tooltip pulled from TIP (keyed by the plain label)."""
    tip = TIP.get(label.replace("&nbsp;", " "), "")
    c = f' class="{cls}"' if cls else ""
    hint = (f'<span class="hint" title="{esc(tip)}" aria-label="{esc(tip)}" tabindex="0">?</span>'
            if tip else "")
    return f'<th{c}>{label}{hint}</th>'


# ---------- equity curve ----------
curve = D["curve"]
curve_js = json.dumps([{"d": c["date"], "f": round(c["flagship"] * 100000), "s": round(c["spy"] * 100000)}
                       for c in curve])
stock_trades_js = json.dumps({s["ticker"]: {"co": s["company"], "dl": s["delisted"],
    "t": [{"d": t["date"], "s": t["sector"], "pb": t["pb"], "pe": t["pe"], "roe": t["roe"], "de": t["de"],
           "gpa": t["gpa"], "rg": t["rev_g"], "mc": t["mc"], "w": t["weight"], "r": t["ret"], "cv": t["conviction"]}
          for t in s["trades"]]} for s in D["by_stock"]}, default=str)
bm, wm = S["best_month"], S["worst_month"]

# ---------- KPIs ----------
kpis = [
    ("Total return", f'+{P["total"]:.1f}%', f'{S["months"]} months, survivorship-free'),
    ("$100,000 becomes", f'${S["final_100k_flagship"]:,}', f'SPY: ${S["final_100k_spy"]:,}'),
    ("CAGR", f'{P["annual"]:.1f}%', "annualized"),
    ("Sharpe", f'{P["sharpe"]:.2f}', "monthly, annualized"),
    ("Max drawdown", f'{P["dd"]:.1f}%', "peak-to-trough"),
    ("t-statistic", f'{P["t_stat"]:.2f}', "> 3 = strong"),
    ("vs S&P 500", f'+{P["vs_spy"]:.0f}pp', "excess total return"),
    ("Names traded", f'{S["unique_stocks"]}', f'{S["total_picks"]} monthly picks'),
]

# ---------- fundamentals-at-purchase band ----------
funda = [
    ("Median P/B", f'{S["med_pb"]:.2f}', "book multiple"),
    ("Median P/E", f'{S["med_pe"]:.1f}', "TTM · profitable names"),
    ("Median ROE", f'{S["med_roe"]*100:.0f}%', "trailing 12-month"),
    ("Median D/E", f'{S["med_de"]:.2f}', "debt / equity"),
    ("Median mkt-cap", cap(S["med_mktcap"]), "size at buy"),
    ("Profitable", f'{S["pct_profitable"]:.0f}%', f'{S.get("pct_lossmaking",0):.0f}% loss-making'),
    ("Small-cap", f'{S["pct_smallcap"]:.0f}%', "< $2B at buy"),
    ("Conviction 2×", f'{S["conviction_pct"]:.0f}%', "A/D-weighted"),
]

# ---------- calendar-year rows ----------
cal_rows = []
for c in D["calendar"]:
    cal_rows.append(f"""<tr>
      <td class="l mono">{esc(c['year'])}</td>
      <td class="num strong">{pctp(c['strategy'])}</td>
      <td class="num">{pctp(c['spy'])}</td>
      <td class="num">{pctp(c['excess'])}</td>
      <td class="num mut">{c['months']}</td>
    </tr>""")

# ---------- leaderboard ----------
stock_rows = []
for s in D["by_stock"]:
    secs = ", ".join(s["sectors"])
    span = f'{s["first"][:7]} – {s["last"][:7]}' if s["first"] else ""
    stock_rows.append(f"""<tr class="clk" data-ticker="{esc(s['ticker'])}" title="Click for this name's full trade history">
      <td class="l">{tick(s['ticker'], s['delisted'])}</td>
      <td class="l nm">{esc(s['company']) or '<span class="mut">—</span>'}</td>
      <td class="l sec">{esc(secs)}</td>
      <td class="num">{s['n']}</td>
      <td class="num">{winfmt(s.get('win_rate'))}</td>
      <td class="num">{pbfmt(s['med_pb'])}</td>
      <td class="num">{pefmt(s['med_pe'])}</td>
      <td class="num">{roefmt(s['med_roe'])}</td>
      <td class="num">{gpafmt(s.get('med_gpa'))}</td>
      <td class="num">{pct(s.get('med_rev_g'))}</td>
      <td class="num">{defmt(s['med_de'])}</td>
      <td class="num">{cap(s['med_mc'])}</td>
      <td class="num">{tierbadge(s['med_mc'])}</td>
      <td class="num">{pct(s['avg_ret'])}</td>
      <td class="num">{pct(s['best'])}</td>
      <td class="num">{pct(s['worst'])}</td>
      <td class="num strong">{pct(s['sum_contrib'])}</td>
      <td class="num span">{esc(span)}</td>
    </tr>""")

# ---------- sectors ----------
maxpick = D["by_sector"][0]["picked"] or 1
sector_rows = []
for s in D["by_sector"]:
    barw = min(100, s["picked"] / maxpick * 100)
    sk = f'{s["skipped"]}' if s["skipped"] else '<span class="mut">0</span>'
    sector_rows.append(f"""<tr>
      <td class="l nm">{esc(s['sector'])}</td>
      <td class="l"><span class="tk">{esc(s['etf'])}</span></td>
      <td class="num">{s['in_top']}</td>
      <td class="num"><div class="barwrap"><span class="barfill" style="width:{barw:.0f}%"></span><b>{s['picked']}</b></div></td>
      <td class="num">{sk}</td>
      <td class="num">{pct(s['avg_pick_ret'])}</td>
    </tr>""")


# ---------- best / worst ----------
def pick_rows(lst):
    out = []
    for r in lst:
        out.append(f"""<tr>
          <td class="l mono">{esc(r['date'])}</td>
          <td class="l">{tick(r['ticker'], r['delisted'])}</td>
          <td class="l nm">{esc(r['company']) or '<span class="mut">—</span>'}</td>
          <td class="l sec">{esc(r['sector'])}</td>
          <td class="num">{pbfmt(r['pb'])}</td>
          <td class="num">{pefmt(r['pe'])}</td>
          <td class="num">{roefmt(r['roe'])}</td>
          <td class="num strong">{pct(r['ret'])}</td>
        </tr>""")
    return "\n".join(out)


# ---------- full blotter ----------
BLOTTER_COLS = 13


def acc_fmt(a):
    return f'{a*100:+.1f}%' if a is not None else '·'


def sector_chips(m):
    """One chip per top-10 accelerating sector this month: name + acceleration, then either the value name it
    resolved to (✓) or the reason the slot was skipped (⊘). This is the 'from the sector to the stock, and
    why skipped' trail the reader asked for."""
    pick_by_etf = {p["etf"]: p for p in m["picks"]}
    skip_by_etf = {sk["etf"]: sk for sk in m["skipped"]}
    chips = []
    for i, ts in enumerate(m["top_sectors"], 1):
        etf, nm, acc = ts["etf"], ts["sector"], ts.get("accel")
        rank = f'<span class="rk">#{i}</span>'
        head = f'{rank}<b>{esc(nm)}</b> <span class="acc">{acc_fmt(acc)}</span>'
        if etf in pick_by_etf:
            p = pick_by_etf[etf]
            rr = pct(p["ret"]) if p["ret"] is not None else '<span class="mut">held</span>'
            chips.append(f'<span class="chip pick" title="Accelerated into the top-10 (accel {acc_fmt(acc)}); '
                         f'cheapest qualifying value name = {esc(p["ticker"])} at P/B {p.get("pb") and round(p["pb"],2)}">'
                         f'{head} → <span class="tk">{esc(p["ticker"])}</span> {rr}</span>')
        elif etf in skip_by_etf:
            sk = skip_by_etf[etf]
            reason = sk.get("reason", "no qualifying value stock")
            chips.append(f'<span class="chip skip" title="{esc(reason)}">{head} '
                         f'<span class="xm">⊘ skipped</span> <span class="rsn">{esc(reason)}</span></span>')
        else:
            chips.append(f'<span class="chip">{head}</span>')
    return "".join(chips)


blotter = []
for m in D["months"]:
    picks = m["picks"]
    npos = sum(1 for p in picks if p["ret"] is not None)
    nskip = len(m["skipped"])
    exc = (m["basket_ret"] - m["spy_ret"]) * 100
    blotter.append(f"""<tr class="mhdr">
      <td class="l mono" colspan="6"><b>{esc(m['date'])}</b> · {npos} holdings · {nskip} sector{'s' if nskip!=1 else ''} skipped</td>
      <td class="num" colspan="3">SPY {pct(m['spy_ret'])}</td>
      <td class="num" colspan="4">basket <b>{pct(m['basket_ret'])}</b> <span class="{'pos' if exc>=0 else 'neg'}">({exc:+.1f} vs SPY)</span></td>
    </tr>
    <tr class="srow"><td colspan="{BLOTTER_COLS}"><div class="chips">{sector_chips(m)}</div></td></tr>""")
    for p in sorted(picks, key=lambda x: -(x["ret"] if x["ret"] is not None else -9)):
        conv = ' <span class="conv" title="A/D-conviction 2× weight">2×</span>' if p["conviction"] else ""
        rr = pct(p["ret"]) if p["ret"] is not None else '<span class="mut">n/a</span>'
        wt = f'{p["weight"]:.0f}×' if p.get("weight") else '1×'
        why = (f'Cheapest positive-P/B value name (P/B {p.get("pb") and round(p["pb"],2)}) among the '
               f'{"small-caps" if (p.get("mktcap_usd") or 0) < 2e9 else "holdings"} of {esc(p["sector"])}, '
               f'which accelerated into the momentum top-10.')
        blotter.append(f"""<tr title="{why}">
          <td class="l">{tick(p['ticker'], p['delisted'])}{conv}</td>
          <td class="l nm">{esc(p['company']) or '<span class="mut">—</span>'}</td>
          <td class="l sec">{esc(p['sector'])}</td>
          <td class="num">{tierbadge(p['mktcap_usd'])}</td>
          <td class="num">{pbfmt(p['pb'])}</td>
          <td class="num">{pefmt(p.get('pe'))}</td>
          <td class="num">{roefmt(p.get('roe'))}</td>
          <td class="num">{gpafmt(p.get('gpa'))}</td>
          <td class="num">{pct(p.get('rev_g'))}</td>
          <td class="num">{defmt(p.get('de'))}</td>
          <td class="num cap">{cap(p['mktcap_usd'])}</td>
          <td class="num">{wt}</td>
          <td class="num strong">{rr}</td>
        </tr>""")

HTML = f"""<title>usca_small — Flagship Trade History</title>
<style>
:root {{
  --paper:#eceff0; --card:#ffffff; --ink:#131a1e; --ink2:#48555c; --mut:#93a1a7;
  --line:#dbe1e3; --line2:#e7ebec; --accent:#0e7c6b; --gold:#a9772a; --pos:#17805a; --neg:#bb4436;
  --warn:#b07a1e; --band:#e3e8e9; --shadow:0 1px 2px rgba(20,30,35,.06),0 8px 24px rgba(20,30,35,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --paper:#0c1113; --card:#141b1f; --ink:#e5eaec; --ink2:#9fb0b6; --mut:#5f7178;
    --line:#232d31; --line2:#1b2327; --accent:#2ea48f; --gold:#d6a648; --pos:#35b07e; --neg:#e0705f;
    --warn:#d6a648; --band:#111719; --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35); }}
}}
:root[data-theme="light"] {{ --paper:#eceff0; --card:#ffffff; --ink:#131a1e; --ink2:#48555c; --mut:#93a1a7;
  --line:#dbe1e3; --line2:#e7ebec; --accent:#0e7c6b; --gold:#a9772a; --pos:#17805a; --neg:#bb4436;
  --warn:#b07a1e; --band:#e3e8e9; --shadow:0 1px 2px rgba(20,30,35,.06),0 8px 24px rgba(20,30,35,.05); }}
:root[data-theme="dark"] {{ --paper:#0c1113; --card:#141b1f; --ink:#e5eaec; --ink2:#9fb0b6; --mut:#5f7178;
  --line:#232d31; --line2:#1b2327; --accent:#2ea48f; --gold:#d6a648; --pos:#35b07e; --neg:#e0705f;
  --warn:#d6a648; --band:#111719; --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35); }}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0}}
.doc{{
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"SF Mono","Cascadia Code",Consolas,"Liberation Mono",ui-monospace,monospace;
  background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:15px; line-height:1.55; padding:0 20px 100px;
}}
.wrap{{max-width:1180px; margin:0 auto}}
.tk{{font-family:var(--mono); font-weight:600; letter-spacing:.02em; font-size:.9em}}
.mono{{font-family:var(--mono); font-variant-numeric:tabular-nums}}
.num{{text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap}}
.l{{text-align:left}}
.pos{{color:var(--pos); font-variant-numeric:tabular-nums}}
.neg{{color:var(--neg); font-variant-numeric:tabular-nums}}
.warn{{color:var(--warn); font-variant-numeric:tabular-nums}}
.mut{{color:var(--mut)}}
.strong{{font-weight:700}}
.mast{{padding:64px 0 30px; border-bottom:2px solid var(--ink)}}
.eyebrow{{font-family:var(--mono); font-size:12px; letter-spacing:.22em; text-transform:uppercase; color:var(--accent); margin:0 0 18px}}
.mast h1{{font-family:var(--serif); font-weight:600; font-size:clamp(30px,5vw,50px); line-height:1.05; margin:0 0 6px; text-wrap:balance; letter-spacing:-.01em}}
.mast .sub{{font-family:var(--serif); font-style:italic; color:var(--ink2); font-size:clamp(17px,2.4vw,21px); margin:0 0 26px}}
.headline{{display:flex; flex-wrap:wrap; align-items:flex-end; gap:10px 34px; margin-top:10px}}
.bignum{{font-family:var(--serif); font-weight:700; font-size:clamp(46px,9vw,86px); line-height:.92; color:var(--gold); letter-spacing:-.02em}}
.bigsub{{display:flex; flex-direction:column; gap:2px; padding-bottom:8px}}
.bigsub .a{{font-size:20px; font-weight:700}}
.bigsub .b{{font-size:13px; color:var(--ink2)}}
.codechip{{font-family:var(--mono); font-size:12px; background:var(--band); color:var(--ink2); padding:3px 9px; border-radius:5px; border:1px solid var(--line)}}
.meta-row{{display:flex; flex-wrap:wrap; gap:8px 14px; margin-top:22px; font-size:12.5px; color:var(--ink2); font-family:var(--mono)}}
.meta-row b{{color:var(--ink)}}
.nav{{position:sticky; top:0; z-index:20; display:flex; flex-wrap:wrap; gap:2px; padding:9px 0; margin-bottom:6px;
  background:color-mix(in srgb,var(--paper) 88%,transparent); backdrop-filter:blur(8px); border-bottom:1px solid var(--line)}}
.nav a{{font-family:var(--mono); font-size:11.5px; letter-spacing:.04em; text-transform:uppercase; color:var(--ink2); text-decoration:none; padding:5px 11px; border-radius:6px}}
.nav a:hover{{background:var(--band); color:var(--ink)}}
.nav a:focus-visible{{outline:2px solid var(--accent); outline-offset:1px}}
.kpis{{display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; margin:34px 0 10px; box-shadow:var(--shadow)}}
.kpi{{background:var(--card); padding:17px 18px}}
.kpi .k{{font-family:var(--mono); font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--mut); margin-bottom:7px}}
.kpi .v{{font-family:var(--serif); font-size:26px; font-weight:700; line-height:1; letter-spacing:-.01em}}
.kpi .d{{font-size:12px; color:var(--ink2); margin-top:6px}}
.kpi.hero .v{{color:var(--gold)}}
.funda{{display:grid; grid-template-columns:repeat(8,1fr); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; margin-top:14px; box-shadow:var(--shadow)}}
.fu{{background:var(--card); padding:13px 14px}}
.fu .k{{font-family:var(--mono); font-size:10px; letter-spacing:.05em; text-transform:uppercase; color:var(--mut); margin-bottom:5px}}
.fu .v{{font-family:var(--serif); font-size:20px; font-weight:700; line-height:1}}
.fu .d{{font-size:10.5px; color:var(--mut); margin-top:4px}}
section{{margin-top:60px; scroll-margin-top:60px}}
.shead{{display:flex; align-items:baseline; gap:14px; border-bottom:1px solid var(--ink); padding-bottom:9px; margin-bottom:8px}}
.shead h2{{font-family:var(--serif); font-weight:600; font-size:25px; margin:0; letter-spacing:-.01em}}
.shead .cnt{{font-family:var(--mono); font-size:12px; color:var(--mut); margin-left:auto}}
.lede{{color:var(--ink2); font-size:14.5px; max-width:76ch; margin:12px 0 20px}}
.lede b{{color:var(--ink)}}
.grid2{{display:grid; grid-template-columns:1.1fr .9fr; gap:22px; align-items:start}}
@media (max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
.chartcard{{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px 20px 12px; box-shadow:var(--shadow)}}
.chartcard canvas{{width:100%; height:320px; display:block}}
.legend{{display:flex; gap:22px; flex-wrap:wrap; margin:14px 2px 2px; font-size:13px; font-family:var(--mono)}}
.legend i{{display:inline-block; width:22px; height:3px; border-radius:2px; margin-right:8px; vertical-align:middle}}
.legend .fl{{background:var(--gold)}} .legend .sp{{background:var(--mut)}}
.chart-note{{font-size:12px; color:var(--mut); margin:6px 2px 0}}
.tablecard{{background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:var(--shadow); margin-top:20px}}
.scrollx{{overflow-x:auto}}
.scrolly{{max-height:74vh; overflow:auto}}
table{{border-collapse:collapse; width:100%; font-size:13.5px}}
thead th{{position:sticky; top:0; z-index:1; background:var(--band); color:var(--ink2); font-family:var(--mono); font-weight:600; font-size:11px; letter-spacing:.04em; text-transform:uppercase; text-align:right; padding:11px 12px; border-bottom:1px solid var(--line); white-space:nowrap}}
thead th.l{{text-align:left}}
tbody td{{padding:9px 12px; border-bottom:1px solid var(--line2)}}
tbody tr:hover td{{background:var(--band)}}
tbody tr:last-child td{{border-bottom:none}}
.nm{{color:var(--ink2); max-width:210px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}}
.sec{{color:var(--ink2); font-size:12.5px; max-width:170px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}}
.span{{color:var(--mut); font-family:var(--mono); font-size:12px}}
.cap{{color:var(--ink2); font-family:var(--mono); font-size:12.5px}}
.dl{{color:var(--gold); font-weight:700; cursor:help}}
.hint{{display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px; margin-left:5px; border-radius:50%; border:1px solid var(--line); color:var(--mut); font-family:var(--mono); font-size:9px; font-weight:700; line-height:1; cursor:help; vertical-align:middle; text-transform:none; letter-spacing:0}}
.hint:hover,.hint:focus-visible{{color:var(--accent); border-color:var(--accent); outline:none}}
th.l .hint{{margin-left:5px}}
.conv{{font-family:var(--mono); font-size:10px; background:color-mix(in srgb,var(--accent) 18%,transparent); color:var(--accent); padding:1px 5px; border-radius:4px; font-weight:700; vertical-align:middle}}
.tier{{font-family:var(--mono); font-size:10px; padding:1px 6px; border-radius:4px; font-weight:700; letter-spacing:.03em; text-transform:uppercase}}
.t-micro{{background:color-mix(in srgb,var(--gold) 24%,transparent); color:var(--gold)}}
.t-small{{background:color-mix(in srgb,var(--accent) 20%,transparent); color:var(--accent)}}
.t-mid{{background:color-mix(in srgb,var(--ink2) 16%,transparent); color:var(--ink2)}}
.t-large{{background:color-mix(in srgb,var(--mut) 16%,transparent); color:var(--mut)}}
.lede2{{color:var(--mut); font-size:12.5px; margin:0 0 6px}}
.sortth{{cursor:pointer; user-select:none}}
.sortth:hover{{color:var(--ink)}}
.arrow{{color:var(--accent); font-size:10px}}
tbody tr.clk{{cursor:pointer}}
tbody tr.clk:hover td{{background:color-mix(in srgb,var(--accent) 10%,transparent)}}
.srow td{{padding:8px 12px 13px; background:var(--paper)}}
.chips{{display:flex; flex-wrap:wrap; gap:6px}}
.chip{{font-family:var(--mono); font-size:11px; padding:3px 8px; border-radius:6px; border:1px solid var(--line); background:var(--card); white-space:nowrap; display:inline-flex; align-items:center; gap:5px}}
.chip .rk{{color:var(--mut); font-size:10px}}
.chip .acc{{color:var(--ink2)}}
.chip.pick{{border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}}
.chip.pick .tk{{color:var(--accent)}}
.chip.skip{{opacity:.72}}
.chip .xm{{color:var(--neg); font-weight:700}}
.chip .rsn{{color:var(--mut); font-size:10px; max-width:300px; overflow:hidden; text-overflow:ellipsis}}
.pane{{position:fixed; inset:0; z-index:60; display:flex; justify-content:flex-end}}
.pane[hidden]{{display:none}}
.pane-bd{{position:absolute; inset:0; background:color-mix(in srgb,#000 45%,transparent); backdrop-filter:blur(2px)}}
.pane-card{{position:relative; width:min(780px,95vw); height:100%; overflow:auto; background:var(--card); border-left:1px solid var(--line); box-shadow:-12px 0 40px rgba(0,0,0,.32); padding:26px 26px 60px; animation:slidein .18s ease}}
@keyframes slidein{{from{{transform:translateX(30px); opacity:.5}} to{{transform:none; opacity:1}}}}
.pane-x{{position:absolute; top:14px; right:16px; width:30px; height:30px; border-radius:50%; border:1px solid var(--line); background:var(--paper); color:var(--ink2); font-size:18px; cursor:pointer; line-height:1}}
.pane-x:hover{{color:var(--ink)}}
.pane-h{{margin:0 0 4px}}
.tk.big{{font-family:var(--mono); font-size:22px; font-weight:700; color:var(--ink)}}
.pane-co{{color:var(--ink2); font-size:15px}}
.pane-sub{{color:var(--mut); font-size:12px; font-family:var(--mono); margin-bottom:16px}}
.ptab{{width:100%; border-collapse:collapse; font-size:13px}}
.ptab th{{position:sticky; top:0; background:var(--band); color:var(--ink2); font-family:var(--mono); font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; text-align:right; padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap}}
.ptab th.l{{text-align:left}}
.ptab td{{padding:8px 10px; border-bottom:1px solid var(--line2)}}
.barwrap{{position:relative; display:flex; align-items:center; justify-content:flex-end; gap:8px}}
.barwrap b{{position:relative; z-index:1; font-weight:700}}
.barfill{{position:absolute; right:0; top:50%; transform:translateY(-50%); height:20px; background:color-mix(in srgb,var(--accent) 15%,transparent); border-radius:4px}}
.mhdr td{{background:var(--band); border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding-top:11px; padding-bottom:11px}}
.sk{{color:var(--neg); font-size:11px; font-family:var(--mono); margin-left:6px}}
.calcard table{{font-size:14px}}
.foot{{margin-top:70px; padding-top:22px; border-top:1px solid var(--line); color:var(--mut); font-size:12.5px; max-width:88ch}}
.foot b{{color:var(--ink2)}}
.foot p{{margin:0 0 10px}}
.tog{{position:fixed; right:16px; bottom:16px; z-index:40; font-family:var(--mono); font-size:12px; background:var(--card); color:var(--ink2); border:1px solid var(--line); border-radius:20px; padding:8px 14px; cursor:pointer; box-shadow:var(--shadow)}}
.tog:hover{{color:var(--ink)}} .tog:focus-visible{{outline:2px solid var(--accent); outline-offset:2px}}
@media (max-width:720px){{ .kpis{{grid-template-columns:repeat(2,1fr)}} .funda{{grid-template-columns:repeat(2,1fr)}} .doc{{padding:0 14px 80px}} .mast{{padding-top:40px}} }}
@media (prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style>

<div class="doc">
<div class="wrap">

  <header class="mast">
    <p class="eyebrow">Flagship Strategy · Full Trade History</p>
    <h1>Small-Cap Value, Bought in Accelerating Sectors</h1>
    <p class="sub">Every sector held and every stock picked that compounded $100,000 into ${S['final_100k_flagship']:,} — with the fundamentals as they stood on each purchase date.</p>
    <div class="headline">
      <div class="bignum">+{P['total']:.0f}%</div>
      <div class="bigsub">
        <span class="a">${S['final_100k_flagship']:,} <span class="mut" style="font-weight:400">from $100k</span></span>
        <span class="b">{S['first']} → {S['last']} · {S['months']} months · vs SPY ${S['final_100k_spy']:,}</span>
      </div>
      <span class="codechip">arm: usca_small</span>
    </div>
    <div class="meta-row">
      <span>CAGR <b>{P['annual']:.1f}%</b></span><span>Sharpe <b>{P['sharpe']:.2f}</b></span>
      <span>Max DD <b>{P['dd']:.1f}%</b></span><span>t <b>{P['t_stat']:.2f}</b></span>
      <span><b>{S['unique_stocks']}</b> names</span><span><b>{S['unique_sectors_picked']}</b> sectors</span>
      <span><b>{S['delisted_picks']}</b> acquired/delisted holds</span>
      <span>survivorship-free · point-in-time · no fees</span>
    </div>
  </header>

  <nav class="nav" aria-label="Sections">
    <a href="#curve">Equity Curve</a><a href="#years">By Year</a><a href="#how">How It Picks</a>
    <a href="#stocks">Leaderboard</a><a href="#sectors">Sectors</a><a href="#best">Best &amp; Worst</a><a href="#blotter">Full Blotter</a>
  </nav>

  <div class="kpis">
    {"".join(f'<div class="kpi{" hero" if i in (0,1) else ""}"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="d">{esc(dd)}</div></div>' for i,(k,v,dd) in enumerate(kpis))}
  </div>
  <div class="funda">
    {"".join(f'<div class="fu"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="d">{esc(dd)}</div></div>' for k,v,dd in funda)}
  </div>
  <p class="chart-note" style="margin-top:8px">Fundamental medians are measured <b>at the moment of each purchase</b> (point-in-time, 45-day report lag) — not current values.</p>

  <section id="curve">
    <div class="shead"><h2>Growth of $100,000</h2><span class="cnt">flagship vs S&amp;P 500 · {S['months']} months</span></div>
    <div class="grid2">
      <div class="chartcard">
        <canvas id="eq" aria-label="Equity curve: flagship vs S&amp;P 500"></canvas>
        <div class="legend">
          <span><i class="fl"></i>usca_small — ${S['final_100k_flagship']:,}</span>
          <span><i class="sp"></i>S&amp;P 500 — ${S['final_100k_spy']:,}</span>
        </div>
        <p class="chart-note">Best month {bm['date']} {pct(bm['ret'])} · worst {wm['date']} {pct(wm['ret'])}. Monthly rebalance, equal-weight, 2× tilt on A/D-conviction names.</p>
      </div>
      <div class="tablecard calcard" style="margin-top:0" id="years">
        <div class="scrollx"><table class="sortable">
          <thead><tr><th class="l">Year</th><th>Strategy</th><th>S&amp;P 500</th><th>Excess</th><th>Mo</th></tr></thead>
          <tbody>{"".join(cal_rows)}</tbody>
        </table></div>
      </div>
    </div>
  </section>

  <section id="how">
    <div class="shead"><h2>How a pick is made</h2></div>
    <p class="lede">Each month the engine ranks ~149 sector ETFs by momentum <b>acceleration</b> (3-month change minus the prior 3-month change) and keeps the <b>top 10</b>. Inside each, it broadens to every US/Canada stock in that GICS sector, screens for positive book value and $5M+ daily liquidity, and buys the <b>cheapest price-to-book small-cap</b> (&lt;$2B). Names accumulating into weakness (A/D divergence) get double weight. Sectors with no qualifying value stock — raw commodities, bonds, foreign index sleeves — are <b>skipped</b>. Every fundamental below is the value <b>as filed and public on the purchase date</b>.</p>
  </section>

  <section id="stocks">
    <div class="shead"><h2>Stock leaderboard</h2><span class="cnt">{S['unique_stocks']} names · P/E · ROE · D/E at purchase</span></div>
    <p class="lede">Every distinct name ever held, ordered by summed weighted return contribution. Multiples are the <b>median at the times it was bought</b> (point-in-time). <b>†</b> = acquired/delisted during a hold. A handful of names carry the book.</p>
    <p class="lede2">Click any column header to sort. Hover the <span class="hint" style="cursor:default">?</span> on each header for its definition.</p>
    <div class="tablecard"><div class="scrolly"><table class="sortable">
      <thead><tr>
        {th("Ticker","l")}{th("Company","l")}{th("Sector(s)","l")}{th("Held")}{th("Win%")}
        {th("P/B")}{th("P/E")}{th("ROE")}{th("GP/A")}{th("Rev&nbsp;g")}{th("D/E")}{th("Mkt&nbsp;cap")}{th("Tier")}
        {th("Avg&nbsp;ret")}{th("Best")}{th("Worst")}{th("Contrib.")}{th("Span")}
      </tr></thead>
      <tbody>{"".join(stock_rows)}</tbody>
    </table></div></div>
  </section>

  <section id="sectors">
    <div class="shead"><h2>Sectors</h2><span class="cnt">{len(D['by_sector'])} sectors touched the top-10</span></div>
    <p class="lede">Which sleeves accelerated into the top-10, how often the engine found a value pick there, and how often it skipped (no qualifying stock). Sorted by picks.</p>
    <div class="tablecard"><div class="scrolly"><table class="sortable">
      <thead><tr>{th("Sector","l")}{th("ETF","l")}{th("In&nbsp;top-10")}{th("Picked&nbsp;from")}{th("Skipped")}{th("Avg&nbsp;pick&nbsp;ret")}</tr></thead>
      <tbody>{"".join(sector_rows)}</tbody>
    </table></div></div>
  </section>

  <section id="best">
    <div class="shead"><h2>Best &amp; worst single months</h2><span class="cnt">top 20 each · fundamentals at buy</span></div>
    <div class="tablecard">
      <div class="scrollx"><table class="sortable">
        <thead><tr><th class="l" colspan="8" style="color:var(--pos)">▲ Best monthly picks</th></tr>
        <tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("P/B")}{th("P/E")}{th("ROE")}{th("Return")}</tr></thead>
        <tbody>{pick_rows(D['best_picks'])}</tbody>
      </table></div>
    </div>
    <div class="tablecard">
      <div class="scrollx"><table class="sortable">
        <thead><tr><th class="l" colspan="8" style="color:var(--neg)">▼ Worst monthly picks</th></tr>
        <tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("P/B")}{th("P/E")}{th("ROE")}{th("Return")}</tr></thead>
        <tbody>{pick_rows(D['worst_picks'])}</tbody>
      </table></div>
    </div>
  </section>

  <section id="blotter">
    <div class="shead"><h2>Full blotter</h2><span class="cnt">{S['total_picks']} picks across {S['months']} months</span></div>
    <p class="lede">The complete month-by-month record. Each month opens with its rebalance date and basket return vs SPY, then a <b>sector-ranking strip</b> — the ten sleeves that accelerated into the momentum top-10 that month, each showing its acceleration and either the value name it resolved to (<span class="chip pick" style="padding:0 5px">→ ticker</span>) or <b>why the slot was skipped</b> (<span class="chip skip" style="padding:0 5px">⊘</span>, hover for the reason). Beneath are the holdings (best-to-worst) with their <b>point-in-time</b> fundamentals. Hover any pick row for the one-line rationale; <b>2×</b> = A/D-conviction weight.</p>
    <div class="tablecard"><div class="scrolly"><table>
      <thead><tr>{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("Tier")}{th("P/B")}{th("P/E")}{th("ROE")}{th("GP/A")}{th("Rev&nbsp;g")}{th("D/E")}{th("Mkt&nbsp;cap")}{th("Weight")}{th("Return")}</tr></thead>
      <tbody>{"".join(blotter)}</tbody>
    </table></div></div>
  </section>

  <footer class="foot">
    <p><b>Fundamentals are point-in-time.</b> Every P/B, P/E, ROE and D/E shown is the value as filed and public on the purchase date (45-day report lag) — the price component is the purchase-month close, never a later close or a current value. <b>There is deliberately no forward P/E:</b> our data providers expose only today's forward estimate, and pasting that onto a 2022 buy would be look-ahead. <b>P/E is the standard trailing multiple</b> — Price ÷ trailing-12-month EPS (market cap ÷ TTM net income, four quarters summed point-in-time); a negative P/E means the company was loss-making over the trailing year, and blank means fewer than four quarters had been filed by the trade date. ROE is likewise trailing-12-month.</p>
    <p><b>Method.</b> usca_small arm of the survivorship-free small-cap study. Universe: current sector-ETF membership (survivors) plus 1,973 major-exchange delisted names mapped by GICS, held during their live window. Returns are USD-translated (include FX P&amp;L on foreign lines), monthly rebalance, no transaction costs or slippage. Acquired/delisted names (†) exit at last/deal price via the delisting-exit mechanism — an M&amp;A announcement-exit overlay is a separate study.</p>
    <p><b>Caveats.</b> The GICS-mapped universe differs from the live ETF-membership universe, so absolutes are internally consistent but not identical to the live scanner. Delisted coverage misses pre-2020 deaths and OTC names. Simulated past performance is not a forecast. Computed {esc(D.get('computed_at',''))[:19]}.</p>
  </footer>

</div>
</div>

<div class="pane" id="pane" hidden>
  <div class="pane-bd" id="paneBd"></div>
  <aside class="pane-card" role="dialog" aria-label="Stock trade history">
    <button class="pane-x" id="paneX" type="button" aria-label="Close">×</button>
    <div id="paneBody"></div>
  </aside>
</div>
<button class="tog" id="tog" type="button" aria-label="Toggle light or dark theme">◐ theme</button>
<script>
(function(){{
  var data = {curve_js};
  var root=document.documentElement, tog=document.getElementById('tog');
  tog.addEventListener('click',function(){{
    var cur=root.getAttribute('data-theme');
    if(!cur) cur = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark':'light';
    root.setAttribute('data-theme', cur==='dark'?'light':'dark'); draw();
  }});
  var cv=document.getElementById('eq'), ctx=cv.getContext('2d');
  function css(v){{return getComputedStyle(root).getPropertyValue(v).trim();}}
  function hexA(hex,a){{ hex=hex.replace('#',''); if(hex.length===3) hex=hex.split('').map(function(c){{return c+c;}}).join('');
    var r=parseInt(hex.slice(0,2),16),g=parseInt(hex.slice(2,4),16),b=parseInt(hex.slice(4,6),16); return 'rgba('+r+','+g+','+b+','+a+')'; }}
  function draw(){{
    var dpr=Math.max(1,window.devicePixelRatio||1), w=cv.clientWidth, h=cv.clientHeight;
    cv.width=w*dpr; cv.height=h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
    var padL=64,padR=16,padT=14,padB=26;
    var gold=css('--gold'), mut=css('--mut'), line=css('--line');
    var maxV=0; data.forEach(function(d){{maxV=Math.max(maxV,d.f,d.s);}}); var minV=100000;
    function X(i){{return padL+(w-padL-padR)*(i/(data.length-1));}}
    function Y(v){{return padT+(h-padT-padB)*(1-(v-minV)/(maxV-minV));}}
    ctx.font='11px "SF Mono",Consolas,monospace'; ctx.textBaseline='middle';
    var step= maxV>800000?200000:100000; ctx.strokeStyle=line; ctx.fillStyle=mut; ctx.lineWidth=1;
    for(var g=minV; g<=maxV; g+=step){{ var y=Y(g); ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
      ctx.textAlign='right'; ctx.fillText('$'+(g/1000)+'k', padL-8, y); }}
    ctx.textAlign='center'; var lastYr='';
    data.forEach(function(d,i){{ var yr=d.d.slice(0,4); if(yr!==lastYr){{lastYr=yr; ctx.fillText(yr, X(i), h-8);}} }});
    var grad=ctx.createLinearGradient(0,padT,0,h-padB); grad.addColorStop(0, hexA(gold,.22)); grad.addColorStop(1, hexA(gold,0));
    ctx.beginPath(); data.forEach(function(d,i){{var x=X(i),y=Y(d.f); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}});
    ctx.lineTo(X(data.length-1),Y(minV)); ctx.lineTo(X(0),Y(minV)); ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
    function plot(key,color,wd){{ ctx.beginPath(); ctx.lineWidth=wd; ctx.strokeStyle=color; ctx.lineJoin='round';
      data.forEach(function(d,i){{ var x=X(i),y=Y(d[key]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }}); ctx.stroke(); }}
    plot('s',mut,1.6); plot('f',gold,2.4);
    var last=data[data.length-1]; ctx.fillStyle=gold; ctx.beginPath(); ctx.arc(X(data.length-1),Y(last.f),4,0,7); ctx.fill();
  }}
  draw(); window.addEventListener('resize',draw);

  // ---- click-to-sort on tables marked .sortable ----
  var TIER={{micro:0,small:1,mid:2,large:3}};
  function cellVal(td){{
    var t=(td.textContent||'').trim(), tl=t.toLowerCase();
    if(tl in TIER) return TIER[tl];
    if(t===''||t==='—'||t==='n/a') return -Infinity;
    var s=t.replace(/[$,×%\s]/g,'').replace(/[–—]/g,'');
    var m=s.match(/^(-?\d*\.?\d+)([BMK])?$/i);
    if(m){{var v=parseFloat(m[1]),u=(m[2]||'').toUpperCase(); if(u==='B')v*=1e9; else if(u==='M')v*=1e6; else if(u==='K')v*=1e3; return v;}}
    return tl;
  }}
  function makeSortable(tbl){{
    if(!tbl.tHead||!tbl.tBodies.length) return;
    var ths=tbl.tHead.rows[tbl.tHead.rows.length-1].cells, tb=tbl.tBodies[0];
    Array.prototype.forEach.call(ths,function(th,idx){{
      th.classList.add('sortth');
      th.addEventListener('click',function(e){{
        if(e.target.classList && e.target.classList.contains('hint')) return;
        var rows=Array.prototype.slice.call(tb.rows);
        var asc=th.getAttribute('data-dir')!=='asc';
        rows.sort(function(a,b){{ var x=cellVal(a.cells[idx]), y=cellVal(b.cells[idx]);
          if(x<y) return asc?-1:1; if(x>y) return asc?1:-1; return 0; }});
        rows.forEach(function(r){{tb.appendChild(r);}});
        Array.prototype.forEach.call(ths,function(o){{ o.removeAttribute('data-dir'); var a=o.querySelector('.arrow'); if(a)a.remove(); }});
        th.setAttribute('data-dir',asc?'asc':'desc');
        var ar=document.createElement('span'); ar.className='arrow'; ar.textContent=asc?' ▲':' ▼'; th.appendChild(ar);
      }});
    }});
  }}
  document.querySelectorAll('table.sortable').forEach(makeSortable);

  // ---- click a leaderboard row -> slide-in pane with that name's full trade history ----
  var TRADES={stock_trades_js};
  var pane=document.getElementById('pane'), paneBody=document.getElementById('paneBody');
  function fp(v){{ if(v==null) return '<span class="mut">—</span>'; var s=v*100; return '<span class="'+(s>=0?'pos':'neg')+'">'+(s>=0?'+':'')+s.toFixed(1)+'%</span>'; }}
  function fn(v,d){{ return v==null?'<span class="mut">—</span>':v.toFixed(d==null?2:d); }}
  function fpe(v){{ if(v==null) return '<span class="mut">—</span>'; return '<span class="'+(v<0?'neg':'')+'">'+v.toFixed(1)+'</span>'; }}
  function froe(v){{ if(v==null) return '<span class="mut">—</span>'; return '<span class="'+(v>=0?'pos':'neg')+'">'+(v*100).toFixed(0)+'%</span>'; }}
  function fc(v){{ if(v==null) return '<span class="mut">—</span>'; if(v>=1e9) return '$'+(v/1e9).toFixed(1)+'B'; if(v>=1e6) return '$'+(v/1e6).toFixed(0)+'M'; return '$'+Math.round(v); }}
  function openPane(tk){{
    var d=TRADES[tk]; if(!d) return;
    var wins=d.t.filter(function(t){{return t.r>0;}}).length, tot=d.t.length;
    var rows=d.t.map(function(t){{
      var cv=t.cv?' <span class="conv">2×</span>':'';
      return '<tr><td class="l mono">'+t.d+'</td><td class="l sec">'+t.s+'</td><td class="num">'+fn(t.pb)+'</td><td class="num">'+fpe(t.pe)+'</td><td class="num">'+froe(t.roe)+'</td><td class="num">'+fn(t.gpa)+'</td><td class="num">'+fp(t.rg)+'</td><td class="num">'+fc(t.mc)+'</td><td class="num">'+(t.w?t.w.toFixed(0)+'×':'1×')+cv+'</td><td class="num strong">'+fp(t.r)+'</td></tr>';
    }}).join('');
    paneBody.innerHTML='<div class="pane-h"><span class="tk big">'+tk+(d.dl?' <span class="dl">†</span>':'')+'</span> <span class="pane-co">'+(d.co||'')+'</span></div>'+
      '<div class="pane-sub">'+tot+' trade'+(tot!==1?'s':'')+' · '+Math.round(100*wins/tot)+'% positive · point-in-time fundamentals at each buy</div>'+
      '<div class="scrollx"><table class="ptab"><thead><tr><th class="l">Month</th><th class="l">Sector sleeve</th><th>P/B</th><th>P/E</th><th>ROE</th><th>GP/A</th><th>Rev&nbsp;g</th><th>Mkt&nbsp;cap</th><th>Wt</th><th>Return</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
    pane.hidden=false; document.body.style.overflow='hidden';
  }}
  function closePane(){{ pane.hidden=true; document.body.style.overflow=''; }}
  document.getElementById('paneX').addEventListener('click',closePane);
  document.getElementById('paneBd').addEventListener('click',closePane);
  document.addEventListener('keydown',function(e){{ if(e.key==='Escape' && !pane.hidden) closePane(); }});
  document.querySelectorAll('tr.clk').forEach(function(r){{ r.addEventListener('click',function(){{ openPane(r.getAttribute('data-ticker')); }}); }});
}})();
</script>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML):,} bytes)")
