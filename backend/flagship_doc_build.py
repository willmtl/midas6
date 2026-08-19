# -*- coding: utf-8 -*-
"""Render the flagship full-history document (self-contained HTML) from the enriched trace.
ALL per-purchase fundamentals shown are POINT-IN-TIME as of the trade date (45-day report lag). There is
deliberately NO forward P/E: our providers only expose today's estimate, which would be look-ahead.
Reads  /app/.data/studies/flagship_history_enriched.json
Writes /app/.data/studies/flagship_history.html
"""
import json, html
from pathlib import Path

import os as _os
_CK = _os.environ.get("CONFIG", "adaptive")
_SUF = "" if _CK == "adaptive" else f"_{_CK}"
IN = Path(f"/app/.data/studies/flagship_history{_SUF}_enriched.json")
OUT = Path(f"/app/.data/studies/flagship_history{_SUF}.html" if _CK != "adaptive" else "/app/.data/studies/flagship_history.html")
D = json.load(open(IN))
S, P = D["summary"], D["perf"]
# filename each config's doc links to (for the subtabs) — adaptive is the main flagship_history.html
_CFG_FILE = {"adaptive": "flagship_history.html", "core": "flagship_history_core.html",
             "middle": "flagship_history_middle.html", "aggressive": "flagship_history_aggressive.html"}

# the regime-bet configs (core / middle / adaptive / aggressive) for the "choose your config" section + tabs
try:
    _cc = json.load(open("/app/.data/studies/configs_compare.json"))
    CONFIGS = _cc.get("configs", _cc) if isinstance(_cc, dict) else _cc
    SPY_CURVE = _cc.get("spy_curve", []) if isinstance(_cc, dict) else []
except Exception:
    CONFIGS, SPY_CURVE = [], []

# ── METHOD-TABS data: pull the Finviz engine result + ETF arms so the doc can show a tab per method.
# Optional/defensive — if the DB isn't reachable the Finviz tab just shows a "pending" note.
FINVIZ = None
ETF_ARMS = {}
try:
    import os as _os, django as _dj
    _os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rotation.settings")
    _dj.setup()
    from core.models import BacktestResult as _BR
    _fr = _BR.objects.filter(kind="finviz_rotation").order_by("-computed_at").first()
    if _fr:
        FINVIZ = _fr.payload
    _sr = _BR.objects.filter(kind="survivorship_smallcap").order_by("-computed_at").first()
    if _sr:
        _res = _sr.payload.get("results", {})
        for _k in ("usca_small", "usca_small_upside_pb", "usca_small_bear_fcf", "usca_small_proxy"):
            if isinstance(_res.get(_k), dict):
                ETF_ARMS[_k] = _res[_k]
except Exception as _e:
    print("method-compare data unavailable:", _e)


def _build_finviz_pane():
    """The Finviz-version tab — the SAME sections as the ETF flagship (KPIs, head-to-head, by-year,
    leaderboard, industries, best/worst, full blotter), built from the engine's detailed trace."""
    if not FINVIZ:
        return ('<div class="mpane" id="mpane-finviz" hidden><header class="mast">'
                '<h1>Finviz Industry Rotation</h1><p class="sub">Pending — run finviz_rotation_study.py.</p>'
                '</header></div>')
    f = FINVIZ["full"]
    u = FINVIZ.get("universe", {})
    sec = FINVIZ.get("sections", {}) or {}
    tmonths = FINVIZ.get("tmonths", [])

    # ── head-to-head vs the ETF arms ──
    _armlab = [("usca_small_upside_pb", "ETF · analyst-upside × P/B blend (live default)"),
               ("usca_small", "ETF · raw cheapest-P/B (apples-to-apples)"),
               ("usca_small_bear_fcf", "ETF · blend + 200-MA regime (demoted — dropped from production)"),
               ("usca_small_proxy", "ETF · buy-the-skipped-ETF (refuted)")]
    rows = ""
    for k, lab in _armlab:
        v = ETF_ARMS.get(k)
        if v:
            rows += (f'<tr><td class="l sec" style="max-width:none">{lab}</td>'
                     f'<td class="num">{v.get("total", 0):,.0f}%</td><td class="num">{v.get("sharpe", "—")}</td></tr>')
    rows += (f'<tr style="background:color-mix(in srgb,var(--gold) 12%,transparent)">'
             f'<td class="l sec strong" style="max-width:none">Finviz · industry rotation v2 ($300M floor + blend)</td>'
             f'<td class="num strong">{f["total"]:,.0f}%</td><td class="num strong">{f["sharpe"]}</td></tr>')

    fk = round(sec.get("final_100k_flagship", 0))
    sk = round(sec.get("final_100k_spy", 0))
    kpis = [("Total return", f'{f["total"]:,.0f}%', f'$100k → ${fk:,}'),
            ("Sharpe", f'{f["sharpe"]}', f'{f.get("months","?")} months'),
            ("Max drawdown", f'{f.get("dd","—")}%', "monthly"),
            ("Industries used", f'{u.get("industries", "—")}', "of 149 (≥3 members)"),
            ("Names traded", f'{sec.get("n_stocks","—")}', f'{u.get("with_fundamentals","—")} w/ fundamentals')]
    kpi_html = "".join(kpi_tile(k, v, dd, hero=(i == 0)) for i, (k, v, dd) in enumerate(kpis))

    # ── by-year calendar ──
    cal_rows = "".join(
        f'<tr><td class="l mono">{c["year"]}</td><td class="num">{pctp(c["strategy"])}</td>'
        f'<td class="num">{pctp(c["spy"])}</td><td class="num">{pctp(c["excess"])}</td>'
        f'<td class="num mut">{c["months"]}</td></tr>' for c in sec.get("calendar", []))

    # ── leaderboard (top stocks by contribution) ──
    lb_rows = "".join(
        f'<tr><td class="l"><span class="tk">{esc(x["ticker"])}</span></td>'
        f'<td class="l nm">{esc(x["company"])}</td><td class="l sec">{esc(x["industry"])}</td>'
        f'<td class="num">{x["held"]}</td><td class="num">{winfmt(x["win"])}</td>'
        f'<td class="num">{pctp(x["avg"])}</td><td class="num strong">{pctp(x["contrib"])}</td></tr>'
        for x in sec.get("leaderboard", [])[:60])

    # ── industries ──
    ind_rows = "".join(
        f'<tr><td class="l nm" style="max-width:none">{esc(x["industry"])}</td>'
        f'<td class="num">{x["in_top"]}</td><td class="num">{x["picked"]}</td>'
        f'<td class="num">{x["skipped"] or "<span class=mut>0</span>"}</td>'
        f'<td class="num">{pctp(x["avg"]) if x["avg"] is not None else "<span class=mut>—</span>"}</td></tr>'
        for x in sec.get("industries", []))

    def bw_rows(lst):
        return "".join(
            f'<tr><td class="l mono">{esc(p["date"])}</td><td class="l"><span class="tk">{esc(p["ticker"])}</span></td>'
            f'<td class="l nm">{esc(p.get("company",""))}</td><td class="l sec">{esc(p["industry"])}</td>'
            f'<td class="num">{pbfmt(p.get("pb"))}</td><td class="num">{roefmt(p.get("roe"))}</td>'
            f'<td class="num strong">{pctp(round((p["ret"] or 0)*100,1))}</td></tr>' for p in lst)

    # ── full blotter (industry ranking strip + holdings) ──
    def fv_chips(m):
        pick_by = {p["industry"]: p for p in m["picks"]}
        skip_ind = {s["industry"] for s in m["skipped"]}
        out = []
        for i, ts in enumerate(m["top_industries"], 1):
            nm, acc, ir = ts["industry"], ts.get("accel"), ts.get("ind_ret")
            head = (f'<span class="rk">#{i}</span><b>{esc(nm)}</b> '
                    f'<span class="acc" title="acceleration — the ranking signal, NOT a return">accel '
                    f'{acc*100:+.1f}%</span>' if acc is not None else f'<span class="rk">#{i}</span><b>{esc(nm)}</b>')
            slv = "" if ir is None else f' <span class="slv">industry <span class="{"pos" if ir>=0 else "neg"}">{ir*100:+.1f}%</span></span>'
            if nm in pick_by:
                p = pick_by[nm]
                _dv = ""
                if p["ret"] is not None and ir is not None:
                    _d = (p["ret"] - ir) * 100
                    _dv = (f' <span class="dvs" title="Stock-selection edge = the stock we bought MINUS what the industry index alone returned.">'
                           f'(<span class="{"pos" if _d>=0 else "neg"}">{_d:+.1f}pp</span> vs industry)</span>')
                out.append(f'<span class="chip pick" title="Ranked #{i} because the industry accelerated (accel is the SELECTION signal, not a return). '
                           f'Inside it we bought {esc(p["ticker"])} at P/B {p.get("pb")} on the {esc(m["date"])} close and sold on the {esc(m["ndate"])} close — that pick returned {(p["ret"] or 0)*100:+.1f}% (can differ from the industry; accel ≠ outcome).">'
                           f'{head}{slv} → bought <span class="tk">{esc(p["ticker"])}</span> {pct((p["ret"] or 0))}{_dv}</span>')
            elif nm in skip_ind:
                out.append(f'<span class="chip skip" title="no qualifying small-cap value name; industry ETF-equivalent returned this month — not held">'
                           f'{head} <span class="xm">⊘ skipped</span>{slv}</span>')
            else:
                out.append(f'<span class="chip">{head}{slv}</span>')
        return "".join(out)

    blot = []
    for m in tmonths:
        npos = len(m["picks"]); nskip = len(m["skipped"])
        exc = ((m["basket_ret"] or 0) - (m["spy_ret"] or 0)) * 100
        blot.append(
            f'<tr class="mhdr"><td class="l mono" colspan="3"><b>bought {esc(m["date"])} close → sold {esc(m["ndate"])} close</b> · {npos} holding{"s" if npos!=1 else ""} · {nskip} skipped</td>'
            f'<td class="num" colspan="2">SPY {pct(m["spy_ret"])}</td>'
            f'<td class="num" colspan="2">basket <b>{pct(m["basket_ret"])}</b> <span class="{"pos" if exc>=0 else "neg"}">({exc:+.1f})</span></td></tr>'
            f'<tr class="srow"><td colspan="7"><div class="chips">{fv_chips(m)}</div></td></tr>')
        for p in sorted(m["picks"], key=lambda x: -(x["ret"] or -9)):
            blot.append(
                f'<tr><td class="l"><span class="tk">{esc(p["ticker"])}</span></td><td class="l nm">{esc(p.get("company",""))}</td>'
                f'<td class="l sec">{esc(p["industry"])}</td><td class="num">{pbfmt(p.get("pb"))}</td>'
                f'<td class="num">{roefmt(p.get("roe"))}</td><td class="num">{cap(p.get("mktcap"))}</td>'
                f'<td class="num">{pct((p["ret"] or 0))}</td></tr>')

    sweep = FINVIZ.get("tightening_sweep") or []
    srows = "".join(
        f'<tr style="{"font-weight:700;background:color-mix(in srgb,var(--gold) 12%,transparent)" if s["variant"].startswith("size floor $300M") else ""}">'
        f'<td class="l sec" style="max-width:none">{esc(s["variant"])}</td><td class="num">{s["full"]:,.0f}%</td>'
        f'<td class="num">{s["h1"]:,.0f}%</td><td class="num">{s["h2"]:,.0f}%</td><td class="num">{s["sharpe"]}</td></tr>'
        for s in sweep)

    return f"""
  <div class="mpane" id="mpane-finviz" hidden>
    <header class="mast">
      <p class="eyebrow">Finviz Version · Industry Rotation <span class="statuspill">v2</span></p>
      <h1>Rotating the Finviz Industry Map</h1>
      <p class="sub">A separate engine that ranks all <b>149 Finviz industries</b> ({u.get("with_fundamentals","?")} names with fundamentals) by momentum <b>acceleration</b>{hint("accel")}, then buys the cheapest-P/B × analyst-upside <b>blend</b>{hint("blend")} small-cap inside the hottest ones. Shares only the point-in-time data library with the ETF version — never its logic.</p>
      <div class="headline"><div class="bignum">+{f["total"]:,.0f}%</div>
        <div class="bigsub"><span class="a">$100k → ${fk:,}</span><span class="b">{f.get("months","?")} months · Sharpe {f["sharpe"]} · vs SPY ${sk:,}</span></div>
        <span class="codechip">v2 · $300M floor + blend</span></div>
    </header>
    <div class="banner"><b>Honest read.</b> Full breadth realized ({sec.get("n_stocks","?")} names traded across {u.get("industries","?")} industries). It beats SPY on a totally separate universe — real cross-validation — but at ~1/10th the ETF flagship's return: the ETF's edge is its <b>curated thematic sleeves</b> (Gold/Clean-Energy/Uranium), which the generic GICS industry map fragments. Still survivorship-optimistic (no delisted handling yet).</div>
    <div class="kpis" style="grid-template-columns:repeat({len(kpis)},1fr)">{kpi_html}</div>

    <section><div class="shead"><h2>Head-to-head vs the ETF flagship</h2><span class="cnt">total · Sharpe</span></div>
      <p class="lede">Same rules (accel → cheap-value small-cap, monthly); the only difference is the <b>rotation map</b> (91 curated ETFs vs 149 Finviz industries) and, for the top rows, the ETF's tuned levers.</p>
      <div class="tablecard"><div class="scrollx"><table><thead><tr><th class="l">Method</th><th>Total</th><th>Sharpe</th></tr></thead><tbody>{rows}</tbody></table></div></div></section>

    <section><div class="shead"><h2>By year</h2><span class="cnt">Finviz vs S&amp;P 500</span></div>
      <div class="tablecard"><div class="scrollx"><table class="sortable"><thead><tr><th class="l">Year</th><th>Finviz</th><th>S&amp;P 500</th><th>Excess</th><th>Mo</th></tr></thead><tbody>{cal_rows}</tbody></table></div></div></section>

    <section><div class="shead"><h2>Rebuilding the ETF’s implicit filter</h2><span class="cnt">walk-forward: FULL / H1 / H2</span></div>
      <p class="lede">The ETF gets quality screening free (index inclusion drops nano-caps/shells/illiquid junk). Finviz has no gatekeeper, so we rebuild it: the winner is a <b>$300M size floor</b> (nano-caps were the value traps); profit + liquidity gates <b>hurt</b>.</p>
      <div class="tablecard"><div class="scrollx"><table><thead><tr><th class="l">Selection rule</th><th>FULL</th><th>H1</th><th>H2</th><th>Sharpe</th></tr></thead><tbody>{srows}</tbody></table></div></div></section>

    <section><div class="shead"><h2>Stock leaderboard</h2><span class="cnt">{sec.get("n_stocks","?")} names · by contribution</span></div>
      <div class="tablecard"><div class="scrolly"><table class="sortable"><thead><tr>{th("Ticker","l")}{th("Company","l")}{th("Industry","l")}{th("Held")}{th("Win%")}{th("Avg&nbsp;ret")}{th("Contrib.")}</tr></thead><tbody>{lb_rows}</tbody></table></div></div></section>

    <section><div class="shead"><h2>Industries</h2><span class="cnt">ranked into the top-10</span></div>
      <div class="tablecard"><div class="scrolly"><table class="sortable"><thead><tr>{th("Industry","l")}{th("In&nbsp;top-10")}{th("Picked")}{th("Skipped")}{th("Avg&nbsp;pick&nbsp;ret")}</tr></thead><tbody>{ind_rows}</tbody></table></div></div></section>

    <section><div class="shead"><h2>Best &amp; worst single months</h2></div>
      <div class="tablecard"><div class="scrollx"><table class="sortable"><thead><tr><th class="l" colspan="7" style="color:var(--pos)">▲ Best picks</th></tr><tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Industry","l")}{th("P/B")}{th("ROE")}{th("Return")}</tr></thead><tbody>{bw_rows(sec.get("best",[]))}</tbody></table></div></div>
      <div class="tablecard"><div class="scrollx"><table class="sortable"><thead><tr><th class="l" colspan="7" style="color:var(--neg)">▼ Worst picks</th></tr><tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Industry","l")}{th("P/B")}{th("ROE")}{th("Return")}</tr></thead><tbody>{bw_rows(sec.get("worst",[]))}</tbody></table></div></div></section>

    <section><div class="shead"><h2>Full blotter</h2><span class="cnt">{len(tmonths)} months</span></div>
      <p class="lede">Bought at each month-end close, sold at the next. Each chip carries up to <b>three different numbers</b>: <b>accel ±x%</b> = the industry's momentum acceleration (the <i>ranking signal</i> — why it's in the top-10, <b>NOT a return</b>); <b>industry ±x%</b> = what that industry's equal-weight index actually returned that month; and <b>bought TICKER ±x%</b> = what the specific small-cap we bought returned. A high accel does <b>not</b> mean the pick went up — e.g. an industry can accelerate +95% and the pick still fall (acceleration is trailing; hot themes mean-revert). Over many months the selection wins on average; any single month is noisy.</p>
      <div class="tablecard"><div class="scrolly"><table><thead><tr>{th("Ticker","l")}{th("Company","l")}{th("Industry","l")}{th("P/B")}{th("ROE")}{th("Mkt&nbsp;cap")}{th("Return")}</tr></thead><tbody>{"".join(blot)}</tbody></table></div></div></section>

    <footer class="foot"><p><b>Finviz engine.</b> <span class="mono">finviz_rotation_study.py</span> + <span class="mono">finviz_config.py</span>, fed by <span class="mono">scrape_finviz_universe.py</span>. Industry momentum = equal-weight constituent index; pick = $300M-floor small-cap ranked 60% analyst-upside + 40% cheap-P/B, monthly. Survivorship-optimistic (no delisted exits yet), FX unadjusted. Shares only the data/PIT library with the ETF flagship.</p></footer>
  </div>
"""


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
    "Held to today": "Buy-and-hold return if you had bought this name at its FIRST purchase month and never sold — first-purchase close to the latest close (split/div-adjusted). Shows whether the monthly rotation left money on the table (name kept running) or dodged a later collapse. Delisted names run to their last traded price.",
    "Weight": "Basket weight: 2× when the name shows A/D accumulation into price weakness (conviction), else 1×.",
    "In top-10": "How many months this sleeve's ETF accelerated into the momentum top-10.",
    "Picked from": "How many of those top-10 months the engine found a qualifying value pick in the sleeve.",
    "Skipped": "Months the sleeve was in the top-10 but had no qualifying value stock (foreign/commodity/bond) — the slot was skipped, held as cash-equivalent.",
    "Avg pick ret": "Average realized next-month return of picks drawn from this sleeve.",
    "Month": "Rebalance date the pick was bought.",
    "Return": "Realized next-month return of the pick (purchase-month close to next rebalance).",
    # ── Finviz-tab headers ──
    "Industry": "Finviz industry (of 149) whose equal-weight constituent index accelerated into the momentum top-10 when this name was picked.",
    "Method": "The rotation engine / arm being compared head-to-head.",
    "Total": "Total COMPOUNDED return over the full backtest window (not annualized). Leans on a few winners + 2020-21; read as a range, not a point.",
    "Sharpe": "Annualized Sharpe = mean monthly return ÷ its standard deviation × √12. Risk-adjusted return; higher = smoother ride.",
    "Year": "Calendar year.",
    "Finviz": "The Finviz industry-rotation engine's compounded return that year.",
    "S&P 500": "SPY total return over the same rebalance months.",
    "Strategy": "The flagship's compounded return that year.",
    "Excess": "Strategy return minus SPY that year — out- (green) or under- (red) performance.",
    "Mo": "Number of monthly rebalances that year.",
    "Selection rule": "The stock-selection filter/lever tested inside the top-accel industries (the ETF's 'implicit filter' rebuilt explicitly).",
    "FULL": "Full backtest window (all months).",
    "H1": "First walk-forward half (through 2022-12).",
    "H2": "Second walk-forward half (2023-01 →). A lever must hold in BOTH halves to be trusted, not just full-sample.",
    "In&nbsp;top-10": "How many months this sleeve/industry ranked into the momentum top-10 by acceleration.",
    "Avg&nbsp;ret": "Average realized next-month return across every month this name was held.",
    "Avg&nbsp;pick&nbsp;ret": "Average realized next-month return of picks drawn from this sleeve/industry.",
    # ── KPI tiles ──
    "Total return": "Total compounded return over the full window ($100k → final value). Not annualized; not a forward forecast.",
    "Max drawdown": "Largest peak-to-trough drop of the monthly equity curve. Less negative = shallower worst loss.",
    "Max DD": "Largest peak-to-trough drop of the monthly equity curve.",
    "Industries used": "How many of the 149 Finviz industries had ≥3 members with price history, so a stable momentum index could be built.",
    "Names traded": "Distinct stocks bought at least once across the backtest.",
    "Names w/ fundamentals": "Universe names that have point-in-time quarterly financials (needed to compute P/B), of the US/CA total.",
    "CAGR": "Compound annual growth rate — the constant yearly rate that compounds to the total return.",
    # ── signals / terms (inline ?) ──
    "accel": "ACCELERATION — the ranking signal, NOT a return or forecast. This period's 3-month momentum minus the prior 3-month's; high = money is rushing in right now. It's why a sleeve is in the top-10; it does not predict the pick's next-month return.",
    "blend": "The selection rule: rank the qualifying small-caps 60% by analyst implied-upside (mean price target ÷ price − 1) + 40% by cheap P/B; pick the best combined rank.",
    "regime": "SPY vs its own 200-day moving average. In the DEMOTED 'blend + regime' variant, a bear regime (SPY below 200-MA) tilts picks toward high-FCF-margin names. Dropped from production (didn't cross-validate).",
    "conviction": "2× basket weight when the name shows Accumulation/Distribution accumulation into price weakness — smart money buying the dip.",
    "deactivated": "A retired sleeve (ARKK/QTUM/SPLV/Nuclear/Grid) whose acceleration we still calculate and display for monitoring, but never trade.",
    "sleeve": "The sector ETF (ETF tab) or equal-weight industry index (Finviz tab) whose acceleration defines a rotatable unit.",
    "skipped": "The sleeve was in the top-10 but had no qualifying small-cap value stock (raw commodity/bond/foreign, or nothing cleared the filters) — the slot is passed and its weight redistributes to the other picks.",
}


def th(label, cls=""):
    """Header cell with a hover/focus '?' tooltip pulled from TIP (keyed by the plain label)."""
    tip = TIP.get(label.replace("&nbsp;", " "), "") or TIP.get(label, "")
    c = f' class="{cls}"' if cls else ""
    hint = (f'<span class="hint" title="{esc(tip)}" aria-label="{esc(tip)}" tabindex="0">?</span>'
            if tip else "")
    return f'<th{c}>{label}{hint}</th>'


def hint(key, txt=None):
    """Standalone '?' tooltip for inline use (KPI labels, signal terms). Looks up TIP by key unless txt given."""
    tip = txt or TIP.get(key, "")
    if not tip:
        return ""
    return f'<span class="hint" title="{esc(tip)}" aria-label="{esc(tip)}" tabindex="0">?</span>'


def kpi_tile(k, v, dd, hero=False):
    """A KPI tile with a '?' tooltip on its label (from TIP)."""
    return (f'<div class="kpi{" hero" if hero else ""}"><div class="k">{esc(k)}{hint(k)}</div>'
            f'<div class="v">{v}</div><div class="d">{esc(dd)}</div></div>')


# ---------- equity curve ----------
curve = D["curve"]
curve_js = json.dumps([{"d": c["date"], "f": round(c["flagship"] * 100000), "s": round(c["spy"] * 100000),
                        "q": round(c.get("qqq", c["spy"]) * 100000), "r": c["ret"]} for c in curve])
configs_js = json.dumps([{"key": c["key"], "total": c["total"], "final_100k": c.get("final_100k"),
                          "cagr": c.get("cagr"), "sharpe": c["sharpe"], "dd": c["dd"], "curve": c.get("curve", [])}
                         for c in CONFIGS])
spy_curve_js = json.dumps(SPY_CURVE)
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
      <td class="num">{pctp(c.get('qqq'))}</td>
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
      <td class="num">{pct(s.get('hold_today'))}</td>
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
          <td class="num">{pct(r.get('hold_today'))}</td>
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
    def sret(r):   # the sector ETF's own return this month (what holding the ETF itself would have made)
        if r is None:
            return ' <span class="slv" title="the sector ETF had no price this month">ETF n/a</span>'
        cls = "pos" if r >= 0 else "neg"
        return (f' <span class="slv" title="the sector ETF\'s OWN return this month — what you would have made '
                f'holding the ETF itself instead of our stock pick">ETF <span class="{cls}">{r*100:+.1f}%</span></span>')

    def dvs(pr, sr):   # stock-selection edge: what we made on the stock MINUS just holding the sleeve ETF
        if pr is None or sr is None:
            return ""
        d = (pr - sr) * 100
        cls = "pos" if d >= 0 else "neg"
        return (f' <span class="dvs" title="Stock-selection edge = the stock we bought MINUS what the sleeve ETF alone returned. '
                f'Positive = picking the value stock beat just buying the sector ETF.">(<span class="{cls}">{d:+.1f}pp</span> vs ETF)</span>')
    pick_by_etf = {p["etf"]: p for p in m["picks"]}
    skip_by_etf = {sk["etf"]: sk for sk in m["skipped"]}
    chips = []
    for i, ts in enumerate(m["top_sectors"], 1):
        etf, nm, acc, er = ts["etf"], ts["sector"], ts.get("accel"), ts.get("etf_ret")
        rank = f'<span class="rk">#{i}</span>'
        head = f'{rank}<b>{esc(nm)}</b> <span class="acc" title="acceleration — the ranking signal (3-month momentum minus the prior 3-month), NOT a return">accel {acc_fmt(acc)}</span>'
        if etf in pick_by_etf:
            p = pick_by_etf[etf]
            rr = pct(p["ret"]) if p["ret"] is not None else '<span class="mut">held</span>'
            chips.append(f'<span class="chip pick" title="Ranked here because the sleeve accelerated (accel {acc_fmt(acc)} — the SELECTION signal, not a return). '
                         f'Bought {esc(p["ticker"])} at P/B {p.get("pb") and round(p["pb"],2)} on the {esc(m["date"])} close, sold on the {esc(m["ndate"])} close. '
                         f'accel ≠ outcome: the pick can fall even when the sleeve accelerated hard.">'
                         f'{head}{sret(er)} → bought <span class="tk">{esc(p["ticker"])}</span> {rr}{dvs(p["ret"], er)}</span>')
        elif etf in skip_by_etf:
            sk = skip_by_etf[etf]
            reason = sk.get("reason", "no qualifying value stock")
            chips.append(f'<span class="chip skip" title="Skipped — {esc(reason)}. The sleeve ETF itself returned '
                         f'{"" if er is None else f"{er*100:+.1f}%"} this month; we did NOT hold it.">{head} '
                         f'<span class="xm">⊘ skipped</span>{sret(er)}</span>')
        else:
            chips.append(f'<span class="chip">{head}{sret(er)}</span>')
    for dz in m.get("deactivated", []):
        der = dz.get("etf_ret")
        der_txt = "" if der is None else f"{der*100:+.1f}%"
        chips.append(f'<span class="chip deact" title="Deactivated sleeve — acceleration still calculated '
                     f'(ranked #{dz.get("rank")}) but never traded. The ETF itself returned {der_txt} — we did NOT hold it.">'
                     f'<span class="rk">#{dz.get("rank")}</span><b>{esc(dz["sector"])}</b> '
                     f'<span class="acc">accel {acc_fmt(dz.get("accel"))}</span> '
                     f'<span class="xm">⊘ deactivated</span>{sret(der)}</span>')
    top_html = "".join(chips)
    # COLD strip: every sleeve ranked #11+ (not in the pickable top-10), with its accel + own month return.
    cold = [s for s in m.get("all_sectors", []) if not s.get("in_top")]
    cchips = []
    for s in cold:
        er = s.get("etf_ret")
        ertxt = "ETF n/a" if er is None else f"ETF {er*100:+.1f}%"
        ercls = "pos" if (er or 0) >= 0 else "neg"
        cchips.append(f'<span class="chip cold" title="{esc(s["sector"])} ranked #{s["rank"]} by acceleration ({acc_fmt(s.get("accel"))}) — below the top-10 cutoff, so NOT eligible this month. '
                      f'That month the sector ETF itself returned {"n/a" if er is None else f"{er*100:+.1f}%"}.">'
                      f'<span class="rk">#{s["rank"]}</span>{esc(s["sector"])} '
                      f'<b class="{ercls}">{ertxt}</b> <span class="acc">accel {acc_fmt(s.get("accel"))}</span></span>')
    cold_html = (f'<details class="coldwrap"><summary>+ {len(cold)} colder sleeves ranked #11–{10+len(cold)} '
                 f'(not eligible this month) — click to show all with their month return</summary>'
                 f'<div class="chips cold-chips">{"".join(cchips)}</div></details>') if cchips else ""
    return top_html + cold_html


blotter = []
for m in D["months"]:
    picks = m["picks"]
    npos = sum(1 for p in picks if p["ret"] is not None)
    nskip = len(m["skipped"])
    exc = (m["basket_ret"] - m["spy_ret"]) * 100
    _mae = m.get("basket_mae")
    _maetxt = (f' · <span class="ddw" title="Worst intra-month drawdown — the deepest the basket sank between the buy and sell dates before recovering. The real risk taken even when the month ended up.">worst −{abs(_mae)*100:.1f}%</span>'
               if _mae is not None else "")
    blotter.append(f"""<tr class="mhdr">
      <td class="l mono" colspan="6"><b>bought {esc(m['date'])} close → sold {esc(m.get('ndate',''))} close</b> · {npos} holding{'s' if npos!=1 else ''} · {nskip} skipped{_maetxt}</td>
      <td class="num" colspan="3">SPY {pct(m['spy_ret'])}</td>
      <td class="num" colspan="4">basket <b>{pct(m['basket_ret'])}</b> <span class="{'pos' if exc>=0 else 'neg'}">({exc:+.1f} vs SPY)</span></td>
    </tr>
    <tr class="srow"><td colspan="{BLOTTER_COLS}"><div class="chips">{sector_chips(m)}</div></td></tr>""")
    for p in sorted(picks, key=lambda x: -(x["ret"] if x["ret"] is not None else -9)):
        conv = ' <span class="conv" title="A/D-conviction 2× weight">2×</span>' if p["conviction"] else ""
        _pm = p.get("mae")
        _pmtxt = (f' <span class="ddw" title="Worst intra-month point: this name was down {abs(_pm)*100:.1f}% at its lowest during the hold before finishing the month.">↓{abs(_pm)*100:.0f}%</span>'
                  if _pm is not None and _pm < -0.001 else "")
        rr = (pct(p["ret"]) + _pmtxt) if p["ret"] is not None else '<span class="mut">n/a</span>'
        wt = f'{p["weight"]:.0f}×' if p.get("weight") else '1×'
        why = (f'Best cheap-P/B × analyst-upside blend (P/B {p.get("pb") and round(p["pb"],2)}) among the '
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

finviz_pane = _build_finviz_pane()


def _configs_section():
    """The 'choose your regime bet' comparison — 3 configs (agnostic core / middle / aggressive) with the
    honest 2016-2026 numbers + the regime framing. Return ladder that makes the return-vs-robustness dial visible."""
    if not CONFIGS:
        return ""
    order = {"core": 0, "middle": 1, "adaptive": 2, "aggressive": 3}
    cs = sorted(CONFIGS, key=lambda c: order.get(c["key"], 9))
    accent = {"core": "var(--accent)", "middle": "var(--gold)", "adaptive": "var(--accent)", "aggressive": "var(--neg)"}
    cards = ""
    for c in cs:
        col = accent.get(c["key"], "var(--ink2)")
        cards += (
            f'<div class="cfgcard" style="border-top:3px solid {col}">'
            f'<div class="cfgname" style="color:{col}">{esc(c["name"])}</div>'
            f'<div class="cfgbig">+{c["total"]:,.0f}%</div>'
            f'<div class="cfgcagr">CAGR {c.get("cagr","—")}%/yr · Sharpe {c["sharpe"]}</div>'
            f'<div class="cfgrow"><span>Max drawdown</span><b class="neg">{c["dd"]:.1f}%</b></div>'
            f'<div class="cfgrow"><span>pre-2020 (other regime)</span><b class="{"pos" if c["pre2020"]>=0 else "neg"}">{c["pre2020"]:+.0f}%</b></div>'
            f'<div class="cfgrow"><span>2020-26 (this regime)</span><b class="pos">{c["post2020"]:,.0f}%</b></div>'
            f'<p class="cfgdesc">{esc(c["desc"])}</p></div>')
    return (
        '<section id="configs"><div class="shead"><h2>Choose your regime bet</h2>'
        '<span class="cnt">honest 2016-2026 · 3 configs</span></div>'
        '<p class="lede">The strategy is a <b>high-beta small-cap-value engine</b>. How aggressively you enforce the '
        '<b>small-cap</b> discipline is a bet on whether the <b>post-2020 regime</b> (commodity supercycle, reshoring, '
        'small-cap revival, higher-for-longer rates) persists. All three trade <i>through</i> the 2018 &amp; 2020 crashes '
        '(no lucky bottom-entry). The ladder below is the return-vs-robustness dial — read the <b>pre-2020</b> column as '
        'the stress test (a mega-cap-dominated, small-cap-hostile world, which flashed back in 2023).</p>'
        f'<div class="cfggrid">{cards}</div>'
        '<p class="chart-note">More small-cap concentration → more return in the current regime, but weaker pre-2020 and '
        'deeper drawdown. The core survives any regime; the aggressive config is levered-long the current one.</p></section>')


configs_html = _configs_section()

# calendar-year strategy returns (from the enriched data) to annotate the regime timeline
_calyr = {c["year"]: c["strategy"] for c in D.get("calendar", [])}
_calspy = {c["year"]: c["spy"] for c in D.get("calendar", [])}
REGIMES = [
    ("2016-17", "Late ZIRP · mega-cap growth melt-up", "Near-zero rates + QE funneled money into long-duration growth (FAANG). No penalty for unprofitable growth; value/small-caps ignored.", "hostile"),
    ("2018", "Rate-hike tantrum · small-cap liquidity crunch", "Fed hiking into QT; Dec −20% correction. Flight to mega-cap quality crushed small-caps and value hardest.", "worst"),
    ("2019", "Fed pivot · growth resumes", "Fed cut rates; long-duration growth led again.", "hostile"),
    ("2020", "COVID crash → liquidity flood → everything rally", "Fastest crash ever, then unprecedented stimulus → retail boom, small-cap/thematic mania (clean energy, EVs, SPACs).", "great"),
    ("2021", "Reflation · commodity & meme boom", "Reopening + stimulus + early inflation + retail. Thematic/commodity rotation paid.", "great"),
    ("2022", "Inflation shock · value's revenge", "40-yr-high inflation, aggressive hikes → growth −30%+, value & commodities WON. The ideal regime.", "great"),
    ("2023", "Narrow AI mega-cap melt-up", "ChatGPT → Magnificent-7 frenzy; ~all of SPY's +20% was 7 stocks. The 2017-style failure mode RECURRED.", "worst"),
    ("2024-26", "Broadening · commodity/critical-minerals supercycle · higher-for-longer", "Rates stay elevated (value-friendly); reshoring/defense/nuclear/minerals capex; small-cap revival; gold/uranium/copper supercycle.", "great"),
]


def _regimes_section():
    tone = {"great": "var(--pos)", "hostile": "var(--warn)", "worst": "var(--neg)"}
    rows = ""
    for yrs, name, cause, t in REGIMES:
        col = tone.get(t, "var(--ink2)")
        rows += (f'<tr><td class="l mono" style="white-space:nowrap"><b>{yrs}</b></td>'
                 f'<td class="l" style="color:{col};font-weight:600">{esc(name)}</td>'
                 f'<td class="l" style="color:var(--ink2)">{esc(cause)}</td></tr>')
    return (
        '<section id="regimes"><div class="shead"><h2>The regimes since 2016 — and what caused them</h2>'
        '<span class="cnt">why the strategy wins or loses</span></div>'
        '<p class="lede">The engine <b>thrives when value / small-caps / commodities lead</b> (2020, 2022, 2024-26) and '
        '<b>gets hurt when narrow mega-cap growth leads</b> (2017, 2018, 2023). Your regime bet is whether the post-2020 '
        'forces — higher-for-longer rates, the commodity/reshoring supercycle, the small-cap revival — are <b>structural</b> '
        'or just another cycle that reverts to a 2023-style AI-mega-cap melt-up.</p>'
        '<div class="tablecard"><div class="scrollx"><table>'
        '<thead><tr><th class="l">Years</th><th class="l">Regime</th><th class="l">What caused it</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div></div>'
        '<div class="legend" style="margin-top:12px"><span><i style="background:var(--pos)"></i>favorable (value/small-cap/commodity leads)</span>'
        '<span><i style="background:var(--warn)"></i>hostile (mega-cap growth leads)</span>'
        '<span><i style="background:var(--neg)"></i>worst (crunch / narrow melt-up)</span></div></section>')


regimes_html = _regimes_section()

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
.wrap{{max-width:1600px; margin:0 auto}}
.mpane#mpane-finviz .wrap, .mpane#mpane-finviz{{max-width:none}}
#mpane-finviz .tablecard{{max-width:none}}
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
.levbtn{{align-self:center; font-family:var(--mono); font-size:12px; padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--ink2); cursor:pointer}}
.levbtn:hover{{color:var(--ink); border-color:var(--accent)}}
.levbtn[aria-pressed="true"]{{background:color-mix(in srgb,var(--gold) 18%,transparent); color:var(--gold); border-color:var(--gold); font-weight:700}}
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
.legend .fl{{background:var(--gold)}} .legend .sp{{background:var(--mut)}} .legend .qq{{background:var(--neg)}}
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
.chip.deact{{opacity:.72; border-style:dashed; border-color:color-mix(in srgb,var(--mut) 55%,var(--line))}}
.chip.deact .xm{{color:var(--mut)}}
.chip .xm{{color:var(--neg); font-weight:700}}
.chip .rsn{{color:var(--mut); font-size:10px; max-width:300px; overflow:hidden; text-overflow:ellipsis}}
.chip .slv{{color:var(--mut); font-size:10px; font-family:var(--mono)}}
.chip .dvs{{font-size:10px; font-family:var(--mono); color:var(--mut)}}
.ddw{{font-family:var(--mono); font-size:10px; color:var(--warn); font-weight:600; cursor:help}}
.cfggrid{{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:14px 0 6px}}
@media (max-width:900px){{.cfggrid{{grid-template-columns:repeat(2,1fr)}}}}
@media (max-width:560px){{.cfggrid{{grid-template-columns:1fr}}}}
.cfgcard{{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px 18px 14px; box-shadow:var(--shadow)}}
.cfgname{{font-family:var(--mono); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px}}
.cfgbig{{font-family:var(--serif); font-size:34px; font-weight:700; color:var(--gold); line-height:1}}
.cfgcagr{{font-family:var(--mono); font-size:11px; color:var(--ink2); margin:4px 0 12px}}
.cfgrow{{display:flex; justify-content:space-between; font-size:12.5px; padding:4px 0; border-top:1px solid var(--line2)}}
.cfgrow span{{color:var(--mut)}}
.cfgdesc{{font-size:12px; color:var(--ink2); margin:12px 0 0; line-height:1.5}}
.chip.cold{{opacity:.6; font-size:10.5px; padding:2px 7px; border-style:dotted}}
.coldwrap{{margin-top:8px}}
.coldwrap summary{{font-family:var(--mono); font-size:11px; color:var(--mut); cursor:pointer; padding:4px 0; list-style:revert}}
.coldwrap summary:hover{{color:var(--ink2)}}
.coldwrap .cold-chips{{margin-top:8px}}
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
.mtabs{{position:sticky; top:0; z-index:30; display:flex; gap:4px; padding:14px 0 0; background:var(--paper); border-bottom:2px solid var(--ink)}}
.mtab{{font-family:var(--mono); font-size:12px; letter-spacing:.03em; color:var(--ink2); background:var(--band); border:1px solid var(--line); border-bottom:none; border-radius:9px 9px 0 0; padding:10px 18px; cursor:pointer; display:flex; align-items:center; gap:8px; margin-bottom:-2px}}
.mtab:hover{{color:var(--ink)}}
.mtab[aria-selected="true"]{{background:var(--card); color:var(--ink); font-weight:700; border-color:var(--ink); border-bottom:2px solid var(--card)}}
.mtab .mpill{{font-size:9.5px; padding:1px 6px; border-radius:10px; background:color-mix(in srgb,var(--accent) 18%,transparent); color:var(--accent); font-weight:700}}
.mtab .mpill.wip{{background:color-mix(in srgb,var(--gold) 20%,transparent); color:var(--gold)}}
.cfgtabs{{display:flex; flex-wrap:wrap; gap:6px; padding:14px 0 2px}}
.cfgtab{{font-family:var(--mono); font-size:11.5px; color:var(--ink2); background:var(--band); border:1px solid var(--line); border-radius:8px; padding:7px 12px; cursor:pointer; display:flex; flex-direction:column; gap:2px; line-height:1.2; text-decoration:none}}
.cfgtab:hover{{color:var(--ink); border-color:var(--accent)}}
.cfgtab[aria-selected="true"]{{background:color-mix(in srgb,var(--accent) 14%,var(--card)); color:var(--ink); border-color:var(--accent); font-weight:700}}
.cfgtab-r{{font-size:9.5px; color:var(--mut); font-weight:400}}
.mpane[hidden]{{display:none}}
.statuspill{{font-family:var(--mono); font-size:10px; letter-spacing:.05em; padding:2px 8px; border-radius:10px; background:color-mix(in srgb,var(--gold) 20%,transparent); color:var(--gold); margin-left:8px; vertical-align:middle}}
.banner{{background:color-mix(in srgb,var(--gold) 9%,var(--card)); border:1px solid color-mix(in srgb,var(--gold) 35%,var(--line)); border-radius:10px; padding:14px 16px; font-size:13.5px; color:var(--ink2); margin:26px 0 4px}}
.banner b{{color:var(--ink)}}
</style>

<div class="doc">
<div class="wrap">

  <div class="mtabs" role="tablist" aria-label="Rotation method">
    <button class="mtab" id="tab-etf" role="tab" aria-selected="true" data-pane="mpane-etf">ETF Flagship <span class="mpill">live</span></button>
    <button class="mtab" id="tab-finviz" role="tab" aria-selected="false" data-pane="mpane-finviz">Finviz Industry <span class="mpill wip">v1</span></button>
  </div>

  <div class="mpane" id="mpane-etf">
  <div class="cfgtabs" id="cfgtabs" role="tablist" aria-label="Configuration — full trades per setup">
    {"".join(f'<a class="cfgtab" href="{_CFG_FILE.get(c["key"],"#")}"{" aria-selected=true" if c["key"]==_CK else ""}>{esc(c["name"].split(" — ")[0].split(" (")[0])}<span class="cfgtab-r">+{c["total"]:,.0f}% · DD {c["dd"]:.0f}%</span></a>' for c in sorted(CONFIGS, key=lambda x: {"core":0,"middle":1,"adaptive":2,"aggressive":3}.get(x["key"],9)))}
  </div>
  <p class="lede2" style="margin-top:6px">Each tab opens that setup's <b>full document</b> — every number and every trade recomputed for that config. Currently viewing: <b>{esc(_CK)}</b>.</p>
  <header class="mast">
    <p class="eyebrow">Flagship Strategy · Full Trade History</p>
    <h1>Small-Cap Value, Bought in Accelerating Sectors</h1>
    <p class="sub">Every sector held and every stock picked that compounded $100,000 into ${S['final_100k_flagship']:,} — with the fundamentals as they stood on each purchase date.</p>
    <div class="headline">
      <div class="bignum" id="hlTotal">+{P['total']:.0f}%</div>
      <div class="bigsub">
        <span class="a"><span id="hlDollar">${S['final_100k_flagship']:,}</span> <span class="mut" style="font-weight:400">from $100k</span></span>
        <span class="b">{S['first']} → {S['last']} · {S['months']} months · vs SPY ${S['final_100k_spy']:,}</span>
      </div>
      <button class="levbtn" id="levBtn" type="button" aria-pressed="false" title="Apply 1.3× leverage to the monthly returns. Return-max dial — scales return AND drawdown proportionally (Sharpe unchanged); a risk decision, not an edge improvement.">1.3× leverage: OFF</button>
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
    {"".join(kpi_tile(k, v, dd, hero=(i in (0,1))) for i,(k,v,dd) in enumerate(kpis))}
  </div>
  <div class="funda">
    {"".join(f'<div class="fu"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="d">{esc(dd)}</div></div>' for k,v,dd in funda)}
  </div>
  <p class="chart-note" style="margin-top:8px">Fundamental medians are measured <b>at the moment of each purchase</b> (point-in-time, 45-day report lag) — not current values.</p>

  {configs_html}

  {regimes_html}

  <section id="curve">
    <div class="shead"><h2>Growth of $100,000</h2><span class="cnt">flagship vs S&amp;P 500 · {S['months']} months</span></div>
    <div class="grid2">
      <div class="chartcard">
        <canvas id="eq" aria-label="Equity curve: flagship vs S&amp;P 500"></canvas>
        <div class="legend">
          <span><i class="fl"></i>flagship — ${S['final_100k_flagship']:,}</span>
          <span><i class="sp"></i>S&amp;P 500 — ${S['final_100k_spy']:,}</span>
          <span><i class="qq"></i>QQQ (Nasdaq-100) — ${S.get('final_100k_qqq', 0):,}</span>
        </div>
        <p class="chart-note">Best month {bm['date']} {pct(bm['ret'])} · worst {wm['date']} {pct(wm['ret'])}. Monthly rebalance. QQQ = the mega-cap-growth benchmark we lose to in hostile regimes.</p>
      </div>
      <div class="tablecard calcard" style="margin-top:0" id="years">
        <div class="scrollx"><table class="sortable">
          <thead><tr><th class="l">Year</th><th>Strategy</th><th>S&amp;P 500</th><th>QQQ</th><th>Excess</th><th>Mo</th></tr></thead>
          <tbody>{"".join(cal_rows)}</tbody>
        </table></div>
      </div>
    </div>
  </section>

  <section id="how">
    <div class="shead"><h2>How a pick is made</h2></div>
    <p class="lede">Each month the engine ranks ~149 sector ETFs by momentum <b>acceleration</b>{hint("accel")} (3-month change minus the prior 3-month change) and keeps the <b>top 10</b>. Inside each, it broadens to every US/Canada stock in that GICS sector, screens for positive book value and $5M+ daily liquidity, and buys the <b>cheapest price-to-book small-cap</b>{hint("P/B")} (&lt;$2B). Names accumulating into weakness (A/D divergence) get <b>double weight</b>{hint("conviction")}. Sectors with no qualifying value stock — raw commodities, bonds, foreign index sleeves — are <b>skipped</b>{hint("skipped")}. The live default adds the analyst-upside <b>blend</b>{hint("blend")}; a demoted variant adds a 200-MA <b>regime</b>{hint("regime")} switch, and retired sleeves stay <b>deactivated</b>{hint("deactivated")}. Every fundamental below is the value <b>as filed and public on the purchase date</b>.</p>
  </section>

  <section id="stocks">
    <div class="shead"><h2>Stock leaderboard</h2><span class="cnt">{S['unique_stocks']} names · P/E · ROE · D/E at purchase</span></div>
    <p class="lede">Every distinct name ever held, ordered by summed weighted return contribution. Multiples are the <b>median at the times it was bought</b> (point-in-time). <b>†</b> = acquired/delisted during a hold. A handful of names carry the book.</p>
    <p class="lede2">Click any column header to sort. Hover the <span class="hint" style="cursor:default">?</span> on each header for its definition.</p>
    <div class="tablecard"><div class="scrolly"><table class="sortable">
      <thead><tr>
        {th("Ticker","l")}{th("Company","l")}{th("Sector(s)","l")}{th("Held")}{th("Win%")}
        {th("P/B")}{th("P/E")}{th("ROE")}{th("GP/A")}{th("Rev&nbsp;g")}{th("D/E")}{th("Mkt&nbsp;cap")}{th("Tier")}
        {th("Avg&nbsp;ret")}{th("Best")}{th("Worst")}{th("Contrib.")}{th("Held&nbsp;to&nbsp;today")}{th("Span")}
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
        <thead><tr><th class="l" colspan="9" style="color:var(--pos)">▲ Best monthly picks</th></tr>
        <tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("P/B")}{th("P/E")}{th("ROE")}{th("Return")}{th("Held&nbsp;to&nbsp;today")}</tr></thead>
        <tbody>{pick_rows(D['best_picks'])}</tbody>
      </table></div>
    </div>
    <div class="tablecard">
      <div class="scrollx"><table class="sortable">
        <thead><tr><th class="l" colspan="9" style="color:var(--neg)">▼ Worst monthly picks</th></tr>
        <tr>{th("Month","l")}{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("P/B")}{th("P/E")}{th("ROE")}{th("Return")}{th("Held&nbsp;to&nbsp;today")}</tr></thead>
        <tbody>{pick_rows(D['worst_picks'])}</tbody>
      </table></div>
    </div>
  </section>

  <section id="blotter">
    <div class="shead"><h2>Full blotter</h2><span class="cnt">{S['total_picks']} picks across {S['months']} months</span></div>
    <p class="lede">The complete month-by-month record. Each month is bought at that month-end <b>close</b> and sold at the <b>next</b> month-end close (shown in the header). Then a <b>sector-ranking strip</b> — the ten sleeves that ranked highest by momentum <b>acceleration</b> that month. <b>The percentage labelled "accel" is the ranking signal (3-month momentum minus the prior 3-month), NOT a return.</b> Each chip then shows what actually happened: <span class="chip pick" style="padding:0 5px">→ bought TICKER +x%</span> is the small-cap we bought and <i>its</i> return; <span class="chip skip" style="padding:0 5px">⊘ skipped · ETF +x%</span> means we did NOT hold it, and <b>"ETF +x%"</b> is what that sector's own ETF returned that month (so you can see exactly what we passed on). Every ranked sleeve — picked, skipped, or cold (#11+) — shows its <b>ETF</b> return. <span class="chip deact" style="padding:0 5px">⊘ deactivated</span> = a retired sleeve we still track but never trade. Beneath are the holdings with their <b>point-in-time</b> fundamentals; <b>2×</b> = A/D-conviction weight.</p>
    <div class="tablecard"><div class="scrolly"><table>
      <thead><tr>{th("Ticker","l")}{th("Company","l")}{th("Sector","l")}{th("Tier")}{th("P/B")}{th("P/E")}{th("ROE")}{th("GP/A")}{th("Rev&nbsp;g")}{th("D/E")}{th("Mkt&nbsp;cap")}{th("Weight")}{th("Return")}</tr></thead>
      <tbody>{"".join(blotter)}</tbody>
    </table></div></div>
  </section>

  <footer class="foot">
    <p><b>Fundamentals are point-in-time.</b> Every P/B, P/E, ROE and D/E shown is the value as filed and public on the purchase date (45-day report lag) — the price component is the purchase-month close, never a later close or a current value. <b>There is deliberately no forward P/E:</b> our data providers expose only today's forward estimate, and pasting that onto a 2022 buy would be look-ahead. <b>P/E is the standard trailing multiple</b> — Price ÷ trailing-12-month EPS (market cap ÷ TTM net income, four quarters summed point-in-time); a negative P/E means the company was loss-making over the trailing year, and blank means fewer than four quarters had been filed by the trade date. ROE is likewise trailing-12-month.</p>
    <p><b>Method.</b> usca_small blend arm of the survivorship-free small-cap study. <b>Selector:</b> in each accelerating sector, pick the small-cap that ranks best on a 60/40 blend of <b>analyst implied-upside</b> (Benzinga price-target ÷ price − 1, point-in-time within 90 days) and <b>cheapest P/B</b>; falls back to cheapest-P/B when no recent analyst target exists (~⅓ of picks). ARK funds are excluded (active multi-theme funds, not sectors). Universe: current sector-ETF membership (survivors) plus 1,973 major-exchange delisted names mapped by GICS, held during their live window. Returns are USD-translated (include FX P&amp;L on foreign lines), monthly rebalance, no transaction costs or slippage. Acquired/delisted names (†) exit at last/deal price via the delisting-exit mechanism — an M&amp;A announcement-exit overlay is a separate study.</p>
    <p><b>Caveats.</b> The GICS-mapped universe differs from the live ETF-membership universe, so absolutes are internally consistent but not identical to the live scanner. Delisted coverage misses pre-2020 deaths and OTC names. Simulated past performance is not a forecast. Computed {esc(D.get('computed_at',''))[:19]}.</p>
  </footer>
  </div>
{finviz_pane}
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
  // config tabs: swap the whole equity curve + headline between core/middle/adaptive/aggressive setups
  var CFGS = {configs_js};
  var SPYC = {spy_curve_js};
  var cfgtabs = document.querySelectorAll('.cfgtab');
  function pickCfg(key){{
    var c = CFGS.find(function(x){{return x.key===key;}}); if(!c || !c.curve) return;
    // rebuild `data` from this config's curve (align SPY by index)
    data = c.curve.map(function(pt,i){{ return {{d: pt.d, f: pt.f, s: (SPYC[i]||SPYC[SPYC.length-1]||100000), r: 0}}; }});
    LEV = 1.0;
    var lb=document.getElementById('levBtn'); if(lb){{lb.setAttribute('aria-pressed','false'); lb.textContent='1.3× leverage: OFF';}}
    var bt=document.getElementById('hlTotal'), bd=document.getElementById('hlDollar');
    if(bt) bt.textContent='+'+Math.round(c.total).toLocaleString()+'%';
    if(bd) bd.textContent='$'+Math.round(c.final_100k).toLocaleString();
    var ke={{'CAGR':c.cagr!=null?c.cagr.toFixed(0)+'%':'—','Sharpe':c.sharpe,'Max DD':c.dd.toFixed(1)+'%'}};
    document.querySelectorAll('.meta-row span').forEach(function(sp){{
      var b=sp.querySelector('b'); if(!b) return; var t=sp.textContent;
      if(t.indexOf('CAGR')===0) b.textContent=ke['CAGR'];
      else if(t.indexOf('Sharpe')===0) b.textContent=ke['Sharpe'];
      else if(t.indexOf('Max DD')===0) b.textContent=ke['Max DD'];
    }});
    cfgtabs.forEach(function(t){{t.setAttribute('aria-selected', t.getAttribute('data-cfg')===key?'true':'false');}});
    try{{draw();}}catch(e){{}}
  }}
  // config subtabs are links to per-config full docs (each opens its own numbers+trades); no in-page swap.
  // leverage toggle: recompute the flagship equity path from monthly returns × LEV, update chart + headline
  var LEV=1.0;
  function applyLev(){{
    var eq=100000;
    data.forEach(function(d){{ var r=(typeof d.r==='number'?d.r:0); eq*=(1+Math.max(-0.99,LEV*r)); d.lf=eq; }});
    var fin=data.length?data[data.length-1].lf:100000, tot=(fin/100000-1)*100;
    var bt=document.getElementById('hlTotal'), bd=document.getElementById('hlDollar');
    if(bt) bt.textContent='+'+Math.round(tot).toLocaleString()+'%';
    if(bd) bd.textContent='$'+Math.round(fin).toLocaleString();
  }}
  applyLev();
  var levBtn=document.getElementById('levBtn');
  if(levBtn) levBtn.addEventListener('click',function(){{
    LEV = (LEV===1.0?1.3:1.0); var on=LEV!==1.0;
    levBtn.setAttribute('aria-pressed', on?'true':'false');
    levBtn.textContent = '1.3× leverage: '+(on?'ON':'OFF');
    applyLev(); draw();
  }});
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
    var maxV=0; data.forEach(function(d){{maxV=Math.max(maxV,(d.lf||d.f),d.s,(d.q||d.s));}}); var minV=100000;
    function X(i){{return padL+(w-padL-padR)*(i/(data.length-1));}}
    function Y(v){{return padT+(h-padT-padB)*(1-(v-minV)/(maxV-minV));}}
    ctx.font='11px "SF Mono",Consolas,monospace'; ctx.textBaseline='middle';
    var step= maxV>800000?200000:100000; ctx.strokeStyle=line; ctx.fillStyle=mut; ctx.lineWidth=1;
    for(var g=minV; g<=maxV; g+=step){{ var y=Y(g); ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
      ctx.textAlign='right'; ctx.fillText('$'+(g/1000)+'k', padL-8, y); }}
    ctx.textAlign='center'; var lastYr='';
    data.forEach(function(d,i){{ var yr=d.d.slice(0,4); if(yr!==lastYr){{lastYr=yr; ctx.fillText(yr, X(i), h-8);}} }});
    var grad=ctx.createLinearGradient(0,padT,0,h-padB); grad.addColorStop(0, hexA(gold,.22)); grad.addColorStop(1, hexA(gold,0));
    ctx.beginPath(); data.forEach(function(d,i){{var x=X(i),y=Y(d.lf||d.f); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}});
    ctx.lineTo(X(data.length-1),Y(minV)); ctx.lineTo(X(0),Y(minV)); ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
    function plot(key,color,wd){{ ctx.beginPath(); ctx.lineWidth=wd; ctx.strokeStyle=color; ctx.lineJoin='round';
      data.forEach(function(d,i){{ var x=X(i),y=Y(key==='f'?(d.lf||d.f):d[key]); i?ctx.lineTo(x,y):ctx.moveTo(x,y); }}); ctx.stroke(); }}
    plot('q',css('--neg'),1.3); plot('s',mut,1.6); plot('f',gold,2.4);
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

  // ---- method tabs (ETF flagship / Finviz industry) ----
  var mtabs=document.querySelectorAll('.mtab');
  function showPane(id){{
    document.querySelectorAll('.mpane').forEach(function(p){{ p.hidden = (p.id!==id); }});
    mtabs.forEach(function(t){{ t.setAttribute('aria-selected', t.getAttribute('data-pane')===id ? 'true':'false'); }});
    if(id==='mpane-etf'){{ try{{draw();}}catch(e){{}} }}
    if(location.hash!=='#'+id) history.replaceState(null,'','#'+id);
    window.scrollTo(0,0);
  }}
  mtabs.forEach(function(t){{ t.addEventListener('click',function(){{ showPane(t.getAttribute('data-pane')); }}); }});
  if(location.hash==='#mpane-finviz') showPane('mpane-finviz');
}})();
</script>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT}  ({len(HTML):,} bytes)")
