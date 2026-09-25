#!/usr/bin/env python
"""Render the Sortino-RSI signal (from meta_sortino_viz.json) to a large, detailed static PNG."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

d = json.load(open("/app/.data/studies/meta_sortino_viz.json"))
dt = [datetime.strptime(x, "%Y-%m-%d") for x in d["dates"]]
close = np.array(d["close"]); rsi14 = np.array(d["rsi14"]); rsi20 = np.array(d["rsi20"])
sma14 = np.array(d["sma14"]); sortino = np.array(d["sortino"]); sortino_s = np.array(d.get("sortino_s", d["sortino"]))
strat_eq = np.array(d["strat_eq"]); bh_eq = np.array(d["bh_eq"])
buy = np.array(d["buy"]); sell = np.array(d["sell"]); held = np.array(d["held"]); cross = np.array(d.get("cross", [False] * len(dt)))
P = d["params"]; TK = d.get("ticker", "META"); vtrades = d.get("vtrades", [])
bt = {t["entry"]: t for t in vtrades}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#9aa3b2", "axes.labelcolor": "#33384a",
                     "xtick.color": "#5b6172", "ytick.color": "#5b6172", "figure.facecolor": "white"})
fig, ax = plt.subplots(5, 1, figsize=(17, 16), sharex=True,
                       gridspec_kw={"height_ratios": [3.2, 1.7, 1.2, 1.2, 1.9], "hspace": 0.10})

def shade(a):
    inpos = False; start = None
    for i in range(len(held)):
        if held[i] and not inpos:
            inpos = True; start = dt[i]
        elif not held[i] and inpos:
            inpos = False; a.axvspan(start, dt[i], color="#5ec8c8", alpha=0.12, lw=0)
    if inpos:
        a.axvspan(start, dt[-1], color="#5ec8c8", alpha=0.12, lw=0)

# ---- P1 price + markers + numbered trades + price labels
shade(ax[0])
ax[0].plot(dt, close, color="#2f6fd0", lw=1.6)
bi = [i for i in range(len(buy)) if buy[i]]; si = [i for i in range(len(sell)) if sell[i]]
ax[0].scatter([dt[i] for i in bi], [close[i] * 0.94 for i in bi], marker="^", color="#16a34a", s=120, zorder=6, label="BUY (entry)")
ax[0].scatter([dt[i] for i in si], [close[i] * 1.06 for i in si], marker="v", color="#dc2626", s=120, zorder=6, label="EXIT (RSI20 red)")
for i in bi:
    key = dt[i].strftime("%Y-%m-%d"); t = bt.get(key)
    lbl = f"#{t['n']}" if t else ""
    ax[0].annotate(lbl, (dt[i], close[i] * 0.94), textcoords="offset points", xytext=(0, -16),
                   ha="center", fontsize=8, color="#166534", fontweight="bold")
    if t:
        ax[0].annotate(f"${t['entry_px']:g}", (dt[i], close[i]), textcoords="offset points", xytext=(0, 8),
                       ha="center", fontsize=7, color="#166534")
for i in si:
    key = dt[i].strftime("%Y-%m-%d")
    ax[0].annotate(f"${close[i]:g}", (dt[i], close[i]), textcoords="offset points", xytext=(0, -12),
                   ha="center", fontsize=7, color="#991b1b")
ax[0].set_ylabel(f"{TK}  price ($)", fontsize=11); ax[0].legend(loc="upper left", fontsize=9, framealpha=.92)
ax[0].set_title(f"Sortino-RSI Technique on {TK}  —  entries ▲, exit-when-red ▼, shaded = in position",
                fontsize=14, fontweight="bold", loc="left", color="#1b2030", pad=10)

# ---- P2 equity: strategy vs buy-and-hold
shade(ax[1])
ax[1].plot(dt, strat_eq, color="#7c3aed", lw=2.0, label=f"Strategy (long only when held)  ×{strat_eq[-1]:.2f}")
ax[1].plot(dt, bh_eq, color="#94a3b8", lw=1.4, ls="--", label=f"Buy & hold {TK}  ×{bh_eq[-1]:.2f}")
ax[1].axhline(1.0, color="#c3c9d6", ls=":", lw=1)
ax[1].set_ylabel("growth of $1\n(in-window)", fontsize=10); ax[1].legend(loc="upper left", fontsize=9, framealpha=.92)

# ---- P3 RSI20 of HIGH (color)
shade(ax[2])
ax[2].plot(dt, rsi20, color="#0d9488", lw=1.5)
ax[2].axhline(P["green"], color="#16a34a", ls="--", lw=1); ax[2].axhline(P["red"], color="#dc2626", ls="--", lw=1)
ax[2].fill_between(dt, P["green"], 92, color="#16a34a", alpha=.05)
ax[2].fill_between(dt, 8, P["red"], color="#dc2626", alpha=.05)
ax[2].set_ylabel("RSI20\nof CLOSE", fontsize=10); ax[2].set_ylim(15, 90)
ax[2].text(dt[2], P["green"] + 1.5, f"{P['green']:.0f} = green (uptrend)", color="#16a34a", fontsize=8)
ax[2].text(dt[2], P["red"] - 5, f"{P['red']:.0f} = red → EXIT", color="#dc2626", fontsize=8)

# ---- P4 rel-Sortino (raw faint + smoothed bold)
shade(ax[3])
ax[3].plot(dt, sortino, color="#d0b48a", lw=0.8, alpha=.6, label="raw")
ax[3].plot(dt, sortino_s, color="#d97706", lw=1.8, label=f"smoothed (SMA{P.get('smooth', 14)})")
ax[3].axhline(0, color="#9aa3b2", ls=":", lw=1)
ax[3].fill_between(dt, 0, sortino_s, where=(sortino_s >= 0), color="#d97706", alpha=.08)
ax[3].fill_between(dt, 0, sortino_s, where=(sortino_s < 0), color="#dc2626", alpha=.08)
ax[3].set_ylabel(f"rel-Sortino\n({P['win']}) vs SPY", fontsize=10)
ax[3].legend(loc="upper left", fontsize=8, framealpha=.9)

# ---- P5 RSI14 of sortino + SMA + entry dots
shade(ax[4])
ax[4].fill_between(dt, 0, P["gate"], color="#16a34a", alpha=.04)
ax[4].plot(dt, rsi14, color="#9333ea", lw=1.7, label=f"RSI{P['rsi14']} of smoothed Sortino")
ax[4].plot(dt, sma14, color="#94a3b8", lw=1.3, label=f"SMA{P['sma']}")
ax[4].axhline(P["gate"], color="#64748b", ls="--", lw=1); ax[4].axhline(20, color="#ea8f5f", ls="--", lw=1)
ci = [i for i in range(len(cross)) if cross[i]]
ax[4].scatter([dt[i] for i in ci], [rsi14[i] for i in ci], marker="x", color="#0ea5e9", s=55, zorder=6, lw=1.6, label="cross (arms entry)")
ax[4].scatter([dt[i] for i in bi], [rsi14[i] for i in bi], color="#16a34a", s=60, zorder=7, edgecolor="white", lw=1.0, label="BUY (green confirms)")
ax[4].set_ylabel(f"RSI{P['rsi14']} of\nSortino", fontsize=10); ax[4].set_ylim(0, 100)
ax[4].legend(loc="upper left", fontsize=9, framealpha=.92)
ax[4].text(dt[2], P["gate"] + 1.5, f"{P['gate']:.0f} gate (must be below)", color="#64748b", fontsize=8)
ax[4].text(dt[2], 22, "20 = ideal deep dip", color="#ea8f5f", fontsize=8)

for a in ax:
    a.grid(True, color="#eceef3", lw=.7); a.set_axisbelow(True)
ax[4].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax[4].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax[4].get_xticklabels(), rotation=0, ha="center", fontsize=9)
fig.savefig("/app/.data/studies/meta_sortino.png", dpi=135, bbox_inches="tight")
print("saved /app/.data/studies/meta_sortino.png  buys=%d sells=%d  strat=x%.2f  b&h=x%.2f" %
      (buy.sum(), sell.sum(), strat_eq[-1], bh_eq[-1]))
