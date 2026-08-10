"""
Stock Market Trend Bot - Report

Displays sector ETF scan with all risk metrics + trend direction.
"""

import os

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

try:
    _width = os.get_terminal_size().columns
except (ValueError, OSError):
    _width = 160
console = Console(width=max(_width, 160))


def _v(val, decimals=3):
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"


def _colored(val, spy_val, fmt=".3f", higher_better=True):
    if val is None:
        return Text("-", style="dim")
    if spy_val is None:
        return Text(f"{val:{fmt}}", style="white")
    better = val > spy_val if higher_better else val < spy_val
    style = "bright_green" if better else "red"
    return Text(f"{val:{fmt}}", style=style)


def _trend_arrow(trend, improving_dir="up"):
    """
    Return a colored arrow for trend direction.
    improving_dir: 'up' means up=good, 'down' means down=good.
    """
    if trend == "up":
        style = "bold bright_green" if improving_dir == "up" else "bold red"
        return Text("^", style=style)
    elif trend == "down":
        style = "bold bright_green" if improving_dir == "down" else "bold red"
        return Text("v", style=style)
    return Text("-", style="dim")


def _val_with_trend(val, spy_val, trend, fmt=".3f", higher_better=True, improving_dir="up"):
    """Combine value (colored vs SPY) with trend arrow."""
    if val is None:
        return Text("-", style="dim")
    if spy_val is None:
        better_style = "white"
    else:
        better = val > spy_val if higher_better else val < spy_val
        better_style = "bright_green" if better else "red"

    arrow = "^" if trend == "up" else "v" if trend == "down" else ""
    if arrow:
        good = (trend == "up" and improving_dir == "up") or (trend == "down" and improving_dir == "down")
        arrow_style = "bright_green" if good else "red"
        t = Text(f"{val:{fmt}}", style=better_style)
        t.append(arrow, style=arrow_style)
        return t
    return Text(f"{val:{fmt}}", style=better_style)


def print_results(results: list[dict]):
    console.print()
    console.print(Panel(
        Text(
            "SECTOR ROTATION  —  RSI(10) > SMA(10)  +  Omega(10) > 1  +  Risk Metrics",
            style="bold white",
            justify="center",
        ),
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(0, 2),
    ))
    console.print()

    if not results:
        console.print("[red]No data available.[/]")
        return

    r0 = results[0]
    console.print(
        f"  SPY:  Sortino [bold]{_v(r0['spy_sortino'])}[/]  |  "
        f"Omega [bold]{_v(r0['spy_omega'])}[/]  |  "
        f"CVaR [bold]{_v(r0['spy_cvar'])}[/]  |  "
        f"Ulcer [bold]{_v(r0['spy_ulcer'])}[/]"
    )
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Sector", min_width=20)
    table.add_column("ETF", width=5, justify="center")
    table.add_column("RSI", justify="right", width=5)
    table.add_column("SMA", justify="right", width=5)
    table.add_column("R>S", justify="center", width=3)
    table.add_column("Sortino", justify="right", width=8)
    table.add_column("Omega", justify="right", width=8)
    table.add_column("CVaR", justify="right", width=7)
    table.add_column("Ulcer", justify="right", width=7)
    table.add_column("Up%", justify="right", width=5)
    table.add_column("Dn%", justify="right", width=7)
    table.add_column("Beta", justify="right", width=5)
    table.add_column("BQQQ", justify="right", width=5)
    table.add_column("cSPY", justify="right", width=5)
    table.add_column("cQQQ", justify="right", width=5)
    table.add_column("MACD", justify="right", width=6)
    table.add_column("Cr", justify="center", width=3)
    table.add_column("Fresh", justify="center", width=6)
    table.add_column("Signal", justify="center", width=13)

    for i, r in enumerate(results, 1):
        rsi_above = r["rsi_above_sma"]
        rsi_check = Text("Y", style="bold bright_green") if rsi_above else Text("N", style="red")
        rsi_style = "bright_green" if rsi_above else "red"

        # Values with trend arrows
        sortino_text = _val_with_trend(
            r["sortino"], r["spy_sortino"], r.get("sortino_trend", "flat"),
            fmt=".3f", higher_better=True, improving_dir="up",
        )
        omega_text = _val_with_trend(
            r["omega"], r["spy_omega"], r.get("omega_trend", "flat"),
            fmt=".3f", higher_better=True, improving_dir="up",
        )
        cvar_text = _colored(r["cvar"], r["spy_cvar"], fmt=".4f", higher_better=True)
        ulcer_text = _val_with_trend(
            r["ulcer"], r["spy_ulcer"], r.get("ulcer_trend", "flat"),
            fmt=".2f", higher_better=False, improving_dir="down",
        )

        up_cap = r.get("up_capture")
        up_text = Text(f"{up_cap:.0f}" if up_cap is not None else "-",
                       style="bright_green" if up_cap and up_cap > 100 else "red" if up_cap else "dim")

        down_cap = r.get("down_capture")
        dn_trend = r.get("down_capture_trend", "flat")
        if down_cap is not None:
            dn_style = "bright_green" if down_cap < 100 else "red"
            dn_text = Text(f"{down_cap:.0f}", style=dn_style)
            if dn_trend == "up":
                dn_text.append("^", style="red")       # dn% going up = bad
            elif dn_trend == "down":
                dn_text.append("v", style="bright_green")  # dn% going down = good
        else:
            dn_text = Text("-", style="dim")

        beta = r.get("beta")
        if beta is not None:
            beta_style = "bright_green" if beta < 1 else "red" if beta > 1.5 else "yellow"
            beta_text = Text(f"{beta:.2f}", style=beta_style)
        else:
            beta_text = Text("-", style="dim")

        # Beta vs QQQ + correlation to SPY / QQQ
        bqqq = r.get("beta_qqq")
        if bqqq is not None:
            bqqq_style = "bright_green" if bqqq < 1 else "red" if bqqq > 1.5 else "yellow"
            bqqq_text = Text(f"{bqqq:.2f}", style=bqqq_style)
        else:
            bqqq_text = Text("-", style="dim")

        def _corr_text(val):
            if val is None:
                return Text("-", style="dim")
            style = "bright_green" if val >= 0.7 else "yellow" if val >= 0.4 else "red"
            return Text(f"{val:.2f}", style=style)

        cspy_text = _corr_text(r.get("corr_spy"))
        cqqq_text = _corr_text(r.get("corr_qqq"))

        # MACD: show histogram with a great-flag and trend arrow
        macd_great = r.get("macd_great", False)
        macd_hist = r.get("macd_hist")
        if macd_hist is not None:
            macd_style = "bold bright_green" if macd_great else "red"
            macd_text = Text(f"{macd_hist:+.3f}", style=macd_style)
            htrend = r.get("macd_hist_trend", "flat")
            if htrend == "up":
                macd_text.append("^", style="bright_green")
            elif htrend == "down":
                macd_text.append("v", style="red")
        else:
            macd_text = Text("-", style="dim")

        if r["rsi_crossover"]:
            days = r.get("crossover_days_ago")
            label = f"{days}d" if days and days > 0 else "now"
            cross = Text(label, style="bold bright_green")
        else:
            cross = Text("-", style="dim")

        # Fresh composite: WEEKLY Sortino>0 + RSI x + RSI-of-Sortino x within 14d
        fresh_state = r.get("fresh_state")
        if fresh_state == "FRESH":
            fd = r.get("fresh_days")
            flabel = f"{fd}d" if fd and fd > 0 else "now"
            fresh_text = Text(flabel, style="bold bright_green")
        elif fresh_state == "POTENTIAL":
            fresh_text = Text("~2/3", style="yellow")
        else:
            fresh_text = Text("-", style="dim")

        signal = r["signal"]
        if signal == "ROTATE IN":
            sig_style = "bold bright_green on dark_green"
        elif signal == "BULLISH":
            sig_style = "bold bright_green"
        elif signal in ("RSI ONLY", "SORTINO ONLY"):
            sig_style = "yellow"
        else:
            sig_style = "red"

        table.add_row(
            str(i),
            r["sector"],
            r["etf"],
            Text(f"{r['rsi']:.0f}", style=rsi_style),
            Text(f"{r['rsi_sma']:.0f}", style="dim"),
            rsi_check,
            sortino_text,
            omega_text,
            cvar_text,
            ulcer_text,
            up_text,
            dn_text,
            beta_text,
            bqqq_text,
            cspy_text,
            cqqq_text,
            macd_text,
            cross,
            fresh_text,
            Text(signal, style=sig_style),
        )

    console.print(table)

    console.print()
    console.print("  [dim]^ = trending up  v = trending down  |  Green arrow = improving  Red arrow = deteriorating[/]")
    console.print("  [dim]Green value = beats SPY  |  Up% = upside capture  |  Dn% = downside capture (lower=better)[/]")
    console.print("  [dim]MACD = histogram (macd - signal); informational only, not part of the signal[/]")
    console.print("  [dim]Fresh = WEEKLY Sortino>0 (completed weeks, no repaint) + RSI x + RSI-of-Sortino x within 14d (Nd = days since completion); expires after 14d; ~2/3 = one condition away[/]")

    # Summary
    bullish = sum(1 for r in results if r["bullish"])
    rsi_only = sum(1 for r in results if r["signal"] == "RSI ONLY")
    omega_only = sum(1 for r in results if r["signal"] == "OMEGA ONLY")
    bearish = sum(1 for r in results if r["signal"] == "BEARISH")
    crossovers = [r["sector"] for r in results if r["rsi_crossover"] and r["bullish"]]

    console.print(
        f"  [bold bright_green]{bullish} BULLISH[/]  |  "
        f"[yellow]{rsi_only} RSI only[/]  |  "
        f"[yellow]{omega_only} Omega only[/]  |  "
        f"[red]{bearish} BEARISH[/]"
    )
    if crossovers:
        console.print(f"  [bold bright_green]Fresh crossovers (both pass): {', '.join(crossovers)}[/]")

    # Fresh composite summary
    fresh = [r for r in results if r.get("fresh_state") == "FRESH"]
    potential = [r for r in results if r.get("fresh_state") == "POTENTIAL"]
    fresh.sort(key=lambda r: (r.get("fresh_days") if r.get("fresh_days") is not None else 99))
    console.print(
        f"  [bold bright_green]{len(fresh)} FRESH[/]  |  "
        f"[yellow]{len(potential)} potentially fresh (~2/3)[/]"
    )
    if fresh:
        top = ", ".join(
            f"{r['sector']} ({r['fresh_days']}d)"
            for r in fresh[:8]
        )
        console.print(f"  [bold bright_green]Freshest: {top}[/]")
    if potential:
        # show what each is waiting on
        def _miss(r):
            return f"{r['sector']} (needs {'+'.join(r.get('fresh_missing', []))})"
        console.print(f"  [yellow]Potential: {', '.join(_miss(r) for r in potential[:6])}[/]")
    console.print()
