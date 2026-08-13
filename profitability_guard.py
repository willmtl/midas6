"""Shared profitability-guard flags for the live value pick. Backtest (profitability_guard_study.py)
found the `ex_trap_turn` guard beats the unguarded cheapest-P/B pick on every metric: DROP only the
true value trap (unprofitable AND book shrinking AND not improving), KEEP genuine negative-EPS
turnarounds. Blanket 'profitable-only' HURTS (kills the turnarounds).

guard_flags(tickers) -> {ticker: {status, profitable, improving, book_stable, trap, net_income, margin}}
from FinancialReport (net_income / total_equity / revenue). status in {profitable, turnaround, ok,
trap, unknown}. Callers exclude `trap` candidates and can badge the rest.
"""
from seq_fundamental_study import load_financial_reports


def _f(x):
    try:
        v = float(x); return v if v == v else None
    except (TypeError, ValueError):
        return None


def guard_flags(tickers):
    reps = load_financial_reports(sorted(set(tickers)))
    out = {}
    for t, df in reps.items():
        if df is None or len(df) == 0:
            out[t] = {"status": "unknown", "profitable": None, "improving": None,
                      "book_stable": None, "trap": False, "net_income": None, "margin": None}
            continue
        d = df.sort_values("period_end")
        ni = d["net_income"].dropna()
        eq = d["total_equity"].dropna()
        rev = d["revenue"].dropna()
        latest_ni = _f(ni.iloc[-1]) if len(ni) else None
        prior_ni = _f(ni.iloc[-2]) if len(ni) >= 2 else None
        latest_eq = _f(eq.iloc[-1]) if len(eq) else None
        eq_back = _f(eq.iloc[-5]) if len(eq) >= 5 else (_f(eq.iloc[0]) if len(eq) else None)
        latest_rev = _f(rev.iloc[-1]) if len(rev) else None

        profitable = latest_ni is not None and latest_ni > 0
        improving = latest_ni is not None and prior_ni is not None and latest_ni > prior_ni
        book_stable = latest_eq is not None and eq_back is not None and latest_eq >= eq_back
        trap = (latest_ni is not None and latest_ni < 0) and (not book_stable) and (not improving)
        if trap:
            status = "trap"
        elif latest_ni is not None and latest_ni < 0 and improving:
            status = "turnaround"
        elif profitable:
            status = "profitable"
        else:
            status = "ok"
        margin = round(latest_ni / latest_rev * 100, 1) if (latest_ni is not None and latest_rev) else None
        out[t] = {"status": status, "profitable": bool(profitable), "improving": bool(improving),
                  "book_stable": bool(book_stable) if latest_eq is not None else None,
                  "trap": bool(trap), "net_income": latest_ni, "margin": margin}
    return out
