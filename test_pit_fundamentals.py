"""Standalone tests for point-in-time fundamentals. Run:
  docker compose exec backend python -u test_pit_fundamentals.py
"""
import numpy as np
import pandas as pd
from datetime import date

import pit_fundamentals as P


def _price(n=400, start=50.0, end=60.0):
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.Series(np.linspace(start, end, n), index=idx)
    return pd.DataFrame({"Close": close, "Volume": np.full(n, 1_000_000.0)}, index=idx)


def _reports():
    # Five quarters so TTM (rolling 4Q) is defined by Q4. Q1 period_end 2022-03-31 ->
    # avail 2022-05-15; Q4 2022-12-31 -> avail 2023-02-14.
    return pd.DataFrame([
        dict(period_end=date(2022, 3, 31), revenue=100, net_income=10, eps_diluted=1.0,
             operating_income=12, total_equity=200, total_debt=100,
             current_assets=150, current_liabilities=75, free_cash_flow=8, shares_outstanding=1_000),
        dict(period_end=date(2022, 6, 30), revenue=110, net_income=11, eps_diluted=1.1,
             operating_income=13, total_equity=210, total_debt=90,
             current_assets=160, current_liabilities=80, free_cash_flow=9, shares_outstanding=1_000),
        dict(period_end=date(2022, 9, 30), revenue=120, net_income=12, eps_diluted=1.2,
             operating_income=14, total_equity=220, total_debt=85,
             current_assets=170, current_liabilities=82, free_cash_flow=10, shares_outstanding=1_000),
        dict(period_end=date(2022, 12, 31), revenue=130, net_income=13, eps_diluted=1.3,
             operating_income=15, total_equity=230, total_debt=80,
             current_assets=180, current_liabilities=85, free_cash_flow=11, shares_outstanding=1_000),
        dict(period_end=date(2023, 3, 31), revenue=140, net_income=14, eps_diluted=1.4,
             operating_income=16, total_equity=240, total_debt=75,
             current_assets=190, current_liabilities=88, free_cash_flow=12, shares_outstanding=1_000),
    ]).assign(avail_date=lambda d: d["period_end"].map(P.avail_date))


def test_avail_date_adds_45_days():
    assert P.avail_date(date(2022, 3, 31)) == date(2022, 5, 15)


def test_no_report_visible_before_avail_date_is_nan():
    m = P.prepare_pit_metrics(_price(), _reports(), pd.Series(dtype=float), None)
    # A date before the first avail_date (2022-05-15) must not see Q1's numbers.
    early = m.loc[m.index < pd.Timestamp("2022-05-15")]
    assert early["pe"].isna().all(), "lookahead: report used before it was public"


def test_report_visible_on_and_after_avail_date():
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    row = P.pit_row(m, "2022-06-01")  # after Q1 avail, before Q2 avail
    close = float(price["Close"].asof(pd.Timestamp("2022-06-01")))
    assert not np.isnan(row["market_cap"])
    assert abs(row["market_cap"] - close * 1_000) < 1e-6      # Close x Q1 shares
    assert abs(row["current_ratio"] - (150 / 75)) < 1e-9      # 2.0 from Q1


def test_ttm_uses_last_four_quarters_and_pe_positive():
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    row = P.pit_row(m, "2023-03-01")  # 4 quarters visible (Q4 avail 2023-02-14)
    assert row["eps_ttm"] > 0 and row["pe"] > 0


def test_dividend_yield_ttm_over_price():
    price = _price(end=50.0)  # flat at 50
    divs = pd.Series([0.5, 0.5], index=[pd.Timestamp("2022-06-15"), pd.Timestamp("2022-09-15")])
    m = P.prepare_pit_metrics(price, _reports(), divs, None)
    row = P.pit_row(m, "2022-10-01")
    assert abs(row["dividend_yield"] - (1.0 / 50.0)) < 1e-6   # 1.0 TTM / 50 price


def test_bucket_boundaries():
    assert P.bucket_pb(0.9) == "cheap (<1)"
    assert P.bucket_pb(6.5) == "expensive (>=6)"
    assert P.bucket_current_ratio(0.8) == "weak (<1)"
    assert P.bucket_fcf_sign(-1.0) == "negative FCF"
    assert P.bucket_div_yield(None) == "NA"


def test_trade_labels_are_pointintime():
    import seq_fundamental_study as S
    price = _price()
    m = P.prepare_pit_metrics(price, _reports(), pd.Series(dtype=float), None)
    labels = S.label_trade(m, pd.Timestamp("2023-03-01"), {})  # 4 quarters visible
    assert labels["Current ratio"] in ("ok (1-2)", "strong (>=2)")  # PIT dim resolved (not NA)
    assert labels["PE (trailing)"] != "NA"                          # PIT dim resolved
    # A trade before any report is public -> all (now PIT-only) dims are NA.
    early = S.label_trade(m, pd.Timestamp("2022-02-01"), {})
    assert early["PE (trailing)"] == "NA"


def test_dimensions_are_all_pointintime():
    # The 6 snapshot-only dims were removed 2026-08-07 (no dated history -> lookahead).
    import seq_fundamental_study as S
    assert all(len(t) == 5 for t in S.DIMENSIONS), "DIMENSIONS must be 5-tuples with pit flag"
    snap = [t for t in S.DIMENSIONS if not t[4]]
    assert len(snap) == 0, f"no snapshot/lookahead dims allowed, found {[t[0] for t in snap]}"
    assert len(S.DIMENSIONS) == 29


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
