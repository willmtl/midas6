#!/usr/bin/env python3
"""Unit tests for h4_c_upside. Run: docker exec rotation-backend-1 python -u /app/test_h4_c_upside.py"""
import sys, datetime as dt


def test_upside_asof():
    from h4_c_upside import upside_asof, bucket_upside
    store = {"AAA": ([dt.date(2023, 1, 1), dt.date(2023, 6, 1)], [110.0, 150.0])}
    # before first target -> None
    assert upside_asof(store, "AAA", dt.date(2022, 12, 1), 100.0) is None
    # after first, before second -> uses 110
    assert abs(upside_asof(store, "AAA", dt.date(2023, 3, 1), 100.0) - 10.0) < 1e-9
    # on/after second -> uses 150 (no future leak: exactly the as-of)
    assert abs(upside_asof(store, "AAA", dt.date(2023, 6, 1), 100.0) - 50.0) < 1e-9
    # unknown ticker -> None
    assert upside_asof(store, "ZZZ", dt.date(2023, 6, 1), 100.0) is None
    print("test_upside_asof OK")


def test_bucket_upside():
    from h4_c_upside import bucket_upside
    assert bucket_upside(None) == "no_target"
    assert bucket_upside(-5) == "<0%"
    assert bucket_upside(10) == "0-25%"
    assert bucket_upside(40) == "25-50%"
    assert bucket_upside(75) == "50-100%"
    assert bucket_upside(250) == ">100%"
    print("test_bucket_upside OK")


CHECKS = {"asof": test_upside_asof, "bucket": test_bucket_upside}


def main():
    names = [a for a in sys.argv[1:] if a in CHECKS] or list(CHECKS)
    for nm in names:
        CHECKS[nm]()
    print(f"\n{len(names)} check(s) passed.")


if __name__ == "__main__":
    main()
