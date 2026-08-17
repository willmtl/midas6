import threading
from datetime import date

from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from core.models import Sector, Candle, Study, Trade, ScanResult, StudySectorResult, TrendStudy, StockDrilldown
from api.serializers import (
    SectorSerializer, ScanResultSerializer, StudySerializer, TradeSerializer, CandleSerializer
)


class SectorListView(APIView):
    def get(self, request):
        sectors = Sector.objects.all()
        return Response(SectorSerializer(sectors, many=True).data)


class ScanView(APIView):
    def get(self, request):
        interval = request.query_params.get("interval", "1d")
        force = request.query_params.get("force", "false") == "true"

        results = ScanResult.objects.filter(interval=interval).select_related("sector")

        if not results.exists() or force:
            # Trigger computation
            from api.tasks import compute_scan
            compute_scan(interval)
            results = ScanResult.objects.filter(interval=interval).select_related("sector")

        data = ScanResultSerializer(results.order_by("-rsi_spread"), many=True).data

        spy_vals = {}
        if data:
            spy_vals = {
                "spy_sortino": data[0].get("spy_sortino"),
                "spy_omega": data[0].get("spy_omega"),
                "spy_cvar": data[0].get("spy_cvar"),
                "spy_ulcer": data[0].get("spy_ulcer"),
            }

        bullish_count = sum(1 for r in data if r.get("bullish"))

        return Response({
            **spy_vals,
            "total": len(data),
            "bullish": bullish_count,
            "sectors": data,
            "cached_at": data[0].get("computed_at") if data else None,
        })


class DrilldownView(APIView):
    def get(self, request, sector_name):
        from api.tasks import compute_drilldown
        sector = Sector.objects.filter(name__icontains=sector_name).first()
        if not sector:
            return Response({"error": f"Sector '{sector_name}' not found"}, status=404)

        result = compute_drilldown(sector)
        return Response(result)


class ChartView(APIView):
    def get(self, request, ticker):
        from api.tasks import get_chart_data
        interval = request.query_params.get("interval", "1d")
        sector_etf = request.query_params.get("sector_etf")
        period = request.query_params.get("period", "5y")

        result = get_chart_data(ticker.upper(), interval, sector_etf, period)
        if result is None:
            return Response({"error": "Ticker not found"}, status=404)
        return Response(result)


class StudyListView(APIView):
    def get(self, request):
        """Serve studies from database. Cursor-paginated when ?paginate=1 (or ?cursor=...) is sent —
        the infinite-scroll frontend opts in; legacy callers get the full list unchanged."""
        from django.db.models import Max, Q
        qs = Study.objects.filter(is_computed=True)
        last = qs.aggregate(m=Max("computed_at"))["m"]
        last_iso = last.isoformat() if last else None

        # Grouped-by-signal view: a "study" is the SIGNAL (tested across every exit), so collapse the
        # signal×exit rows to one entry per signal — its best exit as the headline + the exit range +
        # profitable-exit count. ~354 signals, so returned in one shot (expand fetches the exit ladder
        # via ?signal=X). Honors the same search / category / regime filters.
        if request.query_params.get("group") == "signal":
            qp = request.query_params
            search = (qp.get("search") or "").strip()
            if search:
                qs = qs.filter(Q(signal_name__icontains=search) | Q(name__icontains=search)
                               | Q(exit_name__icontains=search))
            if qp.get("category") and qp.get("category") != "all":
                qs = qs.filter(category=qp.get("category"))
            regime = qp.get("regime")
            REGIME_FIELDS = {"by_regime", "by_curve", "by_vix", "by_spy_trend", "by_season"}
            if regime and ":" in regime:
                rtype, rkey = regime.split(":", 1)
                if rtype in REGIME_FIELDS and rkey:
                    qs = qs.filter(**{f"{rtype}__{rkey}__isnull": False})
            groups = {}
            for r in qs.values("signal_key", "signal_name", "category", "exit_key", "exit_name",
                               "avg_return", "win_rate", "total_trades", "t_stat", "avg_hold",
                               "avg_mae", "clean_pct"):
                g = groups.get(r["signal_key"])
                if g is None:
                    g = groups[r["signal_key"]] = {
                        "signal": r["signal_key"], "signal_name": r["signal_name"],
                        "category": r["category"], "n_exits": 0, "profitable": 0,
                        "min_ret": None, "max_ret": None, "best": None}
                g["n_exits"] += 1
                ar = r["avg_return"]
                if ar is not None:
                    if ar > 0:
                        g["profitable"] += 1
                    g["min_ret"] = ar if g["min_ret"] is None else min(g["min_ret"], ar)
                    g["max_ret"] = ar if g["max_ret"] is None else max(g["max_ret"], ar)
                cur = g["best"]
                if cur is None or (ar or -1e9) > (cur["avg_return"] or -1e9):
                    g["best"] = {"exit": r["exit_key"], "exit_name": r["exit_name"], "avg_return": ar,
                                 "win_rate": r["win_rate"], "trades": r["total_trades"],
                                 "t_stat": r["t_stat"], "avg_hold": r["avg_hold"],
                                 "avg_mae": r["avg_mae"], "clean_pct": r["clean_pct"]}
            grouped = sorted(groups.values(), key=lambda x: -((x["best"] or {}).get("avg_return") or -1e9))
            # Live 'firing now' count per signal (signal_firing_scan.py) for the firing column.
            from core.models import SignalFiring
            fmap = dict(SignalFiring.objects.values_list("signal_key", "n_firing"))
            for g in grouped:
                g["n_firing"] = fmap.get(g["signal"], 0)
            categories = sorted(c for c in Study.objects.filter(is_computed=True).order_by()
                                .values_list("category", flat=True).distinct() if c)
            return Response({"grouped": grouped, "total_signals": len(grouped),
                             "categories": categories, "last_updated": last_iso})

        if request.query_params.get("paginate") == "1" or request.query_params.get("offset") is not None:
            from django.db.models import F, FloatField
            from api.pagination import resolve_ordering, paginate_offset, paged_response
            qp = request.query_params
            # Server-side filters so the paginated query mirrors the old client-side filter UX
            # (search / category / signal / exit / regime). Applied BEFORE the slice so paging walks
            # the filtered set.
            search = (qp.get("search") or "").strip()
            if search:
                qs = qs.filter(Q(signal_name__icontains=search) | Q(name__icontains=search)
                               | Q(exit_name__icontains=search))
            if qp.get("category") and qp.get("category") != "all":
                qs = qs.filter(category=qp.get("category"))
            if qp.get("signal"):
                qs = qs.filter(signal_key=qp.get("signal"))
            if qp.get("exit"):
                qs = qs.filter(exit_key=qp.get("exit"))
            # Every sortable column in the table maps to a DB field; offset paging orders by any of
            # them (NULLS LAST), so nullable metrics like t_stat/avg_mae/clean_pct are fine.
            STUDY_ORDER = {"id": "id", "name": "name", "category": "category", "exit_name": "exit_name",
                           "total_trades": "total_trades", "trades": "total_trades", "t_stat": "t_stat",
                           "avg_return": "avg_return", "win_rate": "win_rate", "avg_hold": "avg_hold",
                           "avg_mae": "avg_mae", "clean_pct": "clean_pct", "peak_day": "peak_day",
                           "peak_avg": "peak_avg", "ret_90d": "ret_90d",
                           "best_peak_ret": "best_peak_ret", "best_ret_90d": "best_ret_90d"}
            REGIME_FIELDS = {"by_regime", "by_curve", "by_vix", "by_spy_trend", "by_season"}
            regime = qp.get("regime")
            ordering = None
            if regime and ":" in regime:
                rtype, rkey = regime.split(":", 1)
                if rtype in REGIME_FIELDS and rkey:
                    # Keep only studies carrying that bucket, then sort by its nested numeric stat.
                    qs = qs.filter(**{f"{rtype}__{rkey}__isnull": False})
                    order_key = qp.get("ordering", "avg_return")
                    leaf = {"avg_return": "avg_return", "win_rate": "win_rate",
                            "trades": "trades", "total_trades": "trades"}.get(order_key)
                    if leaf:
                        from django.db.models.functions import Cast
                        from django.db.models.fields.json import KeyTransform, KeyTextTransform
                        val = Cast(KeyTextTransform(leaf, KeyTransform(rkey, rtype)), FloatField())
                        qs = qs.annotate(_regime_sort=val)
                        asc = qp.get("dir", "desc") == "asc"
                        ordering = [F("_regime_sort").asc(nulls_last=True) if asc
                                    else F("_regime_sort").desc(nulls_last=True),
                                    F("id").asc() if asc else F("id").desc()]
            if ordering is None:
                ordering = resolve_ordering(request, STUDY_ORDER, "avg_return")
            # Unfiltered category list so the filter buttons stay complete without all rows loaded.
            categories = sorted(c for c in Study.objects.filter(is_computed=True).order_by()
                                .values_list("category", flat=True).distinct() if c)
            page, next_offset, total = paginate_offset(request, qs, ordering)
            return paged_response(StudySerializer(page, many=True).data, next_offset, total,
                                  last_updated=last_iso,
                                  extra={"total_studies": total, "categories": categories})
        data = StudySerializer(qs, many=True).data
        return Response({"total_studies": len(data), "studies": data, "last_updated": last_iso})


class StudyTradesView(APIView):
    def get(self, request, study_id):
        sig_key = request.query_params.get("signal")
        exit_key = request.query_params.get("exit")
        sector_filter = request.query_params.get("sector")

        # Find study by ID or signal+exit keys
        study = None
        if sig_key and exit_key:
            study = Study.objects.filter(signal_key=sig_key, exit_key=exit_key).first()
        if not study:
            try:
                study = Study.objects.get(id=study_id)
            except Study.DoesNotExist:
                return Response({"error": "Study not found"}, status=404)

        # Check if trades exist in DB
        from django.db.models import Q
        qs = Trade.objects.filter(study=study).select_related("sector")
        if sector_filter:
            qs = qs.filter(Q(etf=sector_filter) | Q(sector__name=sector_filter))
        trades = list(qs.order_by("-entry_date"))  # single fetch; avoids a separate COUNT on the hit path

        if trades:
            return Response({
                "study_id": study.id,
                "study_name": study.name,
                "total_trades": len(trades),
                "trades": TradeSerializer(trades, many=True).data,
            })

        # Compute on-the-fly using the studies module
        try:
            import studies as studies_mod
            if study.signal_key not in studies_mod.SIGNALS or study.exit_key not in studies_mod.EXITS:
                return Response({"error": "Invalid signal/exit"}, status=400)

            _, sig_fn = studies_mod.SIGNALS[study.signal_key]
            _, exit_fn = studies_mod.EXITS[study.exit_key]

            from api.tasks import _get_df, _get_dfs

            # Load SPY for alpha comparison (vectorized dict build — was a row-by-row iterrows()).
            spy_df = _get_df("SPY")
            spy_close = {}
            if spy_df is not None:
                spy_close = dict(zip(spy_df.index.strftime("%Y-%m-%d"),
                                     spy_df["Close"].astype(float)))

            # Pre-compute regime lookup
            regime_data = {}
            try:
                import rates as rates_mod
                import market_regime
                r = rates_mod.get_rates()
                m = market_regime.get_market_data()
                if len(r) > 0:
                    regime_data['regime'] = r["regime"].astype(str) if "regime" in r.columns else None
                    regime_data['curve'] = r["curve"].astype(str) if "curve" in r.columns else None
                if len(m) > 0:
                    regime_data['vix'] = m["vix_regime"].astype(str) if "vix_regime" in m.columns else None
                    regime_data['spy'] = m["spy_trend"].astype(str) if "spy_trend" in m.columns else None
            except Exception:
                pass

            def _get_regime(date):
                res = {}
                for key, series in regime_data.items():
                    if series is not None:
                        match = series[series.index <= date]
                        if len(match) > 0:
                            res[key] = str(match.iloc[-1])
                month = date.month if hasattr(date, 'month') else int(str(date)[5:7])
                res['season'] = 'NOV_APR' if month >= 11 or month <= 4 else 'MAY_OCT'
                return res

            trades = []
            sectors = list(Sector.objects.all())
            if sector_filter:
                sectors = [s for s in sectors if s.etf == sector_filter or s.name == sector_filter]
            # Bulk-load every sector ETF in ONE query (was an N+1: _get_df per sector).
            sector_dfs = _get_dfs([s.etf for s in sectors])

            for sector in sectors:
                df = sector_dfs.get(sector.etf)
                if df is None or len(df) < 60:
                    continue
                try:
                    signals = sig_fn(df).fillna(False)
                except Exception:
                    continue

                close = df["Close"].values
                n = len(close)
                for entry_date in signals[signals].index:
                    idx = df.index.get_loc(entry_date)
                    exit_idx = exit_fn(df, idx)
                    if exit_idx is None or exit_idx <= idx or exit_idx >= n:
                        continue
                    ep = float(close[idx])
                    if ep <= 0:
                        continue
                    xp = float(close[exit_idx])
                    ret = (xp - ep) / ep * 100

                    # Detect ongoing: exit is last bar or hit max_hold boundary
                    ongoing = (exit_idx >= n - 1) or (exit_idx - idx >= 60 and exit_idx == min(idx + 90, n - 1))

                    # Per-trade peak, drawdown, and 90d return
                    max_look = min(idx + 90, n - 1)
                    peak_ret = None   # true path max, incl. all-declining trades (not floored at 0)
                    peak_day = 0
                    max_drawdown = 0
                    running_peak = ep
                    for d in range(1, exit_idx - idx + 1):
                        p = float(close[idx + d])
                        r = (p - ep) / ep * 100
                        if peak_ret is None or r > peak_ret:
                            peak_ret = r
                            peak_day = d
                        if p > running_peak:
                            running_peak = p
                        dd = (p - running_peak) / running_peak * 100
                        if dd < max_drawdown:
                            max_drawdown = dd
                    for d in range(exit_idx - idx + 1, max_look - idx + 1):
                        r = (float(close[idx + d]) - ep) / ep * 100
                        if peak_ret is None or r > peak_ret:
                            peak_ret = r
                            peak_day = d
                    if peak_ret is None:
                        peak_ret = 0.0
                    ret_90d = None
                    if idx + 90 < n:
                        ret_90d = round((float(close[idx + 90]) - ep) / ep * 100, 2)

                    # SPY return for same period
                    entry_d = str(entry_date)[:10]
                    exit_d = str(df.index[exit_idx])[:10]
                    spy_entry = spy_close.get(entry_d)
                    spy_exit = spy_close.get(exit_d)
                    spy_ret = round((spy_exit - spy_entry) / spy_entry * 100, 3) if spy_entry and spy_exit and spy_entry > 0 else None
                    alpha = round(ret - spy_ret, 3) if spy_ret is not None else None

                    trade_rec = {
                        "sector": sector.name,
                        "etf": sector.etf,
                        "entry_date": entry_d,
                        "exit_date": exit_d,
                        "entry_price": round(ep, 2),
                        "exit_price": round(xp, 2),
                        "return_pct": round(ret, 3),
                        "hold_days": exit_idx - idx,
                        "peak_day": peak_day,
                        "peak_ret": round(peak_ret, 2),
                        "ret_90d": ret_90d,
                        "max_drawdown": round(max_drawdown, 2),
                        "ongoing": ongoing,
                        "spy_ret": spy_ret,
                        "alpha": alpha,
                    }
                    trade_rec.update(_get_regime(entry_date))
                    trades.append(trade_rec)

            trades.sort(key=lambda x: x["entry_date"], reverse=True)
            return Response({
                "study_id": study.id,
                "study_name": study.name,
                "total_trades": len(trades),
                "trades": trades,
            })
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class RegimeView(APIView):
    def get(self, request):
        """Return current market regime data."""
        try:
            import rates as rates_mod
            import market_regime
            r = rates_mod.get_rates()
            m = market_regime.get_market_data()
            last_r = r.iloc[-1] if len(r) > 0 else {}
            last_m = m.iloc[-1] if len(m) > 0 else {}
            return Response({
                "rate_3m": round(float(last_r.get("rate_3m", 0)), 2),
                "rate_10y": round(float(last_r.get("rate_10y", 0)), 2),
                "regime": str(last_r.get("regime", "?")),
                "curve": str(last_r.get("curve", "?")),
                "rate_trend": str(last_r.get("rate_trend", "?")),
                "vix": round(float(last_m.get("vix", 0)), 1),
                "vix_regime": str(last_m.get("vix_regime", "?")),
                "spy_trend": str(last_m.get("spy_trend", "?")),
                "season": str(last_m.get("sell_in_may", "?")),
            })
        except Exception as e:
            return Response({"error": str(e)})


class RegimeHistoryView(APIView):
    def get(self, request):
        """Return historical regime data + SPY for documentation charts."""
        try:
            import rates as rates_mod
            import market_regime
            import data_fetcher
            import numpy as np

            r = rates_mod.get_rates()
            m = market_regime.get_market_data()
            spy_data = data_fetcher.fetch_tickers(["SPY"], "5y", "1d")
            spy_df = spy_data.get("SPY")

            points = []
            if spy_df is not None and len(r) > 0:
                # Merge rate data with SPY
                for i, (date, row) in enumerate(spy_df.iterrows()):
                    d = str(date)[:10]
                    rec = {
                        "date": d,
                        "spy": round(float(row["Close"]), 2),
                    }
                    # Find matching rate data
                    rate_row = r[r.index <= date]
                    if len(rate_row) > 0:
                        lr = rate_row.iloc[-1]
                        rec["rate_3m"] = round(float(lr.get("rate_3m", 0)), 2) if lr.get("rate_3m") == lr.get("rate_3m") else None
                        rec["rate_10y"] = round(float(lr.get("rate_10y", 0)), 2) if lr.get("rate_10y") == lr.get("rate_10y") else None
                        rec["regime"] = str(lr.get("regime", ""))
                        rec["curve"] = str(lr.get("curve", ""))

                    # Find matching market data
                    mkt_row = m[m.index <= date]
                    if len(mkt_row) > 0:
                        lm = mkt_row.iloc[-1]
                        vix_val = lm.get("vix", 0)
                        rec["vix"] = round(float(vix_val), 1) if vix_val == vix_val else None
                        rec["vix_regime"] = str(lm.get("vix_regime", ""))
                        rec["spy_trend"] = str(lm.get("spy_trend", ""))

                    # Season
                    month = date.month if hasattr(date, 'month') else int(d[5:7])
                    rec["season"] = "NOV_APR" if month >= 11 or month <= 4 else "MAY_OCT"

                    points.append(rec)

            return Response({"data": points})
        except Exception as e:
            return Response({"error": str(e)})


class DarkPoolView(APIView):
    """Off-exchange / dark-pool activity. With ?ticker= : the daily Polygon blended off-% series
    (DarkPoolDay) + the official weekly FINRA ATS off-% series (DarkPoolWeek, keyed by publish date)
    for the overlay chart. Without a ticker: a market snapshot (top tickers by latest daily off-%) +
    the 'Dark-pool share' amplifier result from the stock-study sweep (null until the sweep populates
    it)."""

    def get(self, request):
        from core.models import DarkPoolDay, DarkPoolWeek, StockStudy
        from django.db.models import Max
        tk = (request.GET.get("ticker") or "").upper().strip()
        if tk:
            daily = list(DarkPoolDay.objects.filter(ticker=tk, off_pct__isnull=False)
                         .order_by("date").values("date", "off_pct", "total_vol", "off_vol"))
            weekly = list(DarkPoolWeek.objects.filter(ticker=tk, off_pct__isnull=False,
                          source="finra_ats").order_by("week_start")
                          .values("week_start", "off_pct", "ats_shares", "published_date"))
            for d in daily:
                d["date"] = str(d["date"])
            for w in weekly:
                w["week_start"] = str(w["week_start"])
                w["published_date"] = str(w["published_date"]) if w["published_date"] else None
            return Response({
                "ticker": tk, "daily": daily, "weekly": weekly,
                "daily_last": daily[-1] if daily else None,
                "weekly_last": weekly[-1] if weekly else None,
            })

        # Market snapshot: latest daily-file date, top tickers by off-%.
        latest = DarkPoolDay.objects.aggregate(m=Max("date"))["m"]
        snap = []
        if latest:
            snap = list(DarkPoolDay.objects.filter(date=latest, off_pct__isnull=False,
                        total_vol__gt=0).order_by("-off_pct")
                        .values("ticker", "off_pct", "total_vol")[:40])
            for s in snap:
                s["date"] = str(latest)
        # Amplifier: does high dark-pool share amplify a robust signal's edge? Pull the
        # 'Dark-pool share' dimension buckets from the highest-sample stock study that has them.
        amp = None
        for s in (StockStudy.objects.filter(total_trades__gte=500)
                  .order_by("-total_trades").values("signal_key", "exit_key", "avg_return",
                                                     "win_rate", "total_trades", "by_dimension")[:200]):
            bd = s.get("by_dimension") or {}
            # by_dimension buckets are a LIST of {bucket, avg_return, win_rate, t_stat, trades, ...}
            # per dimension. Transform each dimension to {bucket_label: {...}} for the UI.
            def _to_map(dim):
                lst = bd.get(dim)
                if not isinstance(lst, list) or not lst:
                    return None
                return {b["bucket"]: {"avg_return": b.get("avg_return"), "win_rate": b.get("win_rate"),
                                      "total_trades": b.get("trades"), "t_stat": b.get("t_stat")}
                        for b in lst if isinstance(b, dict) and "bucket" in b}
            share = _to_map("Dark-pool share")
            if share:
                amp = {"signal": s["signal_key"], "exit": s["exit_key"],
                       "base_avg_return": s["avg_return"], "base_win_rate": s["win_rate"],
                       "base_trades": s["total_trades"], "buckets": share,
                       "trend_buckets": _to_map("Dark-pool trend")}
                break
        return Response({
            "snapshot": snap, "snapshot_date": str(latest) if latest else None,
            "amplifier": amp,
            "finra_last_week": str(DarkPoolWeek.objects.aggregate(m=Max("week_start"))["m"] or ""),
            "note": ("Polygon = daily blended off-exchange proxy (ATS+internalizers); FINRA = official "
                     "weekly ATS only, ~2-4wk publication lag."),
        })


class FundamentalsView(APIView):
    def get(self, request, ticker):
        """Return fundamental data for a ticker."""
        from core.models import Fundamental
        latest = Fundamental.objects.filter(ticker=ticker.upper()).order_by('-date').first()
        if latest is None:
            return Response({"error": "No fundamentals found"}, status=404)
        fields = {}
        for f in latest._meta.get_fields():
            if f.name in ('id',):
                continue
            val = getattr(latest, f.name, None)
            if val is not None:
                fields[f.name] = val
        # Convert date
        if 'date' in fields:
            fields['date'] = str(fields['date'])
        return Response(fields)


class FundamentalsListView(APIView):
    def get(self, request):
        """Return latest fundamentals for all tickers."""
        from core.models import Fundamental
        FIELDS = ['ticker', 'date', 'eps', 'forward_eps', 'pe_ratio', 'forward_pe', 'market_cap',
                  'dividend_yield', 'revenue_growth', 'profit_margin', 'debt_to_equity',
                  'short_ratio', 'short_pct_float', 'insider_pct', 'institution_pct',
                  'analyst_rating', 'analyst_target', 'float_shares', 'shares_outstanding',
                  'beta_5y', 'sector', 'industry', 'free_cash_flow', 'book_value',
                  'analyst_strong_buy', 'analyst_buy', 'analyst_hold', 'analyst_sell',
                  'analyst_strong_sell', 'analyst_rating_mean', 'cusip', 'cik', 'isin']
        # ONE query, newest-first, keep the first (latest) row per ticker. Was an N+1: a group-by
        # for the latest date, then a .first() per ticker → ~1000 round-trips per request.
        import math
        result = []
        seen = set()
        for row in Fundamental.objects.order_by('ticker', '-date').values(*FIELDS):
            tk = row['ticker']
            if tk in seen:
                continue
            seen.add(tk)
            row['date'] = str(row['date'])
            # Sanitize non-finite floats (inf/-inf/nan) — some stored ratios are inf and would
            # raise "Out of range float values are not JSON compliant" on serialization.
            for k, v in row.items():
                if isinstance(v, float) and not math.isfinite(v):
                    row[k] = None
            result.append(row)
        # Fundamental has no computed/updated column; the report `date` is the freshness proxy.
        from django.db.models import Max
        last = Fundamental.objects.aggregate(m=Max('date'))['m']
        return Response({'fundamentals': result, 'total': len(result),
                         'last_updated': str(last) if last else None})


class TrendStudyListView(APIView):
    def get(self, request):
        """Return all trend study results. Optional ?mode=etf|momentum|hibeta filter."""
        # defer the big JSON blobs (equity_curve/spy_curve/trade_log ~1250 pts + full trade log each) —
        # the list loop only reads scalars, so pulling them was multi-MB fetched and discarded per request.
        studies = TrendStudy.objects.defer("equity_curve", "spy_curve", "trade_log")
        mode = request.GET.get("mode")
        if mode:
            studies = studies.filter(hold_mode=mode)
        # .order_by() clears the model's default ordering, else it pollutes DISTINCT.
        modes = sorted(TrendStudy.objects.order_by().values_list("hold_mode", flat=True).distinct())
        from django.db.models import Max
        last = TrendStudy.objects.aggregate(m=Max('computed_at'))['m']
        last_iso = last.isoformat() if last else None

        def _row(s):
            return {
                "id": s.id,
                "lookback_months": s.lookback_months,
                "hold_months": s.hold_months,
                "top_n": s.top_n,
                "hold_mode": s.hold_mode,
                "total_return": s.total_return,
                "annual_return": s.annual_return,
                "spy_total": s.spy_total,
                "alpha": s.alpha,
                "max_drawdown": s.max_drawdown,
                "num_trades": s.num_trades,
                "win_rate": s.win_rate,
                "t_stat": s.t_stat,
                "robust": bool(s.num_trades >= 12 and s.t_stat is not None and abs(s.t_stat) >= 2),
            }

        if request.query_params.get("paginate") == "1" or request.query_params.get("offset") is not None:
            from api.pagination import resolve_ordering, paginate_offset, paged_response
            TREND_ORDER = {"total_return": "total_return", "annual_return": "annual_return",
                           "vs_spy": "alpha", "alpha": "alpha", "max_drawdown": "max_drawdown",
                           "num_trades": "num_trades", "periods": "num_trades",
                           "win_rate": "win_rate", "t_stat": "t_stat", "sharpe": "t_stat",
                           "hold_months": "hold_months", "lookback_months": "lookback_months"}
            ordering = resolve_ordering(request, TREND_ORDER, "total_return")
            page, next_offset, total = paginate_offset(request, studies, ordering)
            return paged_response([_row(s) for s in page], next_offset, total,
                                  last_updated=last_iso, extra={"total": total, "modes": modes})

        data = [_row(s) for s in studies]
        return Response({"total": len(data), "strategies": data, "modes": modes,
                         "last_updated": last_iso})


class TrendStudyDetailView(APIView):
    def get(self, request, study_id):
        """Return a single trend study with full equity curve and trade log."""
        try:
            s = TrendStudy.objects.get(id=study_id)
        except TrendStudy.DoesNotExist:
            return Response({"error": "Not found"}, status=404)
        return Response({
            "id": s.id,
            "lookback_months": s.lookback_months,
            "hold_months": s.hold_months,
            "top_n": s.top_n,
            "hold_mode": s.hold_mode,
            "total_return": s.total_return,
            "annual_return": s.annual_return,
            "spy_total": s.spy_total,
            "alpha": s.alpha,
            "max_drawdown": s.max_drawdown,
            "num_trades": s.num_trades,
            "win_rate": s.win_rate,
            "equity_curve": s.equity_curve,
            "spy_curve": s.spy_curve,
            "trade_log": s.trade_log,
        })


class StockDrilldownListView(APIView):
    def get(self, request):
        """Return stock drilldown results for top indicator studies."""
        drilldowns = StockDrilldown.objects.select_related('study').all()
        data = []
        for d in drilldowns:
            data.append({
                "id": d.id,
                "study_id": d.study.id,
                "study_name": d.study.name,
                "signal": d.study.signal_key,
                "signal_name": d.study.signal_name,
                "exit": d.study.exit_key,
                "exit_name": d.study.exit_name,
                "category": d.study.category,
                "etf_avg_return": d.etf_avg_return,
                "stock_trades": d.stock_trades,
                "stock_avg_return": d.stock_avg_return,
                "stock_win_rate": d.stock_win_rate,
                "alpha_vs_etf": d.alpha_vs_etf,
                "stock_max_drawdown": d.stock_max_drawdown,
                "best_stocks": d.best_stocks,
                "worst_stocks": d.worst_stocks,
            })
        from django.db.models import Max
        last = StockDrilldown.objects.aggregate(m=Max('computed_at'))['m']
        return Response({"total": len(data), "drilldowns": data,
                         "last_updated": last.isoformat() if last else None})


class RefreshView(APIView):
    def get(self, request):
        ScanResult.objects.all().delete()
        return Response({"status": "cleared"})


class ImportCandlesView(APIView):
    def post(self, request):
        """Trigger Yahoo Finance data import in background."""
        from api.tasks import import_candles_task
        thread = threading.Thread(target=import_candles_task, daemon=True)
        thread.start()
        return Response({"status": "import started"})


class RunStudiesView(APIView):
    def post(self, request):
        """Trigger study computation in background (multithreaded)."""
        from api.tasks import run_studies_task
        thread = threading.Thread(target=run_studies_task, daemon=True)
        thread.start()
        return Response({"status": "studies started"})


import json as _json
import os as _os

STOCK_STUDIES_PATH = "/app/.data/studies/stock_studies_all.json"


class StockStudiesView(APIView):
    """Serve the all-on-all stock studies sweep (every signal × exit over ~1035 stocks,
    with fundamental-bucket breakdowns). GET reads the JSON the sweep writes; POST
    triggers a fresh sweep in the background."""

    def get(self, request):
        from core.models import StockStudy
        p = request.query_params
        category = p.get("category")
        signal = p.get("signal")
        exit_key = p.get("exit")
        search = (p.get("search") or "").strip()
        try:
            min_trades = int(p.get("min_trades")) if p.get("min_trades") else None
        except ValueError:
            min_trades = None
        try:
            limit = int(p.get("limit", 500))
        except (TypeError, ValueError):
            limit = 500

        # Which dimensions are point-in-time vs snapshot-only (for UI badging). Derived
        # from the single source of truth (DIMENSIONS); {} if the module can't import.
        try:
            from seq_fundamental_study import DIMENSIONS
            dimension_meta = {d[0]: {"pit": bool(d[4])} for d in DIMENSIONS}
        except Exception:
            dimension_meta = {}

        # Prefer Postgres (StockStudy); fall back to the JSON cache if the table is empty.
        qs = StockStudy.objects.all()
        n_results = qs.count()  # single query; also serves as the "is it computed" check
        if n_results:
            # DB-level DISTINCT (don't pull 23k category strings into Python). Clear the
            # model's default -avg_return ordering so DISTINCT keys on category alone.
            categories = sorted(qs.order_by().values_list("category", flat=True).distinct())
            # Metadata from one lightweight row (values(), not a full model instance).
            meta = qs.order_by("-computed_at").values("universe_size", "computed_at").first()
            fq = qs
            if category:
                fq = fq.filter(category=category)
            if signal:
                fq = fq.filter(signal_key=signal)
            if exit_key:
                fq = fq.filter(exit_key=exit_key)
            if min_trades:
                fq = fq.filter(total_trades__gte=min_trades)
            if search:
                from django.db.models import Q as _Q
                fq = fq.filter(_Q(signal_name__icontains=search) | _Q(exit_name__icontains=search))
            computed_at = meta["computed_at"] if meta else None
            _last_iso = computed_at.isoformat() if computed_at else None

            # Live 'firing now' count per signal (signal_firing_scan.py) — the 3d firing indicator.
            from core.models import SignalFiring
            _firemap = dict(SignalFiring.objects.values_list("signal_key", "n_firing"))

            def _shape(r):
                return {
                    "signal_key": r["signal_key"], "signal_name": r["signal_name"],
                    "exit_key": r["exit_key"], "exit_name": r["exit_name"],
                    "category": r["category"], "trades": r["total_trades"],
                    "eff_trades": r["eff_trades"], "t_stat": r["t_stat"],
                    "avg_return": r["avg_return"], "win_rate": r["win_rate"],
                    "avg_hold": r["avg_hold"], "avg_mae": r["avg_mae"], "clean_pct": r["clean_pct"],
                    "by_dimension": r["by_dimension"] or {},
                    "n_firing": _firemap.get(r["signal_key"], 0),
                }
            _STOCK_VALS = ("id", "signal_key", "signal_name", "exit_key", "exit_name", "category",
                           "total_trades", "eff_trades", "t_stat", "avg_return", "win_rate",
                           "avg_hold", "avg_mae", "clean_pct", "by_dimension")
            # Opt-in offset pagination (infinite-scroll frontend); legacy full list otherwise.
            if request.query_params.get("paginate") == "1" or request.query_params.get("offset") is not None:
                from api.pagination import resolve_ordering, paginate_offset, paged_response
                STOCK_ORDER = {"signal_name": "signal_name", "exit_name": "exit_name",
                               "category": "category", "avg_return": "avg_return",
                               "win_rate": "win_rate", "trades": "total_trades",
                               "t_stat": "t_stat", "avg_hold": "avg_hold",
                               "avg_mae": "avg_mae", "clean_pct": "clean_pct"}
                ordering = resolve_ordering(request, STOCK_ORDER, "avg_return")
                page, next_offset, total = paginate_offset(request, fq.values(*_STOCK_VALS), ordering)
                return paged_response([_shape(r) for r in page], next_offset, total,
                                      last_updated=_last_iso, extra={
                    "computed": True, "source": "db",
                    "universe_size": meta["universe_size"] if meta else None,
                    "n_results": n_results, "categories": categories,
                    "computed_at": _last_iso, "dimension_meta": dimension_meta})
            # .values() → dicts, skips model instantiation for the (up to `limit`) rows.
            rows = fq.order_by("-avg_return").values(
                "signal_key", "signal_name", "exit_key", "exit_name", "category",
                "total_trades", "eff_trades", "t_stat",
                "avg_return", "win_rate", "avg_hold", "avg_mae", "clean_pct",
                "by_dimension")[:limit]
            results = [{
                "signal_key": r["signal_key"], "signal_name": r["signal_name"],
                "exit_key": r["exit_key"], "exit_name": r["exit_name"],
                "category": r["category"], "trades": r["total_trades"],
                "eff_trades": r["eff_trades"], "t_stat": r["t_stat"],
                "avg_return": r["avg_return"], "win_rate": r["win_rate"],
                "avg_hold": r["avg_hold"], "avg_mae": r["avg_mae"], "clean_pct": r["clean_pct"],
                "by_dimension": r["by_dimension"] or {},
            } for r in rows]
            computed_at = meta["computed_at"] if meta else None
            return Response({
                "computed": True, "source": "db",
                "universe_size": meta["universe_size"] if meta else None,
                "n_signals": None, "n_exits": None,
                "n_results": n_results, "categories": categories,
                "computed_at": computed_at.isoformat() if computed_at else None,
                "last_updated": computed_at.isoformat() if computed_at else None,
                "returned": len(results), "results": results,
                "dimension_meta": dimension_meta,
            })

        # ── JSON fallback ──
        if not _os.path.exists(STOCK_STUDIES_PATH):
            return Response({
                "computed": False, "results": [], "n_results": 0,
                "message": "Stock studies not computed yet. POST to this endpoint to run the sweep.",
            })
        with open(STOCK_STUDIES_PATH) as f:
            payload = _json.load(f)
        results = payload.get("results", [])
        if category:
            results = [r for r in results if r.get("category") == category]
        if signal:
            results = [r for r in results if r.get("signal_key") == signal]
        if exit_key:
            results = [r for r in results if r.get("exit_key") == exit_key]
        if min_trades:
            results = [r for r in results if r.get("trades", 0) >= min_trades]
        categories = sorted({r.get("category", "Other") for r in payload.get("results", [])})
        computed_at = None
        try:
            computed_at = timezone.datetime.fromtimestamp(
                _os.path.getmtime(STOCK_STUDIES_PATH), tz=timezone.get_current_timezone()
            ).isoformat()
        except Exception:
            pass
        return Response({
            "computed": True, "source": "json",
            "universe_size": payload.get("universe_size"),
            "n_signals": payload.get("n_signals"), "n_exits": payload.get("n_exits"),
            "n_results": payload.get("n_results"), "categories": categories,
            "computed_at": computed_at, "last_updated": computed_at,
            "returned": min(len(results), limit), "results": results[:limit],
            "next": None,  # JSON fallback is single-page (DB-empty case only)
            "dimension_meta": dimension_meta,
        })

    def post(self, request):
        """Kick off a fresh all-on-all sweep in the background."""
        from api.tasks import run_stock_studies_task
        thread = threading.Thread(target=run_stock_studies_task, daemon=True)
        thread.start()
        return Response({"status": "stock studies sweep started"})


class LiveSignalsView(APIView):
    """Serve the 'firing now' scan: stocks currently triggering a top signal, with the
    signal's historical edge + fundamentals + sector. GET reads LiveSignal (server-side
    filters); POST triggers a fresh scan in the background."""

    def get(self, request):
        from core.models import LiveSignal
        p = request.query_params
        qs = LiveSignal.objects.all()
        n_total = qs.count()
        if not n_total:
            return Response({"computed": False, "results": [], "message":
                             "No firing scan yet. POST to run it (needs stock studies first)."})
        signal = p.get("signal")
        sector = p.get("sector")
        try:
            max_days = int(p["max_days"]) if p.get("max_days") else None
        except ValueError:
            max_days = None
        try:
            limit = int(p.get("limit", 500))
        except (TypeError, ValueError):
            limit = 500
        if signal:
            qs = qs.filter(signal_key=signal)
        if max_days is not None:
            qs = qs.filter(days_ago__lte=max_days)
        if sector:
            qs = qs.filter(sectors__contains=sector)
        signals = sorted(LiveSignal.objects.order_by().values_list("signal_key", flat=True).distinct())
        meta = LiveSignal.objects.order_by("-computed_at").values("computed_at").first()
        computed_at = meta["computed_at"] if meta else None
        _last_iso = computed_at.isoformat() if computed_at else None
        import math
        def _finite(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        _VALS = ("ticker", "signal_key", "signal_name", "days_ago", "last_close",
                 "best_exit_key", "hist_avg_return", "hist_win_rate", "hist_trades",
                 "hist_avg_mae", "hist_clean_pct",
                 "market_cap", "pe_ratio", "forward_pe", "profit_margin", "fund_buckets", "sectors",
                 "insider_buy_90d", "recent_13d", "recent_13g")
        # Opt-in offset pagination (infinite-scroll frontend); legacy full list otherwise.
        if request.query_params.get("paginate") == "1" or request.query_params.get("offset") is not None:
            from api.pagination import resolve_ordering, paginate_offset, paged_response
            LIVE_ORDER = {"ticker": "ticker", "signal_name": "signal_name", "days_ago": "days_ago",
                          "last_close": "last_close", "hist_avg_return": "hist_avg_return",
                          "hist_win_rate": "hist_win_rate", "hist_trades": "hist_trades",
                          "hist_avg_mae": "hist_avg_mae", "hist_clean_pct": "hist_clean_pct",
                          "market_cap": "market_cap", "pe_ratio": "pe_ratio",
                          "forward_pe": "forward_pe", "profit_margin": "profit_margin"}
            ordering = resolve_ordering(request, LIVE_ORDER, "days_ago", default_dir="asc")
            page, next_offset, total = paginate_offset(request, qs.values("id", *_VALS), ordering)
            clean = [{k: _finite(v) for k, v in r.items() if k != "id"} for r in page]
            return paged_response(clean, next_offset, total, last_updated=_last_iso,
                                  extra={"computed": True, "n_firing": n_total, "signals": signals,
                                         "computed_at": _last_iso})
        rows = qs.order_by("days_ago", "-hist_avg_return").values(*_VALS)[:limit]
        clean = [{k: _finite(v) for k, v in r.items()} for r in rows]
        return Response({
            "computed": True, "n_firing": n_total, "signals": signals,
            "computed_at": _last_iso, "last_updated": _last_iso,
            "returned": min(n_total, limit), "results": clean,
        })

    def post(self, request):
        from api.tasks import run_live_firing_task
        thread = threading.Thread(target=run_live_firing_task, daemon=True)
        thread.start()
        return Response({"status": "firing scan started"})


class NewsEventStudyView(APIView):
    """Market-adjusted news event study (keyed on OUR model's read, not EODHD sentiment): abnormal
    returns AR = R_stock − beta·R_spy per (dir×beta 252d/60d), (dir×impact), (IV regime×dir), and
    (news_type×beta), with next-morning gap + IV surprise ratio. GET reads the stored blob; POST
    recomputes in the background."""

    def get(self, request):
        from core.models import NewsEventStudy
        row = (NewsEventStudy.objects.filter(label="latest")
               .values("n_events", "n_tickers", "data", "computed_at").first())
        if not row or not row["data"]:
            return Response({"computed": False, "results": {},
                             "message": "Not computed yet. POST to run (needs classified news)."})
        return Response({
            "computed": True, "n_events": row["n_events"], "n_tickers": row["n_tickers"],
            "computed_at": row["computed_at"].isoformat() if row["computed_at"] else None,
            "groupings": row["data"],
        })

    def post(self, request):
        from api.news_market_study import run_and_save
        threading.Thread(target=run_and_save, daemon=True).start()
        return Response({"status": "news event study started"})


class IvCalibrationView(APIView):
    """IV calibration: is ATM implied vol a good predictor of the next-day move? Aggregate
    variance-risk-premium stats + per-ticker ranking of over/under-priced options. GET reads the
    stored blob; POST recomputes in the background."""

    def get(self, request):
        from core.models import IvCalibration
        row = IvCalibration.objects.filter(label="latest").values("data", "computed_at").first()
        if not row or not row["data"]:
            return Response({"computed": False, "aggregate": {}, "per_ticker": [],
                             "message": "Not computed yet. POST to run."})
        d = row["data"]
        return Response({
            "computed": True,
            "computed_at": row["computed_at"].isoformat() if row["computed_at"] else None,
            "aggregate": d.get("aggregate", {}), "per_ticker": d.get("per_ticker", []),
            "n_tickers": d.get("n_tickers", 0),
        })

    def post(self, request):
        from api.iv_calibration import run_and_save
        threading.Thread(target=run_and_save, daemon=True).start()
        return Response({"status": "iv calibration started"})


class NewsHorizonSignalsView(APIView):
    """Serve the news-horizon scan: recent material news joined to the horizon-conditioned drift
    we measured for its TYPE. Only the FADES are robustness-validated (earnings-beat / product /
    strong-bullish-pop, mid/small caps); the rest are WATCH. GET reads NewsHorizonSignal
    (server-side filters); POST triggers a fresh scan in the background."""

    def get(self, request):
        from core.models import NewsHorizonSignal
        import math
        p = request.query_params
        qs = NewsHorizonSignal.objects.all()
        n_total = qs.count()
        if not n_total:
            return Response({"computed": False, "results": [], "message":
                             "No news-horizon scan yet. POST to run it (needs classified news)."})
        if p.get("stance"):
            qs = qs.filter(stance=p["stance"].upper())
        if p.get("robust") in ("1", "true", "True"):
            qs = qs.filter(robust=True)
        if p.get("cat"):
            qs = qs.filter(cat=p["cat"])
        if p.get("horizon"):
            qs = qs.filter(horizon=p["horizon"])
        if p.get("sector"):
            qs = qs.filter(sectors__contains=p["sector"])
        try:
            limit = int(p.get("limit", 500))
        except (TypeError, ValueError):
            limit = 500
        cats = sorted(NewsHorizonSignal.objects.order_by().values_list("cat", flat=True).distinct())
        meta = NewsHorizonSignal.objects.order_by("-computed_at").values("computed_at").first()
        rows = qs.order_by("-robust", "days_left", "-impact").values(
            "ticker", "news_date", "cat", "direction", "impact", "horizon", "pop_pct",
            "market_cap", "cap_bucket", "exp_drift", "stance", "robust", "days_since", "days_left",
            "last_close", "title", "sectors")[:limit]
        def _finite(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        clean = [{k: _finite(v) for k, v in r.items()} for r in rows]
        return Response({
            "computed": True, "n_total": n_total, "n_fade": qs.filter(stance="FADE").count(),
            "cats": cats, "computed_at": meta["computed_at"].isoformat() if meta and meta["computed_at"] else None,
            "returned": len(clean), "results": clean,
        })

    def post(self, request):
        from api.tasks import run_news_horizon_scan_task
        thread = threading.Thread(target=run_news_horizon_scan_task, daemon=True)
        thread.start()
        return Response({"status": "news-horizon scan started"})


def _signed_cat(cat, direction):
    """Fold the local UNSIGNED category (cat_llm) + local_dir into the signed labels users expect in
    the 'News type' column (earnings_beat/miss, guidance_up/down, upgrade/downgrade). dir>=0 -> bullish
    variant, dir<0 -> bearish. Mirrors news_horizon_scan._signed_cat (kept local to avoid importing
    that module's heavy candle deps into the web process)."""
    c = (cat or "other").lower(); d = direction or 0
    if c == "earnings":    return "earnings_beat" if d >= 0 else "earnings_miss"
    if c == "guidance":    return "guidance_up" if d >= 0 else "guidance_down"
    if c == "analyst":     return "upgrade" if d >= 0 else "downgrade"
    if c in ("offering", "buyback"): return "capital"
    if c == "partnership": return "contract"
    if c == "insider":     return "other"
    return c


class NewsEffectView(APIView):
    """Browse the classified news corpus with its same-day PRICE-EFFECT columns (day_abn = β-adj
    abnormal move over the news's reaction session, incl. overnight/pre-market gap; day_effect =
    moved >=2σ that day; day_suspect = bad-candle / illiquid-OTC artifact, force-excluded from
    effect). Server-side filtered + paginated. GET only (data is computed by compute_news_effect.py)."""

    def get(self, request):
        from core.models import NewsItem
        from django.db.models import Q
        from django.db.models.functions import Abs
        import math
        p = request.query_params

        def flag(name, default=False):
            v = p.get(name)
            if v is None:
                return default
            return str(v).lower() in ("1", "true", "yes", "on")

        qs = NewsItem.objects.filter(day_abn__isnull=False)
        # defaults chosen to show ONLY real events first (user: "filter out all the junk news, all
        # the news where nothing happened"): moved that day, not editorial junk, not a bad candle.
        if flag("effect", True):
            qs = qs.filter(day_effect=True)
        if flag("hide_junk", True):
            qs = qs.exclude(junk=True)
        if flag("hide_suspect", True):
            qs = qs.exclude(day_suspect=True)
        # off_ticker = the LLM judged the headline is NOT about this ticker (macro / different company
        # in this feed). Dropped by default — functionally "the ticker was removed for this item".
        if flag("hide_offticker", True):
            qs = qs.exclude(off_ticker=True)
        # "rated" now means LOCAL-LLM rated (local_rating) — the Anthropic llm_* layer is fully retired.
        if flag("classified", True):
            qs = qs.filter(local_rating__isnull=False)
        try:
            min_impact = int(p.get("min_impact", 0))
        except (TypeError, ValueError):
            min_impact = 0
        if min_impact:
            qs = qs.filter(local_impact__gte=min_impact)
        if p.get("ticker"):
            qs = qs.filter(ticker=p["ticker"].strip().upper())
        if p.get("cat"):
            qs = qs.filter(cat_llm=p["cat"])
        if p.get("acat"):
            x = p["acat"]  # effective category: prefer the LLM label, fall back to the title heuristic
            qs = qs.filter(Q(cat_llm=x) | (Q(cat_llm="") & Q(cat_auto=x)))
        if p.get("horizon"):
            qs = qs.filter(local_horizon=p["horizon"])
        if p.get("dir") == "up":
            qs = qs.filter(day_abn__gt=0)
        elif p.get("dir") == "down":
            qs = qs.filter(day_abn__lt=0)
        try:
            min_abn = float(p.get("min_abn", 0) or 0)
        except (TypeError, ValueError):
            min_abn = 0.0
        if min_abn > 0:
            qs = qs.filter(Q(day_abn__gte=min_abn) | Q(day_abn__lte=-min_abn))

        sort = p.get("sort", "date")
        if sort == "abn":
            qs = qs.annotate(_absabn=Abs("day_abn")).order_by("-_absabn", "-dt")
        elif sort == "impact":
            qs = qs.order_by("-local_impact", "-dt")
        else:
            qs = qs.order_by("-dt")

        n_total = qs.count()
        n_effect = qs.filter(day_effect=True).count()
        try:
            limit = min(int(p.get("limit", 200)), 1000)
        except (TypeError, ValueError):
            limit = 200
        try:
            offset = max(int(p.get("offset", 0)), 0)
        except (TypeError, ValueError):
            offset = 0

        # distinct categories present (for the filter dropdowns) — computed once, cheap at DB level.
        cats = sorted(c for c in NewsItem.objects.filter(local_rating__isnull=False)
                      .order_by().values_list("cat_llm", flat=True).distinct() if c)
        acats = sorted(c for c in NewsItem.objects.exclude(cat_auto="")
                       .order_by().values_list("cat_auto", flat=True).distinct() if c)
        rows = list(qs.values(
            "ticker", "dt", "title", "local_dir", "local_impact", "local_horizon",
            "local_rating", "day_abn", "day_effect", "day_suspect", "junk", "sentiment", "url",
            "ret_1m", "ret_3m", "ret_1y", "cat_auto", "cat_llm", "off_ticker")[offset:offset + limit])

        # GROUNDED earnings verdict (ground_earnings.py): where a news day maps to an earnings report,
        # show the hard-data verdict (beat_guided_down etc.) so headline sentiment isn't the last word.
        from core.models import EarningsEvent
        from datetime import timedelta as _td
        _ge = {}
        for etk, erd, glab, gsc in (EarningsEvent.objects.filter(
                ticker__in={r["ticker"] for r in rows}).exclude(grounded_label="")
                .values_list("ticker", "report_date", "grounded_label", "grounded_score")):
            _ge.setdefault(etk, []).append((erd, glab, gsc))

        def _f(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        clean = []
        for r in rows:
            nd = r["dt"].date() if r["dt"] else None
            r["grounded_label"], r["grounded_score"] = None, None
            if nd and r["ticker"] in _ge:
                for erd, glab, gsc in _ge[r["ticker"]]:
                    if abs((erd - nd).days) <= 2:      # this news day is at/near the earnings report
                        r["grounded_label"], r["grounded_score"] = glab, gsc
                        break
            r["dt"] = r["dt"].isoformat() if r["dt"] else None
            for k in ("day_abn", "ret_1m", "ret_3m", "ret_1y"):
                r[k] = round(r[k], 2) if r[k] is not None else None
            # Serve the LOCAL labels under the historical llm_* response keys so the frontend needs no
            # change. The Anthropic layer is retired; these ARE the on-box qwen labels now. The 'News
            # type' column shows the SIGNED label derived from cat_llm + local_dir.
            r["llm_cat"] = _signed_cat(r.get("cat_llm"), r.get("local_dir"))
            r["llm_dir"] = r.pop("local_dir")
            r["llm_impact"] = r.pop("local_impact")
            r["llm_horizon"] = r.pop("local_horizon")
            r["llm_rating"] = r.pop("local_rating")
            clean.append({k: _f(v) for k, v in r.items()})
        from django.db.models import Max
        _last = NewsItem.objects.aggregate(m=Max('created_at'))['m']
        return Response({
            "n_total": n_total, "n_effect": n_effect, "cats": cats, "acats": acats,
            "returned": len(clean), "offset": offset, "limit": limit, "results": clean,
            "last_updated": _last.isoformat() if _last else None,
        })


class NewsEffectChartView(APIView):
    """Price series for ONE ticker with every news item plotted as a marker at its reaction bar —
    powers the click-through chart on the News Effect page (user: "make the stock clickable and
    show all the news on a graph"). Each marker carries day_abn / direction / effect / junk /
    suspect / category / title so the frontend can color and label it. GET only."""

    def get(self, request):
        import numpy as np
        import pandas as pd
        from core.models import NewsItem, Candle
        from zoneinfo import ZoneInfo

        ticker = (request.query_params.get("ticker") or "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=400)
        try:
            bars = min(int(request.query_params.get("bars", 1300)), 2000)
        except (TypeError, ValueError):
            bars = 1300

        def flag(name, default=False):
            v = request.query_params.get(name)
            return default if v is None else str(v).lower() in ("1", "true", "yes", "on")

        cds = (Candle.objects.filter(ticker=ticker, interval="1d")
               .order_by("date").values_list("date", "close"))
        cds = list(cds)
        if len(cds) < 5:
            return Response({"error": f"no candles for {ticker}"}, status=404)
        cds = cds[-bars:]
        dates = pd.to_datetime([d for d, _ in cds])
        close = [round(float(c), 4) if c is not None else None for _, c in cds]
        d0 = dates[0]

        ET = ZoneInfo("America/New_York")

        def reaction_date(dt):
            et = pd.Timestamp(dt)
            et = et.tz_convert(ET) if et.tzinfo else et.tz_localize("UTC").tz_convert(ET)
            d = et.normalize().tz_localize(None)
            return d + pd.Timedelta(days=1) if et.hour >= 16 else d

        nqs = NewsItem.objects.filter(ticker=ticker, day_abn__isnull=False, dt__gte=d0)
        # default: only real, non-junk, non-artifact events — same north-star filter as the table.
        if flag("effect", True):
            nqs = nqs.filter(day_effect=True)
        if flag("hide_junk", True):
            nqs = nqs.exclude(junk=True)
        if flag("hide_suspect", True):
            nqs = nqs.exclude(day_suspect=True)
        if flag("classified", True):
            nqs = nqs.filter(local_rating__isnull=False)
        news = nqs.values("dt", "title", "cat_llm", "local_dir", "local_impact",
                          "day_abn", "day_effect", "day_suspect", "junk", "sentiment", "url")

        markers = []
        for it in news:
            rd = reaction_date(it["dt"])
            pos = int(dates.searchsorted(np.datetime64(rd), side="left"))
            if pos >= len(dates):
                pos = len(dates) - 1
            da = it["day_abn"]
            markers.append({
                "idx": pos,
                "dt": it["dt"].isoformat() if it["dt"] else None,
                "title": it["title"],
                "cat": _signed_cat(it["cat_llm"], it["local_dir"]), "impact": it["local_impact"],
                "day_abn": round(da, 2) if da is not None else None,
                "dir": (1 if (da or 0) > 0 else -1 if (da or 0) < 0 else 0),
                "effect": it["day_effect"], "suspect": it["day_suspect"], "junk": it["junk"],
                "sentiment": it["sentiment"], "url": it["url"],
            })
        markers.sort(key=lambda m: m["idx"])
        return Response({
            "ticker": ticker,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "close": close,
            "markers": markers,
            "n_markers": len(markers),
        })


class NewsClusterView(APIView):
    """Detect and analyze NEWS CLUSTERS — bursts of headlines on one ticker in a short window, the
    footprint of a promotion / "propping" campaign (user: "sometimes news come in like a bunch of
    cluster for a specific stock, like someone is propping the stock"). A cluster = a chain of items
    on the same ticker where each is within `gap` days of the previous, with at least `min_items`.

    For each cluster we surface (a) composition — count, bullish vs bearish share, junk/PR share,
    how many actually moved the stock; (b) the net abnormal push DURING the cluster (Σ day_abn); and
    (c) the FORWARD return AFTER the cluster (the last item's ret_1m/3m/1y) — the fade check that tells
    a real story from a pump-and-fizzle. `prop_score` (0-100) is a transparent heuristic combining
    volume + bullish skew + junk/PR share + a positive pop; `faded` flags pump-then-drop. GET only."""

    def get(self, request):
        import math
        from itertools import groupby
        from core.models import NewsItem
        p = request.query_params

        def _int(name, default):
            try:
                return int(p.get(name, default))
            except (TypeError, ValueError):
                return default

        gap = max(_int("gap", 3), 1)              # max quiet gap (days) that still chains a cluster
        min_items = max(_int("min_items", 4), 2)  # a burst needs at least this many headlines
        limit = min(_int("limit", 200), 1000)
        include_junk = str(p.get("include_junk", "1")).lower() in ("1", "true", "yes", "on")
        tick = (p.get("ticker") or "").strip().upper()

        qs = NewsItem.objects.all()
        if not include_junk:
            qs = qs.exclude(junk=True)
        # off-ticker items aren't really this ticker's news — they'd fabricate phantom "propping"
        # clusters (a macro selloff would look like a burst on every name). Always excluded here.
        qs = qs.exclude(off_ticker=True)
        if tick:
            qs = qs.filter(ticker=tick)
        rows = qs.order_by("ticker", "dt").values_list(
            "ticker", "dt", "title", "cat_llm", "local_dir", "day_abn", "day_effect",
            "junk", "ret_1m", "ret_3m", "ret_1y")

        clusters = []
        for ticker, grp in groupby(rows, key=lambda r: r[0]):
            items = list(grp)  # already sorted by dt within ticker
            i = 0
            n = len(items)
            while i < n:
                j = i
                while j + 1 < n and (items[j + 1][1] - items[j][1]).days <= gap:
                    j += 1
                run = items[i:j + 1]
                i = j + 1
                if len(run) < min_items:
                    continue
                n_items = len(run)
                n_junk = sum(1 for r in run if r[7])
                n_moved = sum(1 for r in run if r[6])
                n_bull = sum(1 for r in run if (r[4] or 0) > 0)
                n_bear = sum(1 for r in run if (r[4] or 0) < 0)
                # One abnormal session-move per unique reaction DATE: several headlines on the same
                # day all carry that day's move, so summing per-item multiplied a single +X% day by
                # the headline count (a 5-PR +10% day read as +50%). r[1] is a DateTimeField, so key
                # on its .date() — keying on the raw timestamp leaves intraday same-day headlines
                # un-collapsed and still double-counts.
                _abn_by_day = {(r[1].date() if hasattr(r[1], "date") else r[1]): r[5]
                               for r in run if r[5] is not None}
                net_abn = round(sum(_abn_by_day.values()), 2) if _abn_by_day else None
                span_days = (run[-1][1] - run[0][1]).days
                n_dir = n_bull + n_bear
                bull_share = (n_bull / n_dir) if n_dir else 0.0
                junk_share = n_junk / n_items
                # forward return AFTER the cluster = the last item's precomputed forward returns
                last = run[-1]
                fwd_1m, fwd_3m, fwd_1y = last[8], last[9], last[10]
                pop = net_abn if net_abn is not None else 0.0
                # prop_score: volume + bullish skew + junk/PR share, gated by an actual UP push
                vol_f = min(n_items / 10.0, 1.0)
                raw = 0.40 * bull_share + 0.30 * junk_share + 0.30 * vol_f
                prop_score = int(round(100 * raw * (1.0 if pop > 0 else 0.35)))
                faded = bool(pop > 0 and fwd_1m is not None and fwd_1m < 0)
                # a couple of representative headlines for context
                heads = [r[2] for r in run][:3]
                clusters.append({
                    "ticker": ticker,
                    "start": run[0][1].date().isoformat(),
                    "end": run[-1][1].date().isoformat(),
                    "span_days": span_days,
                    "n_items": n_items, "n_moved": n_moved, "n_junk": n_junk,
                    "n_bull": n_bull, "n_bear": n_bear,
                    "bull_share": round(bull_share, 2), "junk_share": round(junk_share, 2),
                    "net_abn": net_abn,
                    "ret_1m": fwd_1m, "ret_3m": fwd_3m, "ret_1y": fwd_1y,
                    "prop_score": prop_score, "faded": faded,
                    "headlines": heads,
                })

        sort = p.get("sort", "prop")
        if sort == "recent":
            clusters.sort(key=lambda c: c["end"], reverse=True)
        elif sort == "items":
            clusters.sort(key=lambda c: c["n_items"], reverse=True)
        elif sort == "pop":
            clusters.sort(key=lambda c: (c["net_abn"] or -1e9), reverse=True)
        elif sort == "fade":
            clusters.sort(key=lambda c: (c["ret_1m"] if c["ret_1m"] is not None else 1e9))
        else:  # prop
            clusters.sort(key=lambda c: (c["prop_score"], c["n_items"]), reverse=True)

        n_total = len(clusters)
        n_faded = sum(1 for c in clusters if c["faded"])

        def _f(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        out = [{k: _f(v) for k, v in c.items()} for c in clusters[:limit]]
        from django.db.models import Max
        _last = NewsItem.objects.aggregate(m=Max('created_at'))['m']
        return Response({
            "n_total": n_total, "n_faded": n_faded, "returned": len(out),
            "gap": gap, "min_items": min_items, "results": out,
            "last_updated": _last.isoformat() if _last else None,
        })


class SmartMoneyView(APIView):
    """Full detail behind the 'Smart money' badges (13D×N / 13G×N / insider $) for ONE ticker —
    powers the click-through popup (user: "clicking 13Gx5 should show a popup with ALL the details").
    Lists the individual SEC ownership filings (13D activist / 13G passive 5%+ stakes) and the
    per-day insider open-market buy/sell aggregates, newest first, plus an EDGAR deep link. GET only."""

    def get(self, request):
        from core.models import SecFiling, InsiderBuy
        ticker = (request.query_params.get("ticker") or "").strip().upper()
        if not ticker:
            return Response({"error": "ticker required"}, status=400)

        filings = list(SecFiling.objects.filter(ticker=ticker)
                       .order_by("-filed_date").values("form_group", "filed_date", "accession"))
        f13d = [{"filed_date": f["filed_date"].isoformat(), "accession": f["accession"]}
                for f in filings if f["form_group"] == "13D"]
        f13g = [{"filed_date": f["filed_date"].isoformat(), "accession": f["accession"]}
                for f in filings if f["form_group"] == "13G"]

        ins = list(InsiderBuy.objects.filter(ticker=ticker)
                   .order_by("-filed_date").values("filed_date", "buy_value", "sell_value", "buy_count"))
        insider = [{"filed_date": i["filed_date"].isoformat(), "buy_value": i["buy_value"],
                    "sell_value": i["sell_value"], "buy_count": i["buy_count"]} for i in ins]

        edgar = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}"
                 f"&type=SC+13&dateb=&owner=include&count=100")
        return Response({
            "ticker": ticker,
            "filings_13d": f13d, "filings_13g": f13g, "insider": insider,
            "totals": {
                "n_13d": len(f13d), "n_13g": len(f13g),
                "insider_buy": sum(i["buy_value"] for i in ins),
                "insider_sell": sum(i["sell_value"] for i in ins),
                "insider_buy_count": sum(i["buy_count"] for i in ins),
            },
            "edgar_url": edgar,
        })


class AdDivergenceView(APIView):
    """Serve the A/D-divergence scan: stocks whose Accumulation/Distribution line is in
    'accum divergence' state right now. `primed` rows are also firing a capitulation signal
    (the setup that ~tripled the edge); the rest are 'watch'. GET reads AdDivergenceSignal;
    POST triggers a fresh in-process scan in the background."""

    def get(self, request):
        from core.models import AdDivergenceSignal
        import math
        qs = AdDivergenceSignal.objects.all()
        n_total = qs.count()
        if not n_total:
            return Response({"computed": False, "results": [], "message":
                             "No A/D-divergence scan yet. POST to run it (needs stock studies first)."})
        try:
            limit = int(request.query_params.get("limit", 1000))
        except (TypeError, ValueError):
            limit = 1000
        meta = AdDivergenceSignal.objects.order_by("-computed_at").values("computed_at").first()
        rows = qs.order_by("knife", "low_quality", "-primed", "min_days_ago", "-hist_avg_return").values(
            "ticker", "last_close", "primed", "firing", "min_days_ago",
            "fires_60d", "pct_above_low", "knife", "low_quality",
            "best_signal_key", "best_signal_name", "best_exit_key",
            "hist_avg_return", "hist_win_rate", "hist_trades",
            "market_cap", "pe_ratio", "forward_pe", "profit_margin", "fund_buckets", "sectors",
            "insider_buy_90d", "recent_13d", "recent_13g")[:limit]

        def _finite(v):
            return None if (isinstance(v, float) and not math.isfinite(v)) else v
        clean = [{k: _finite(v) for k, v in r.items()} for r in rows]
        n_primed = qs.filter(primed=True).count()
        computed_at = meta["computed_at"] if meta else None
        return Response({
            "computed": True, "n_total": n_total, "n_primed": n_primed,
            "n_watch": n_total - n_primed,
            "computed_at": computed_at.isoformat() if computed_at else None,
            "results": clean,
        })

    def post(self, request):
        from api.tasks import compute_ad_divergence
        thread = threading.Thread(target=compute_ad_divergence, daemon=True)
        thread.start()
        return Response({"status": "A/D-divergence scan started"})


class AdDivergenceChartView(APIView):
    """Price + Accumulation/Distribution LINE for one ticker over the last N bars, plus the
    A/D-state code per bar and the capitulation-signal fire markers — everything the A/D
    Divergence page needs to draw the divergence (price flat/down while the ADL rises)."""

    def get(self, request):
        import numpy as np
        import pandas as pd
        from seq_fundamental_study import load_candles
        from pit_fundamentals import _ad_state
        from studies import SIGNALS
        from api.tasks import AD_CAPIT_SIGNALS

        ticker = request.query_params.get("ticker")
        if not ticker:
            return Response({"error": "ticker required"}, status=400)
        try:
            bars = int(request.query_params.get("bars", 160))
        except (TypeError, ValueError):
            bars = 160
        df = load_candles([ticker]).get(ticker)
        if df is None or len(df) < 30:
            return Response({"error": f"no candles for {ticker}"}, status=404)

        h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
        rng = (h - l).replace(0, np.nan)
        mfm = (((c - l) - (h - c)) / rng).fillna(0.0)
        adl = (mfm * v).cumsum()
        adl_sma = adl.rolling(10).mean()
        state = _ad_state(df)

        n = len(df)
        start = max(0, n - bars)
        fires = []
        for sk in AD_CAPIT_SIGNALS:
            try:
                sig = SIGNALS[sk][1](df).fillna(False)
            except Exception:
                continue
            for pos in np.where(sig.values)[0]:
                if pos >= start:
                    fires.append({"idx": int(pos - start), "signal_key": sk,
                                  "signal_name": SIGNALS[sk][0]})
        sl = slice(start, n)

        def ser(s, nd=2):
            return [None if pd.isna(x) else round(float(x), nd) for x in s.values[sl]]

        return Response({
            "ticker": ticker,
            "dates": [d.strftime("%Y-%m-%d") for d in df.index[sl]],
            "close": ser(c), "adl": ser(adl), "adl_sma": ser(adl_sma),
            "state": [None if pd.isna(x) else int(x) for x in state.values[sl]],
            "fires": fires,
        })


class StrategyForwardView(APIView):
    """Serve the two-mode sector-gated strategy's average forward path (day 1..90 from entry):
    where a trade sits, on average/median, N days after we bought it. GET reads the saved JSON;
    POST recomputes it in the background."""

    def get(self, request):
        from pathlib import Path
        import json
        p = Path(__file__).resolve().parent.parent / ".data" / "studies" / "strategy_forward.json"
        if not p.exists():
            return Response({"computed": False,
                             "message": "Not computed yet. POST to run it."})
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            return Response({"computed": False, "message": f"read error: {e}"})
        data["computed"] = True
        return Response(data)

    def post(self, request):
        from api.tasks import compute_strategy_forward
        thread = threading.Thread(target=compute_strategy_forward, daemon=True)
        thread.start()
        return Response({"status": "strategy forward-path computation started"})


class ResearchView(APIView):
    """Serve the Research/Lab comparisons (trigger×exit matrix, entry-timeframe, regime,
    cap-band risk, MPT). GET reads the cached JSON; POST recomputes (heavy, background)."""

    def get(self, request):
        from pathlib import Path
        import json
        p = Path(__file__).resolve().parent.parent / ".data" / "studies" / "research.json"
        if not p.exists():
            return Response({"computed": False, "message": "Not computed yet. POST to run it (few min)."})
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            return Response({"computed": False, "message": f"read error: {e}"})
        data["computed"] = True
        return Response(data)

    def post(self, request):
        from api.tasks import compute_research
        thread = threading.Thread(target=compute_research, daemon=True)
        thread.start()
        return Response({"status": "research recompute started (few min)"})


class PaperTradesView(APIView):
    """Live forward paper-trading record of Playbook picks. GET returns open + closed positions
    and the running track record; POST snapshots today's picks / marks positions to market."""

    def get(self, request):
        from core.models import PaperTrade
        import math
        rows = list(PaperTrade.objects.all().values(
            "ticker", "mode", "sector", "entry_date", "entry_price", "last_price", "peak_price",
            "status", "exit_date", "exit_price", "ret_pct", "hist_avg_return", "opened_at", "updated_at"))
        if not rows:
            return Response({"computed": False, "message": "No paper trades yet. POST to snapshot today's Playbook picks."})
        closed = [r for r in rows if r["status"] == "closed"]
        opn = [r for r in rows if r["status"] == "open"]

        def avg(xs):
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 1) if xs else None
        crets = [r["ret_pct"] for r in closed if r["ret_pct"] is not None]
        summary = {
            "n_open": len(opn), "n_closed": len(closed),
            "closed_win_rate": round(sum(1 for x in crets if x > 0) / len(crets) * 100, 1) if crets else None,
            "closed_avg_ret": avg(crets),
            "open_unrealized_avg": avg([r["ret_pct"] for r in opn]),
        }

        def _clean(r):
            return {k: (None if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()}
        return Response({"computed": True, "summary": summary, "trades": [_clean(r) for r in rows]})

    def post(self, request):
        from api.tasks import update_paper_trades
        thread = threading.Thread(target=update_paper_trades, daemon=True)
        thread.start()
        return Response({"status": "paper-trade snapshot started"})


class EquityCurveView(APIView):
    """Serve the portfolio backtest's equity curve vs SPY + metrics. GET reads the saved JSON;
    POST recomputes (heavy — full backtest — runs in the background)."""

    def get(self, request):
        from pathlib import Path
        import json
        p = Path(__file__).resolve().parent.parent / ".data" / "studies" / "equity_curve.json"
        if not p.exists():
            return Response({"computed": False, "message": "Not computed yet. POST to run it (takes a few min)."})
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            return Response({"computed": False, "message": f"read error: {e}"})
        data["computed"] = True
        return Response(data)

    def post(self, request):
        from api.tasks import compute_equity_curve
        thread = threading.Thread(target=compute_equity_curve, daemon=True)
        thread.start()
        return Response({"status": "equity-curve backtest started (few min)"})


class PlaybookView(APIView):
    """Serve the live end-to-end Playbook: the sector board (IN/TURNING/OUT) + today's ranked
    candidates through the full funnel. GET reads the saved JSON; POST recomputes in the bg."""

    def get(self, request):
        from pathlib import Path
        import json
        p = Path(__file__).resolve().parent.parent / ".data" / "studies" / "playbook.json"
        if not p.exists():
            return Response({"computed": False, "message": "Not computed yet. POST to run it."})
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            return Response({"computed": False, "message": f"read error: {e}"})
        data["computed"] = True
        return Response(data)

    def post(self, request):
        from api.tasks import compute_playbook
        thread = threading.Thread(target=compute_playbook, daemon=True)
        thread.start()
        return Response({"status": "playbook computation started"})


class DimIntersectionView(APIView):
    """Targeted multi-dimension intersection: bucket ONE signal×exit's trades by the
    COMBINATION of chosen dimensions. GET serves the last result; POST runs a fresh one."""
    _PATH = "/app/.data/studies/dim_intersection.json"

    def get(self, request):
        if not _os.path.exists(self._PATH):
            return Response({"computed": False})
        with open(self._PATH) as f:
            data = _json.load(f)
        data["computed"] = True
        return Response(data)

    def post(self, request):
        d = request.data
        signal = d.get("signal") or "obv_div_sort_pos"
        exit_key = d.get("exit") or "6m"
        dims = d.get("dims") or ["Market cap", "PB", "Insider buying"]
        if isinstance(dims, str):
            dims = [x.strip() for x in dims.split(",") if x.strip()]
        from api.tasks import run_dim_intersection_task
        threading.Thread(target=run_dim_intersection_task, args=(signal, exit_key, dims), daemon=True).start()
        return Response({"status": "running", "signal": signal, "exit": exit_key, "dims": dims})


class SectorStockDrilldownView(APIView):
    """Turn a sector-level signal into per-stock signals: which stocks in the sector's
    holdings backtest well on this signal×exit, and which are firing right now."""

    def get(self, request):
        sector = request.query_params.get("sector")
        signal = request.query_params.get("signal")
        exit_key = request.query_params.get("exit", "6m")
        try:
            recent = int(request.query_params.get("recent", 10))
        except ValueError:
            recent = 10
        if not sector or not signal:
            return Response({"error": "sector and signal query params required"}, status=400)
        from api.tasks import compute_sector_drilldown
        result = compute_sector_drilldown(sector, signal, exit_key, recent_window=recent)
        return Response(result)


class BacktestLabView(APIView):
    """Sector-rotation lab: many rotation rules backtested vs SPY (phase1) + in/out-of-sample split
    (phase2) + top-signals portfolio (phase3). GET serves the precomputed JSON that
    backtest_concept.py writes; POST triggers a fresh recompute in the background."""

    def get(self, request):
        import os as _os, json as _json
        from core.models import BacktestResult
        row = BacktestResult.objects.filter(kind="rotation_lab").first()
        if row:
            data = dict(row.payload or {})
            data["computed"] = True
            data["last_updated"] = row.computed_at.isoformat()
            return Response(data)
        path = "/app/.data/studies/backtest_concept.json"   # JSON fallback (pre-DB runs)
        if not _os.path.exists(path):
            return Response({"computed": False,
                             "message": "Backtest lab not computed yet. POST to run it."})
        with open(path) as f:
            data = _json.load(f)
        data["computed"] = True
        data["last_updated"] = data.get("computed_at")
        return Response(data)

    def post(self, request):
        import threading, subprocess
        def _run():
            try:
                subprocess.run(["python", "-u", "backtest_concept.py"], cwd="/app", timeout=1800)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        return Response({"status": "backtest lab recompute started"})


class BacktestDecompView(APIView):
    """WHERE the rotation edge comes from: the 3-arm decomposition (pick-only vs rotation-only vs
    rotation+pick), the 200-day-MA rotation with BOTH numbers (hold the ETF vs pick the stock after),
    and the value×technical section (cheapest-P/B pick). GET serves the JSON backtest_lowpb.py writes;
    POST triggers a fresh recompute in the background."""

    def get(self, request):
        import os as _os, json as _json
        from core.models import BacktestResult
        row = BacktestResult.objects.filter(kind="decomposition").first()
        if row:
            data = dict(row.payload or {})
            data["computed"] = True
            data["last_updated"] = row.computed_at.isoformat()
            return Response(data)
        path = "/app/.data/studies/backtest_lowpb.json"     # JSON fallback (pre-DB runs)
        if not _os.path.exists(path):
            return Response({"computed": False,
                             "message": "Decomposition not computed yet. POST to run it."})
        with open(path) as f:
            data = _json.load(f)
        data["computed"] = True
        data["last_updated"] = data.get("computed_at")
        return Response(data)

    def post(self, request):
        import threading, subprocess
        def _run():
            try:
                subprocess.run(["python", "-u", "backtest_lowpb.py"], cwd="/app", timeout=3600)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        return Response({"status": "decomposition recompute started"})


class _StudyResultView(APIView):
    """Serve a persisted analysis payload from BacktestResult by `kind` (DB-first, JSON fallback).
    POST re-runs the backing script in the background."""
    kind = None
    script = None
    json_path = None

    def get(self, request):
        import os as _os, json as _json
        from core.models import BacktestResult
        row = BacktestResult.objects.filter(kind=self.kind).first()
        if row:
            data = dict(row.payload or {})
            data["computed"] = True
            data["last_updated"] = row.computed_at.isoformat()
            return Response(data)
        if self.json_path and _os.path.exists(self.json_path):
            with open(self.json_path) as f:
                data = _json.load(f)
            data["computed"] = True
            data["last_updated"] = data.get("computed_at")
            return Response(data)
        return Response({"computed": False, "message": f"{self.kind} not computed yet. POST to run it."})

    def post(self, request):
        import threading, subprocess
        script = self.script

        def _run():
            try:
                subprocess.run(["python", "-u", script], cwd="/app", timeout=3600)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        return Response({"status": f"{self.kind} recompute started"})


class CongressStudyView(_StudyResultView):
    """Congressional-trades forward-return study (PIT, market-adjusted vs SPY)."""
    kind = "congress_study"
    script = "congress_study.py"
    json_path = "/app/.data/studies/congress_study.json"


class DelistedSurvivorshipView(_StudyResultView):
    """Survivorship-bias audit of the study universe vs the EODHD delisted list."""
    kind = "delisted_survivorship"
    script = "delisted_survivorship.py"
    json_path = "/app/.data/studies/delisted_survivorship.json"


class DarkPoolBacktestView(_StudyResultView):
    """Historical dark-pool backtest: monthly PIT high-share / accumulation strategies vs SPY."""
    kind = "darkpool_backtest"
    script = "darkpool_backtest.py"
    json_path = "/app/.data/studies/darkpool_backtest.json"


class CongressBacktestView(_StudyResultView):
    """Equity-curve backtest of following legislator buys (all / Senate / large) vs SPY."""
    kind = "congress_backtest"
    script = "congress_backtest.py"
    json_path = "/app/.data/studies/congress_backtest.json"


class VolShockStudyView(_StudyResultView):
    """Vol-normalized shock continuation study: after a big vol-relative day, continue or reverse?
    Continuation matrix + slices (cap/volume/regime/sector) + exit-ladder backtest of vol_shock_*."""
    kind = "vol_shock_study"
    script = "vol_shock_study.py"
    json_path = "/app/.data/studies/vol_shock_study.json"


class H4StudyView(_StudyResultView):
    """H4 short-horizon studies engine: 5 signal families × bar-based exit ladder over the liquid
    top-250 stocks, magnitude-bucketed (tail-not-average), daily-trend split, daily benchmark.
    GET reads BacktestResult[h4_study]; POST recomputes (fetch may take a while)."""
    kind = "h4_study"
    script = "h4_study.py"
    json_path = "/app/.data/studies/h4_study.json"


class H4OnSignalsView(_StudyResultView):
    """H4-on-daily-signals: daily A/B/C selects the name, the H4 engine times a 0-3 day entry inside
    the candidate window. GET reads BacktestResult[h4_on_signals]; POST recomputes (fetch may take a while)."""
    kind = "h4_on_signals"
    script = "h4_on_signals_study.py"
    json_path = "/app/.data/studies/h4_on_signals.json"


class SignalFiringView(APIView):
    """Names (stock/ETF/commodity) currently firing a given study signal (last N bars).
    GET ?signal=<key> -> the firing list for the Studies 'firing now' pane."""
    def get(self, request):
        from core.models import SignalFiring
        sk = request.query_params.get("signal")
        row = SignalFiring.objects.filter(signal_key=sk).first() if sk else None
        if not row:
            return Response({"signal": sk, "n_firing": 0, "firing": [], "computed": False})
        return Response({"signal": sk, "signal_name": row.signal_name, "recent_bars": row.recent_bars,
                         "n_firing": row.n_firing, "firing": row.firing or [], "computed": True,
                         "last_updated": row.computed_at.isoformat() if row.computed_at else None})

    def post(self, request):
        import threading, subprocess

        def _run():
            try:
                subprocess.run(["python", "-u", "/app/signal_firing_scan.py", "--db", "--jobs", "4"],
                               cwd="/app", timeout=3600)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        return Response({"status": "signal firing scan started"})


class RotationPicksView(_StudyResultView):
    """LIVE rotation-pick scanner: cheapest positive-P/B stock in each top-momentum sector — the only
    sector-rotation arm with real alpha (arm3_lowpb, +154% vs SPY t2.09). GET reads it; POST recomputes."""
    kind = "rotation_picks"
    script = "rotation_pick_scan.py"
    json_path = "/app/.data/studies/rotation_picks.json"


class RotationHistoryView(_StudyResultView):
    """TIME MACHINE: the flagship basket reconstructed point-in-time at every month-end over ~5y, with each
    pick's realized next-month return + div_2x weights. Powers the month-by-month ◀ ▶ replay."""
    kind = "rotation_history"
    script = "rotation_history_scan.py"
    json_path = "/app/.data/studies/rotation_history.json"


class RsMethodsView(_StudyResultView):
    """RS-trend method sweep: ~20 ways to read the ETF/SPY relative-strength bar, each feeding the SAME
    cheapest-P/B large-cap pick, backtested side-by-side (t-stat / Sharpe / drawdown / coverage) so the
    dashboard can compare selection rules. Multiple-comparisons caveat travels in the payload."""
    kind = "rs_methods"
    script = "rs_methods_backtest.py"
    json_path = "/app/.data/studies/rs_methods.json"


class SyntheticMaCrossView(_StudyResultView):
    """MA crossover run on all 93 synthetic relative-strength candles (etf/spy). Pooled forward RELATIVE
    return after each golden/death cross + the golden-vs-death state drift + a live golden/death snapshot.
    Key result: a golden cross ON the RS bar is mean-reverting (lags SPY) — only the state carries info."""
    kind = "synthetic_ma_cross"
    script = "synthetic_ma_cross.py"
    json_path = "/app/.data/studies/synthetic_ma_cross.json"


class OversoldBounceView(_StudyResultView):
    """Short-term ABSOLUTE single-stock oversold-reversal entry (the edge the RS bar lacks). Entry backtest
    by RSI threshold + by oversold depth (deeper = bigger bounce), plus a live firing list of stocks that
    just crossed up out of oversold, most-oversold first. GET reads it; POST recomputes."""
    kind = "oversold_bounce"
    script = "oversold_bounce.py"
    json_path = "/app/.data/studies/oversold_bounce.json"


class DiversifierView(_StudyResultView):
    """Rank all 93 sector sleeves by how much they diversify SPY (low correlation) rather than by return.
    Commodities don't reliably beat SPY but pure ones are uncorrelated (Gold the standout) — drawdown value,
    not outperformance. GET reads BacktestResult[diversifier]; POST recomputes."""
    kind = "diversifier"
    script = "diversifier_scan.py"
    json_path = "/app/.data/studies/diversifier.json"


class RegimeView(_StudyResultView):
    """Macro regime -> sector leadership. Classifies each month by rates (TLT), inflation (TIP/TLT), and
    market (SPY vs 200d), and reports which sectors historically led in each regime + today's regime and its
    leaders. In-sample descriptive (hypothesis generator), per the payload caveat. POST recomputes."""
    kind = "regime"
    script = "regime_scan.py"
    json_path = "/app/.data/studies/regime.json"


class EntrySignalView(_StudyResultView):
    """Which ENTRY signal is right for the value-pick basket? Holds selection fixed (arm3_lowpb: top-momentum
    sectors -> cheapest-P/B pick) and varies only the entry gate on the pick's ABSOLUTE price. Result: buying
    the value name into an oversold DIP (RSI(10)<45/<35) adds forward return; buying strength/confirmation
    (above-MA, RSI cross-up, MACD-up) subtracts. GET reads BacktestResult[entry_signal]; POST recomputes."""
    kind = "entry_signal"
    script = "entry_signal_study.py"
    json_path = "/app/.data/studies/entry_signal.json"


class RotationCallView(_StudyResultView):
    """THE headline rotation call: regime-leader sectors (Regime) -> cheapest positive-P/B value pick
    (arm3_lowpb) -> entered on an oversold dip (entry_signal winner). The flagship that reflects what beats
    SPY — explicitly NOT the sector TREND TURN alert. GET reads BacktestResult[rotation_call]; POST recomputes."""
    kind = "rotation_call"
    script = "rotation_call_scan.py"
    json_path = "/app/.data/studies/rotation_call.json"


class ProfitabilityGuardView(_StudyResultView):
    """Does a PROFITABILITY guard improve the cheapest-P/B value pick? Tests the value-trap thesis (cheap +
    unprofitable + eroding book = melting ice cube). Result: blanket 'profitable-only' HURTS (kills negative-
    EPS turnarounds); the nuanced 'ex_trap_turn' (drop only unprofitable+book-shrinking+not-improving, keep
    turnarounds) beats the unguarded baseline. GET reads BacktestResult[profitability_guard]; POST recomputes."""
    kind = "profitability_guard"
    script = "profitability_guard_study.py"
    json_path = "/app/.data/studies/profitability_guard.json"


class FactorLabView(_StudyResultView):
    """Factor Lab: sweep many filters/tilts/combos (profit guard, FCF+, low-debt, gross margin, rev growth,
    A/D accumulation divergence, small/micro cap, momentum, earnings yield, composite value, inverse-vol
    weighting, and stacked combos) on the value-pick baseline, ranked to find the best return / risk-adjusted.
    GET reads BacktestResult[factor_lab]; POST recomputes."""
    kind = "factor_lab"
    script = "factor_lab.py"
    json_path = "/app/.data/studies/factor_lab.json"


class PortfolioBlenderView(_StudyResultView):
    """Portfolio Blender: mix the CORE value engine (C+B: guard+low_debt value pick) with a CAPITULATION
    sleeve (A: deep-oversold + A/D accumulation, ~10d bounce) as monthly return streams. Measures
    correlation + crisis-alpha, sweeps allocations, vol-matches, and tests a regime-switched split.
    GET reads BacktestResult[portfolio_blender]; POST recomputes."""
    kind = "portfolio_blender"
    script = "portfolio_blender.py"
    json_path = "/app/.data/studies/portfolio_blender.json"


class StrategyLabView(_StudyResultView):
    """Strategy Lab: can A or B beat C WITHOUT the sector rotation, and do C's rules travel? Runs B (dip in
    uptrend) and A (capitulation) pure and with C's rules (posP/B + guard + low_debt) applied, no rotation,
    vs the C reference. Isolates rotation-filter vs pick-rules. GET reads BacktestResult[strategy_lab]."""
    kind = "strategy_lab"
    script = "strategy_lab.py"
    json_path = "/app/.data/studies/strategy_lab.json"


class ValueRankingView(_StudyResultView):
    """Value Ranking Lab: hold the rotation+guard+low_debt selection fixed, vary ONLY the value metric that
    picks the stock (cheapest P/B vs EV/EBIT vs FCF-yield vs earnings-yield vs P/S vs composite). Finds which
    value lens picks the best-returning name. GET reads BacktestResult[value_ranking]; POST recomputes."""
    kind = "value_ranking"
    script = "value_ranking_lab.py"
    json_path = "/app/.data/studies/value_ranking.json"


class ReturnLabView(_StudyResultView):
    """Return Lab: push the validated engine (rotation+guard+low_debt+cheapest-P/B) for MORE return across
    4 levers — concentration+weighting, leverage, orthogonal-sleeve blends, and regime/drawdown overlays.
    Reports the return-vs-drawdown frontier so the risk cost of each lever is explicit. GET reads
    BacktestResult[return_lab]; POST recomputes."""
    kind = "return_lab"
    script = "return_lab.py"
    json_path = "/app/.data/studies/return_lab.json"


class DeepPoolView(_StudyResultView):
    """More stocks per ETF: cheapest-P/B from the ETF's full expanded holdings vs our top-20. Same engine,
    deeper candidate pool. GET reads BacktestResult[deep_pool]; POST recomputes."""
    kind = "deep_pool"
    script = "deep_pool_study.py"
    json_path = "/app/.data/studies/deep_pool.json"


class BearDefenseView(_StudyResultView):
    """Dual-momentum bear defense: relative momentum picks sectors, absolute momentum gates in/out (cash).
    GET reads BacktestResult[bear_defense]; POST recomputes."""
    kind = "bear_defense"
    script = "bear_defense.py"
    json_path = "/app/.data/studies/bear_defense.json"


class V2StrategyView(_StudyResultView):
    """V2: stack the levers that helped (deep pool + top-5 inverse-vol + slow-momentum bear gate),
    decomposed additively vs the validated baseline. GET reads BacktestResult[v2_strategy]; POST recomputes."""
    kind = "v2_strategy"
    script = "v2_strategy.py"
    json_path = "/app/.data/studies/v2_strategy.json"


class WalkForwardView(_StudyResultView):
    """Walk-forward / subperiod validation of the validated engine (halves, thirds, per-year, rolling-12mo).
    GET reads BacktestResult[walk_forward]; POST recomputes."""
    kind = "walk_forward"
    script = "walk_forward.py"
    json_path = "/app/.data/studies/walk_forward.json"


class SectorAccelerationView(_StudyResultView):
    """Live sector-acceleration leaderboard — the walk-forward-validated sector signal (accel = 3mo-now
    minus 3mo-3ago; +422% vs SPY). Ranks all sectors by momentum acceleration; top-10 = the rotation
    pick-zone; flags 'hidden turns' (negative 6mo momentum but inflecting up). GET reads
    BacktestResult[sector_acceleration]; POST recomputes."""
    kind = "sector_acceleration"
    script = "sector_acceleration_scan.py"
    json_path = "/app/.data/studies/sector_acceleration.json"


class NewsOverreactionView(_StudyResultView):
    """News overreaction detector + reversion backtest: good-news-crash / bad-news-pop divergence,
    forward reversion bucketed by move size (the edge lives in the 10-15% tail), + gap profile."""
    kind = "news_overreaction"
    script = "news_overreaction.py"
    json_path = "/app/.data/studies/news_overreaction.json"


class RsiIntradayView(APIView):
    """Intraday RSI(14) crossover backtest, bucketed by RSI level at the cross (edge only from
    oversold). GET ?tf=4h|8h|12h reads BacktestResult[rsi_<tf>_backtest]; POST recomputes 4h (fetch)."""
    def get(self, request):
        from core.models import BacktestResult
        tf = request.query_params.get("tf", "12h")
        if tf not in ("4h", "8h", "12h"):
            tf = "12h"
        row = BacktestResult.objects.filter(kind=f"rsi_{tf}_backtest").first()
        if not row:
            return Response({"computed": False, "tf": tf,
                             "message": "Not computed yet. POST to run the intraday fetch + backtest."})
        data = dict(row.payload or {})
        data["computed"] = True
        data["tf"] = tf
        data["last_updated"] = row.computed_at.isoformat()
        data["available_tf"] = sorted(
            k.replace("rsi_", "").replace("_backtest", "")
            for k in BacktestResult.objects.filter(kind__startswith="rsi_")
            .values_list("kind", flat=True))
        return Response(data)

    def post(self, request):
        import threading, subprocess

        def _run():
            try:
                subprocess.run(["python", "-u", "/app/rsi_4h_study.py", "--tf", "4h,8h,12h"],
                               cwd="/app", timeout=3600)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()
        return Response({"status": "rsi intraday recompute started"})


def _burst_scan_bg():
    import subprocess
    try:
        subprocess.run(["python", "-u", "/app/burst_scan.py", "--db", "--jobs", "4"],
                       cwd="/app", timeout=3600)
    except Exception:
        pass


def _clean_rows(rows):
    """Null out non-finite floats so DRF/orjson can serialize (some ratios are inf)."""
    import math
    for r in rows:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                r[k] = None
    return rows


class ShortTermView(APIView):
    """Short-term BURST scanner (ShortTermSignal): what is bursting today, momentum or reversal,
    with the trigger's short-horizon edge. GET reads the DB (optional ?burst_type=); POST re-runs
    the burst scan in the background."""
    FIELDS = ["ticker", "signal_key", "signal_name", "burst_type", "days_ago", "last_close",
              "day1_move", "z_shock", "best_exit_key", "hist_avg_return", "hist_win_rate",
              "hist_trades", "market_cap", "pe_ratio", "forward_pe", "fund_buckets", "sectors",
              "insider_buy_90d", "recent_13d", "recent_13g", "intraday_signal", "intraday_rsi"]

    def get(self, request):
        from core.models import ShortTermSignal
        from django.db.models import Max
        if not ShortTermSignal.objects.exists():
            return Response({"computed": False, "results": [],
                             "message": "No burst scan yet. POST to run it (needs stock studies first)."})
        qs = ShortTermSignal.objects.all()
        bt = request.query_params.get("burst_type")
        if bt in ("momentum", "reversal"):
            qs = qs.filter(burst_type=bt)
        rows = _clean_rows(list(qs.values(*self.FIELDS)))
        last = ShortTermSignal.objects.aggregate(m=Max("computed_at"))["m"]
        return Response({
            "computed": True, "n_burst": len(rows), "results": rows,
            "n_momentum": ShortTermSignal.objects.filter(burst_type="momentum").count(),
            "n_reversal": ShortTermSignal.objects.filter(burst_type="reversal").count(),
            "last_updated": last.isoformat() if last else None,
        })

    def post(self, request):
        threading.Thread(target=_burst_scan_bg, daemon=True).start()
        return Response({"status": "burst scan started"})


class GlobalView(APIView):
    """GLOBAL confluence scanner (GlobalSignal): live burst confirmed by our other validated layers,
    scored 0-100 with a per-component breakdown, ranked by score. GET reads the DB; POST re-runs."""
    FIELDS = ["ticker", "global_score", "components", "burst_signal_key", "burst_signal_name",
              "burst_type", "burst_days_ago", "last_close", "best_signal_key", "hist_avg_return",
              "hist_win_rate", "hist_trades", "ad_state", "darkpool_off_pct", "darkpool_rising",
              "market_cap", "pe_ratio", "forward_pe", "fund_buckets", "sectors", "insider_buy_90d",
              "recent_13d", "recent_13g", "regime_bull", "sector_state",
              "intraday_signal", "intraday_rsi"]

    def get(self, request):
        from core.models import GlobalSignal
        from django.db.models import Max
        if not GlobalSignal.objects.exists():
            return Response({"computed": False, "results": [],
                             "message": "No burst scan yet. POST to run it (needs stock studies first)."})
        rows = _clean_rows(list(GlobalSignal.objects.all().values(*self.FIELDS)))
        last = GlobalSignal.objects.aggregate(m=Max("computed_at"))["m"]
        return Response({
            "computed": True, "n_global": len(rows), "results": rows,
            "weights": {"burst": 12, "edge": 12, "ad": 12, "darkpool": 10, "smart_money": 8,
                        "fundamentals": 8, "regime": 8, "news": 15, "intraday": 15},
            "regime_bull": bool(rows and rows[0].get("regime_bull")),
            "last_updated": last.isoformat() if last else None,
        })

    def post(self, request):
        threading.Thread(target=_burst_scan_bg, daemon=True).start()
        return Response({"status": "burst scan started"})
