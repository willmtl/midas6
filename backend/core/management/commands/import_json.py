"""
Import existing JSON candle data and study results into PostgreSQL.
Usage: python manage.py import_json [--studies-only] [--candles-only]
"""

import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.utils import timezone
from core.models import Sector, Candle, Study, Trade, StudySectorResult
import config
import sector_holdings


class Command(BaseCommand):
    help = "Import existing JSON data into PostgreSQL"

    def add_arguments(self, parser):
        parser.add_argument('--studies-only', action='store_true')
        parser.add_argument('--candles-only', action='store_true')
        parser.add_argument('--trades', action='store_true', help='Also import trades (slow)')

    def handle(self, *args, **options):
        studies_only = options.get('studies_only', False)
        candles_only = options.get('candles_only', False)

        if not studies_only and not candles_only:
            self._import_sectors()
            self._import_candles()
            self._import_studies()
        elif studies_only:
            self._import_sectors()
            self._import_studies()
        elif candles_only:
            self._import_sectors()
            self._import_candles()

        if options.get('trades'):
            self._import_trades()

    def _import_sectors(self):
        self.stdout.write("Importing sectors...")
        for name, etf in config.SECTOR_ETFS.items():
            # Try by name first, then by etf, else create
            sector = Sector.objects.filter(name=name).first()
            if sector:
                if sector.etf != etf:
                    sector.etf = etf
                    sector.save()
            else:
                sector = Sector.objects.filter(etf=etf).first()
                if sector:
                    sector.name = name
                    sector.save()
                else:
                    Sector.objects.create(name=name, etf=etf)
        self.stdout.write(f"  {Sector.objects.count()} sectors")

    def _import_candles(self):
        data_dir = Path(__file__).resolve().parent.parent.parent.parent / ".data" / "5y"
        if not data_dir.exists():
            self.stdout.write("  No candle data found in .data/5y/")
            return

        files = list(data_dir.glob("*.json"))
        self.stdout.write(f"Importing candles from {len(files)} JSON files...")

        total = 0
        for f in files:
            ticker = f.stem
            with open(f) as fp:
                data = json.load(fp)

            candles = data.get("candles", [])
            if not candles:
                continue

            bulk = []
            for c in candles:
                try:
                    bulk.append(Candle(
                        ticker=ticker, date=c["date"][:10], interval="1d",
                        open=c.get("open", 0), high=c.get("high", 0),
                        low=c.get("low", 0), close=c.get("close", 0),
                        volume=c.get("volume", 0),
                    ))
                except Exception:
                    continue

            if bulk:
                Candle.objects.bulk_create(bulk, ignore_conflicts=True, batch_size=5000)
                total += len(bulk)

            self.stdout.write(f"  {ticker}: {len(bulk)} candles")

        self.stdout.write(f"  Total: {total} candles imported")

    def _import_studies(self):
        studies_file = Path(__file__).resolve().parent.parent.parent.parent / ".data" / "studies" / "results.json"
        if not studies_file.exists():
            self.stdout.write("  No studies results found")
            return

        with open(studies_file) as f:
            data = json.load(f)

        studies = data.get("studies", [])
        self.stdout.write(f"Importing {len(studies)} studies...")

        created_count = 0
        updated_count = 0
        for sd in studies:
            fields = {
                "name": sd.get("name", ""),
                "signal_name": sd.get("signal_name", ""),
                "exit_name": sd.get("exit_name", ""),
                "category": sd.get("category", ""),
                "total_trades": sd.get("total_trades", 0),
                "avg_return": sd.get("avg_return", 0),
                "win_rate": sd.get("win_rate", 0),
                "avg_hold": sd.get("avg_hold", 0),
                "sector_count": sd.get("sector_count", 0),
                "peak_day": sd.get("peak_day"),
                "peak_avg": sd.get("peak_avg"),
                "ret_90d": sd.get("ret_90d"),
                "best_peak_day": sd.get("best_peak_day"),
                "best_peak_ret": sd.get("best_peak_ret"),
                "best_ret_90d": sd.get("best_ret_90d"),
                "by_regime": sd.get("by_regime"),
                "by_curve": sd.get("by_curve"),
                "by_vix": sd.get("by_vix"),
                "by_spy_trend": sd.get("by_spy_trend"),
                "by_season": sd.get("by_season"),
                "best_sectors": sd.get("best_sectors"),
                "worst_sectors": sd.get("worst_sectors"),
                "is_computed": True,
                "computed_at": timezone.now(),
            }

            study, created = Study.objects.update_or_create(
                signal_key=sd.get("signal", ""),
                exit_key=sd.get("exit", ""),
                defaults=fields,
            )

            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(f"  {created_count} created, {updated_count} updated. Total: {Study.objects.count()}")

    def _import_trades(self):
        """Compute and import trades for all studies into the DB."""
        import studies as studies_mod
        import data_fetcher

        self.stdout.write("Loading market data...")
        all_data = data_fetcher.fetch_all()
        self.stdout.write(f"  Loaded {len(all_data)} tickers")

        # Pre-compute indicators
        import ta
        for ticker, df in all_data.items():
            if len(df) < 20:
                continue
            df["_sortino"] = studies_mod._rolling_sortino(df)
            df["_omega"] = studies_mod._rolling_omega(df)
            df["_rsi"] = ta.momentum.rsi(df["Close"], window=10)
            df["_rsi_sma"] = df["_rsi"].rolling(10).mean()

        sector_lookup = {s.etf: s for s in Sector.objects.all()}
        etf_to_sector = {v: k for k, v in config.SECTOR_ETFS.items()}

        studies = Study.objects.filter(is_computed=True)
        self.stdout.write(f"Importing trades for {studies.count()} studies...")

        for i, study in enumerate(studies):
            if Trade.objects.filter(study=study).exists():
                continue

            sig_key = study.signal_key
            exit_key = study.exit_key
            if sig_key not in studies_mod.SIGNALS or exit_key not in studies_mod.EXITS:
                continue

            _, sig_fn = studies_mod.SIGNALS[sig_key]
            _, exit_fn = studies_mod.EXITS[exit_key]

            bulk_trades = []
            for etf, sector_name in etf_to_sector.items():
                df = all_data.get(etf)
                if df is None or len(df) < 60:
                    continue

                sector = sector_lookup.get(etf)
                if not sector:
                    continue

                try:
                    signals = sig_fn(df).fillna(False)
                except Exception:
                    continue

                for entry_date in signals[signals].index:
                    idx = df.index.get_loc(entry_date)
                    exit_idx = exit_fn(df, idx)
                    if exit_idx is None or exit_idx <= idx or exit_idx >= len(df):
                        continue
                    ep = float(df["Close"].iloc[idx])
                    xp = float(df["Close"].iloc[exit_idx])
                    if ep <= 0:
                        continue
                    ret = (xp - ep) / ep * 100
                    bulk_trades.append(Trade(
                        study=study,
                        sector=sector,
                        etf=etf,
                        entry_date=entry_date.date() if hasattr(entry_date, 'date') else entry_date,
                        exit_date=df.index[exit_idx].date() if hasattr(df.index[exit_idx], 'date') else df.index[exit_idx],
                        entry_price=round(ep, 2),
                        exit_price=round(xp, 2),
                        return_pct=round(ret, 3),
                        hold_days=exit_idx - idx,
                    ))

            if bulk_trades:
                Trade.objects.bulk_create(bulk_trades, batch_size=5000)

            if (i + 1) % 50 == 0:
                self.stdout.write(f"  [{i+1}/{studies.count()}] trades imported")

        self.stdout.write(f"  Total trades: {Trade.objects.count()}")
