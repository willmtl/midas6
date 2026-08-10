from rest_framework import serializers
from core.models import Sector, Holding, Candle, Study, StudySectorResult, Trade, ScanResult


class SectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sector
        fields = ["id", "name", "etf", "category"]


class HoldingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Holding
        fields = ["ticker", "weight", "rank"]


class CandleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candle
        fields = ["date", "open", "high", "low", "close", "volume"]


class ScanResultSerializer(serializers.ModelSerializer):
    sector = serializers.CharField(source="sector.name")
    etf = serializers.CharField(source="sector.etf")

    class Meta:
        model = ScanResult
        exclude = ["id"]


class StudySectorResultSerializer(serializers.ModelSerializer):
    sector = serializers.CharField(source="sector.name")

    class Meta:
        model = StudySectorResult
        exclude = ["id", "study"]


class StudySerializer(serializers.ModelSerializer):
    signal = serializers.CharField(source="signal_key")
    exit = serializers.CharField(source="exit_key")

    class Meta:
        model = Study
        fields = [
            "id", "name", "signal", "signal_name", "exit", "exit_name",
            "category", "total_trades", "avg_return", "win_rate", "avg_hold",
            "avg_mae", "clean_pct",
            "sector_count", "peak_day", "peak_avg", "ret_90d",
            "best_peak_day", "best_peak_ret", "best_ret_90d",
            "by_regime", "by_curve", "by_vix", "by_spy_trend", "by_season",
            "best_sectors", "worst_sectors",
        ]


class TradeSerializer(serializers.ModelSerializer):
    sector = serializers.CharField(source="sector.name")

    class Meta:
        model = Trade
        fields = [
            "sector", "etf", "entry_date", "exit_date",
            "entry_price", "exit_price", "return_pct", "hold_days",
        ]
