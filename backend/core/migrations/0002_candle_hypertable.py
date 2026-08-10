from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql="""
                CREATE EXTENSION IF NOT EXISTS timescaledb;

                CREATE TABLE IF NOT EXISTS core_candle (
                    date DATE NOT NULL,
                    ticker VARCHAR(20) NOT NULL,
                    interval VARCHAR(5) NOT NULL DEFAULT '1d',
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume BIGINT NOT NULL,
                    UNIQUE (ticker, date, interval)
                );

                SELECT create_hypertable('core_candle', by_range('date'), if_not_exists => TRUE);

                CREATE INDEX IF NOT EXISTS idx_candle_ticker_date ON core_candle (ticker, date);
            """,
            reverse_sql="DROP TABLE IF EXISTS core_candle;",
        ),
    ]
