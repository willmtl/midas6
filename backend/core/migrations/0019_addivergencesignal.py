# Generated for the A/D-divergence live scanner.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0018_livesignal_insider_buy_90d_livesignal_recent_13d_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AdDivergenceSignal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ticker', models.CharField(max_length=20, unique=True)),
                ('last_close', models.FloatField(default=0)),
                ('primed', models.BooleanField(default=False)),
                ('firing', models.JSONField(blank=True, null=True)),
                ('min_days_ago', models.IntegerField(blank=True, null=True)),
                ('best_signal_key', models.CharField(blank=True, max_length=60)),
                ('best_signal_name', models.CharField(blank=True, max_length=120)),
                ('best_exit_key', models.CharField(blank=True, max_length=60)),
                ('hist_avg_return', models.FloatField(blank=True, null=True)),
                ('hist_win_rate', models.FloatField(blank=True, null=True)),
                ('hist_trades', models.IntegerField(blank=True, null=True)),
                ('market_cap', models.FloatField(blank=True, null=True)),
                ('pe_ratio', models.FloatField(blank=True, null=True)),
                ('forward_pe', models.FloatField(blank=True, null=True)),
                ('profit_margin', models.FloatField(blank=True, null=True)),
                ('fund_buckets', models.JSONField(blank=True, null=True)),
                ('sectors', models.JSONField(blank=True, null=True)),
                ('insider_buy_90d', models.BigIntegerField(blank=True, null=True)),
                ('recent_13d', models.IntegerField(default=0)),
                ('recent_13g', models.IntegerField(default=0)),
                ('computed_at', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-primed', 'min_days_ago', '-hist_avg_return'],
                'indexes': [models.Index(fields=['primed'], name='core_addiv_primed_idx'), models.Index(fields=['min_days_ago'], name='core_addiv_mindays_idx')],
            },
        ),
    ]
