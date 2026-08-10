# Forward paper-trade tracker for Playbook picks.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0021_addiv_low_quality'),
    ]

    operations = [
        migrations.CreateModel(
            name='PaperTrade',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ticker', models.CharField(max_length=20)),
                ('mode', models.CharField(blank=True, max_length=4)),
                ('sector', models.CharField(blank=True, max_length=100)),
                ('entry_date', models.DateField()),
                ('entry_price', models.FloatField()),
                ('peak_price', models.FloatField(default=0)),
                ('last_price', models.FloatField(default=0)),
                ('status', models.CharField(default='open', max_length=8)),
                ('exit_date', models.DateField(blank=True, null=True)),
                ('exit_price', models.FloatField(blank=True, null=True)),
                ('ret_pct', models.FloatField(blank=True, null=True)),
                ('hist_avg_return', models.FloatField(blank=True, null=True)),
                ('opened_at', models.DateTimeField(blank=True, null=True)),
                ('updated_at', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'ordering': ['status', '-opened_at'],
                'unique_together': {('ticker', 'entry_date')},
            },
        ),
    ]
