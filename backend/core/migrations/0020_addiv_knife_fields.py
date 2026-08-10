# Falling-knife metrics for the A/D-divergence scanner.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0019_addivergencesignal'),
    ]

    operations = [
        migrations.AddField(
            model_name='addivergencesignal',
            name='fires_60d',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='addivergencesignal',
            name='pct_above_low',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='addivergencesignal',
            name='knife',
            field=models.BooleanField(default=False),
        ),
    ]
