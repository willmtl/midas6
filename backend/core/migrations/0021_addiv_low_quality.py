# Low-quality (landmine) flag for the A/D-divergence scanner.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0020_addiv_knife_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='addivergencesignal',
            name='low_quality',
            field=models.BooleanField(default=False),
        ),
    ]
