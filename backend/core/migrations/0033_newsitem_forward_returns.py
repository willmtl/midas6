# Generated for NewsItem.ret_1m / ret_3m / ret_1y (forward "results since" returns)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0032_newsitem_junk'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsitem',
            name='ret_1m',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='newsitem',
            name='ret_3m',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='newsitem',
            name='ret_1y',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
