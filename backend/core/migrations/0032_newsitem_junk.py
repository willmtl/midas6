# Generated for NewsItem.junk (title-based opinion/clickbait guard)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0031_newsitem_day_suspect'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsitem',
            name='junk',
            field=models.BooleanField(blank=True, db_index=True, null=True),
        ),
    ]
