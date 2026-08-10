# Generated for NewsItem.cat_auto (title-based event category)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0033_newsitem_forward_returns'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsitem',
            name='cat_auto',
            field=models.CharField(blank=True, db_index=True, default='', max_length=20),
        ),
    ]
