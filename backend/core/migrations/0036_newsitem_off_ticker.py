# Generated for NewsItem.off_ticker (local-LLM off-ticker / not-about-this-ticker guard)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0035_newsitem_cat_llm'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsitem',
            name='off_ticker',
            field=models.BooleanField(blank=True, db_index=True, null=True),
        ),
    ]
