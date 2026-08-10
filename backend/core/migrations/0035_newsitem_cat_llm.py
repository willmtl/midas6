# Generated for NewsItem.cat_llm (local-LLM refined category)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0034_newsitem_cat_auto'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsitem',
            name='cat_llm',
            field=models.CharField(blank=True, db_index=True, default='', max_length=20),
        ),
    ]
