from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("wagtail_localize_ai", "0003_alter_aitranslatorsettings_prompt"),
    ]

    operations = [
        migrations.AddField(
            model_name="translationlog",
            name="page_id",
            field=models.IntegerField(
                blank=True,
                help_text="Best-effort ID of the page this string belongs to.",
                null=True,
                verbose_name="Page ID",
            ),
        ),
        migrations.AddField(
            model_name="translationlog",
            name="string_id",
            field=models.IntegerField(
                blank=True,
                help_text="ID of the wagtail_localize.String this segment was translated from.",
                null=True,
                verbose_name="String ID",
            ),
        ),
        migrations.AddField(
            model_name="translationlog",
            name="source_text",
            field=models.TextField(
                blank=True,
                help_text="The original text submitted for translation, for later quality review.",
                null=True,
                verbose_name="Source Text",
            ),
        ),
        migrations.AddField(
            model_name="translationlog",
            name="translated_text",
            field=models.TextField(
                blank=True,
                help_text="The translated text returned by the model, for later quality review.",
                null=True,
                verbose_name="Translated Text",
            ),
        ),
        migrations.AddField(
            model_name="translationlog",
            name="cost_usd",
            field=models.DecimalField(
                blank=True,
                decimal_places=6,
                help_text="Dollar cost of this translation, computed from token usage and the pricing table.",
                max_digits=12,
                null=True,
                verbose_name="Cost (USD)",
            ),
        ),
    ]