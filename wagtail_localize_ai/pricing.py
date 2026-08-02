from decimal import Decimal
from django.conf import settings


def get_pricing_table() -> dict:
    return getattr(settings, "WAGTAIL_LOCALIZE_AI_PRICING", {})


def _lookup_rates(model: str):
    if not model:
        return None
    pricing = get_pricing_table()
    rates = pricing.get(model)
    if rates is not None:
        return rates
    if "/" in model or ":" in model:
        slug = model.replace(":", "/").rsplit("/", 1)[-1]
        return pricing.get(slug)
    return None


def compute_cost(model: str, input_tokens: int, output_tokens: int):
    rates = _lookup_rates(model)
    if not rates:
        return None
    try:
        in_rate, out_rate = rates
        return (
            Decimal(input_tokens) * Decimal(str(in_rate))
            + Decimal(output_tokens) * Decimal(str(out_rate))
        ) / Decimal(1000000)
    except Exception:
        return None