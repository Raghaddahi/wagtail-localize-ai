import concurrent
import time

from django.utils.translation import gettext_lazy as _, get_language_info
from bs4 import BeautifulSoup, NavigableString, Tag

from wagtail.models import Locale

from wagtail_localize.machine_translators.base import BaseMachineTranslator
from wagtail_localize.strings import INLINE_TAGS, StringValue

from wagtail_localize_ai.models import AITranslatorSettings, TranslationLog
from wagtail_localize_ai.pricing import compute_cost
from wagtail_localize_ai.utils import get_llm_client, get_provider_display_name, normalize_model_identifier


# Concurrency for per-page segment translation. The Argyll gateway enforces a
# per-minute token quota on the API key; max_workers=8 bursts ~8 simultaneous
# requests and can drain the bucket before any of them returns, causing every
# segment on the page to come back "Rate limit exceeded" simultaneously and
# the whole page to fail with a single concatenated RuntimeError. max_workers=3
# keeps parallelism high enough to be fast while staying under the burst cap.
MAX_WORKERS = 3

# Rate-limit retry. On HTTP 429 or a body containing "rate limit" we sleep with
# exponential backoff + a little jitter, then retry the SAME segment. Limits:
# 3 retries (4 attempts total) keeps worst-case wait under ~15s, well inside
# wagtail-localize's request timeout.
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds; sleeps 2s, 4s, 8s (+jitter)


class AITranslator(BaseMachineTranslator):
    @property
    def display_name(self):
        provider = AITranslatorSettings.load().provider
        return get_provider_display_name(provider) if provider else _("AI Translator")

    def translate(self, source_locale: Locale, target_locale: Locale, strings: list[StringValue]) -> list[StringValue]:
        source_language = get_language_info(source_locale.language_code)[
            "name"
        ]
        target_language = get_language_info(target_locale.language_code)[
            "name"
        ]

        translator_settings = AITranslatorSettings.load()
        provider = translator_settings.provider
        model = translator_settings.model

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor_results = list(
                executor.map(translate_text, strings, [source_language] * len(strings), [target_language] * len(strings))
            )

        results = {}
        error = ""
        seen_errors = []
        for text, result in zip(strings, executor_results):
            usage = result.get("usage") or {"input_tokens": 0, "output_tokens": 0}
            source_text = result.get("source_text") or text.get_translatable_html()
            translated_text = result.get("translated_text")
            err = result.get("error")

            string_id, page_id = _resolve_ids(source_locale, text)
            cost_usd = compute_cost(model, usage["input_tokens"], usage["output_tokens"])

            TranslationLog.objects.create(
                provider=provider,
                model=model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                error=err,
                page_id=page_id,
                string_id=string_id,
                source_text=source_text,
                translated_text=translated_text,
                cost_usd=cost_usd,
            )

            if err:
                if err not in seen_errors:
                    seen_errors.append(err)
                error += err + "\n"
                continue
            results.update(result["result"])

        if not results and strings:
            # Surface only the distinct error messages, not 22 copies of the
            # same "Rate limit exceeded". Keeps the admin-visible RuntimeError
            # and the TranslationLog readable.
            summary = "; ".join(seen_errors) if seen_errors else str(_("Translation failed"))
            raise RuntimeError(summary)

        return results

    def can_translate(self, source_locale: Locale, target_locale: Locale):
        translator_settings = AITranslatorSettings.load()
        if not translator_settings:
            return False
        has_provider = translator_settings.provider
        has_model = translator_settings.model
        not_same_language = source_locale.language_code != target_locale.language_code

        return has_provider and has_model and not_same_language

def _is_rate_limit(exc) -> bool:
    """True if `exc` is a rate-limit rejection from the gateway.

    Detects both the OpenAI SDK's typed ``RateLimitError`` (HTTP 429) and the
    Argyll gateway's text-body form (``"Rate limit exceeded"`` in the message,
    served as a generic ``APIStatusError`` without a clean status).
    """
    # Typed 429 from the OpenAI SDK.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    # Body-substring fallback (Argyll gateway puts it in the message text).
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "try again" in msg and "later" in msg


def sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    def clean(element):
        for child in list(getattr(element, "children", [])):
            if isinstance(child, NavigableString):
                continue
            if isinstance(child, Tag):
                clean(child)
                if child.name not in INLINE_TAGS:
                    child.unwrap()
                else:
                    allowed_keys = {"id"} if child.name == "a" else set()
                    child.attrs = {
                        k: v for k, v in child.attrs.items() if k in allowed_keys
                    }

    clean(soup)
    return str(soup)


def _resolve_ids(source_locale: Locale, text: StringValue):
    """Best-effort resolve of (string_id, page_id) for a source segment.

    Returns (None, None) if either can't be determined. A source String may
    belong to multiple pages/segments; only the first is used for page_id.
    """
    string_id = None
    page_id = None
    try:
        from wagtail_localize.models import String, StringSegment
        from wagtail.models import Page

        data_hash = String._get_data_hash(text.data)
        string = String.objects.filter(locale=source_locale, data_hash=data_hash).first()
        if string:
            string_id = string.id
            seg = (
                StringSegment.objects.filter(string_id=string.id)
                .select_related("context")
                .first()
            )
            if seg and seg.context_id:
                translation_key = seg.context.object_id
                page = Page.objects.filter(translation_key=translation_key).first()
                if page:
                    page_id = page.id
    except Exception:
        pass
    return string_id, page_id


def translate_text(text: StringValue, source_language: str, target_language: str):
    translator_settings = AITranslatorSettings.load()

    provider = translator_settings.provider
    model = translator_settings.model
    if not model or not provider:
        return {
            "error": _("No provider or model configured"),
            "source_text": text.get_translatable_html(),
        }

    style_prompt = translator_settings.prompt
       
    SYSTEM_PROMPT = (
        f"Translate from {source_language} to {target_language}.\n"
        "Output ONLY the translated text. Do NOT output your reasoning, "
        "analysis, thought process, breakdown, or any commentary about the "
        "translation. Start directly with the first translated word — no "
        "introductory phrase, no markdown fences, no <p>, <html>, or <body> "
        "wrapping tags.\n\n"
        "Rules:\n"
        "- Keep all HTML tags and their id attributes exactly as in the source, same order, no new attributes.\n"
        "- Do not translate product/brand/feature names, URLs, code, or config keys.\n"
        "- Use natural target-language word order and formal register, not source word order.\n"
        "- Wrap clickable UI labels (buttons, menu items) in the target language's standard quotation convention "
        "(e.g. « » in Arabic/French) only when the text follows a verb like click/press/select. "
        "Don't wrap section headings or plain nouns.\n"
        "- Use one consistent term per concept throughout.\n"
        "- Text inside <b> or <i> tags marks a UI label/button. Keep that text in English verbatim (e.g. <b>Publish</b> stays <b>Publish</b>, not <b>«نشر»</b>). Do not translate or wrap it in guillemets."
    )
    if style_prompt:
        SYSTEM_PROMPT += f"\n\n#Style Instructions  \n{style_prompt}"
    
    SYSTEM_PROMPT += f"\n\nTranslate the following text to {target_language}."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": text.get_translatable_html()},
    ]

    client = get_llm_client(provider)
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.completion(
                model=model,
                temperature=0,
                messages=messages,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES and _is_rate_limit(e):
                # Backoff + jitter to stagger concurrent retries and avoid a
                # synchronized retry burst that just re-trips the limit.
                import random
                sleep_s = BACKOFF_BASE * (2**attempt) + random.uniform(0, 1)
                time.sleep(sleep_s)
                continue
            return {
                "error": str(e),
                "source_text": text.get_translatable_html(),
            }
    else:
        return {
            "error": str(last_error),
            "source_text": text.get_translatable_html(),
        }

    content = (response.choices[0].message.content or "").strip()
    usage = {
        "input_tokens": response.usage.prompt_tokens or 0,
        "output_tokens": response.usage.completion_tokens or 0,
    }

    source_text = text.get_translatable_html()

    if not content:
        return {"error": _("Translation failed"), "usage": usage, "source_text": source_text}

    lines = content.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].lstrip().startswith("```"):
        lines = lines[:-1]
    content = "\n".join(lines).strip()
    sanitized = sanitize_html(content)

    try:
        value = StringValue.from_translated_html(sanitized)
    except ValueError as e:
        return {"error": str(e), "usage": usage, "source_text": source_text, "translated_text": sanitized}

    return {
        "result": {text: value},
        "usage": usage,
        "source_text": source_text,
        "translated_text": sanitized,
    }
