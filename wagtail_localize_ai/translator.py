import concurrent
import time

from django.utils.translation import gettext_lazy as _, get_language_info
from bs4 import BeautifulSoup, NavigableString, Tag

from wagtail.models import Locale

from wagtail_localize.machine_translators.base import BaseMachineTranslator
from wagtail_localize.strings import INLINE_TAGS, StringValue

from wagtail_localize_ai.models import AITranslatorSettings, TranslationLog
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

        translation_log = TranslationLog(
            provider=translator_settings.provider,
            model=translator_settings.model,
            input_tokens=0,
            output_tokens=0,
            error=None,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor_results = list(
                executor.map(translate_text, strings, [source_language] * len(strings), [target_language] * len(strings))
            )
        
        results = {}
        error = ""
        seen_errors = []
        for result in executor_results:
            if "usage" in result:
                translation_log.input_tokens += result["usage"]["input_tokens"]
                translation_log.output_tokens += result["usage"]["output_tokens"]
            if "error" in result:
                err = result["error"]
                if err not in seen_errors:
                    seen_errors.append(err)
                error += err + "\n"
                continue
            results.update(result["result"])

        translation_log.error = error
        translation_log.save()

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


def translate_text(text: StringValue, source_language: str, target_language: str):
    translator_settings = AITranslatorSettings.load()

    provider = translator_settings.provider
    model = translator_settings.model
    if not model or not provider:
        return {
            "error": _("No provider or model configured"),
        }
    
    style_prompt = translator_settings.prompt
       
    SYSTEM_PROMPT = (
        "# Role\n"
        "You are an expert technical translator for the Wagtail CMS user guide.\n\n"
        "# Nature of the task\n"
        f"Translate the given {source_language} text to {target_language}. You are translating "
        "user-facing documentation, so UI labels, buttons, and HTML markup must be handled precisely.\n\n"
        "# Formatting\n"
        "Output ONLY the translated text. "
        "Do NOT output your reasoning, analysis, thought process, breakdown, or any commentary about the "
        "translation. Start directly with the first translated word — no introductory phrase, "
        "no markdown fences, no <p>, <html>, or <body> wrapping tags.\n\n"
        "# Guidelines\n"
        "- Keep all HTML tags and their id attributes exactly as in the source, same order, no new attributes.\n"
        "- Do not translate URLs, code, or config keys.\n"
        f"- Use natural {target_language} word order and formal register, not {source_language} word order.\n"
        "- Use one consistent term per concept throughout.\n"
        f"- Text inside <b> or <i> tags marks a UI label/button. Keep that text in {source_language} verbatim "
        "(e.g. <b>Publish</b> stays <b>Publish</b>, not <b>«نشر»</b>). Do not translate or wrap it in guillemets.\n"
        f"- Translate plain prose and <a> link text (NOT inside <b>/<i>) to {target_language}. "
        f"Only text inside <b>/<i> is a UI label that stays in {source_language}."
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
    # DeepSeek-V4-flash is a reasoning model; with reasoning left at the
    # gateway default it spends ~15K output tokens "thinking" on short strings
    # and hits the output cap, truncating the actual translation. "low" keeps a
    # small reasoning budget while leaving room for the translated text. GLM
    # and other models stay on the provider default ("auto").
    completion_kwargs = {"reasoning_effort": "low"} if "deepseek" in (model or "").lower() else {}
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.completion(
                model=model,
                temperature=0,
                messages=messages,
                **completion_kwargs,
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
            }
    else:
        return {
            "error": str(last_error),
        }

    content = (response.choices[0].message.content or "").strip()
    usage = {
        "input_tokens": response.usage.prompt_tokens or 0,
        "output_tokens": response.usage.completion_tokens or 0,
    }

    if not content:
        return {"error": _("Translation failed"), "usage": usage}

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
        return {"error": str(e), "usage": usage}

    return {
        "result": {text: value},
        "usage": usage,
    }
