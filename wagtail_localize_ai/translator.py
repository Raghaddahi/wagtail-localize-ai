import concurrent
from django.utils.translation import gettext_lazy as _, get_language_info
from bs4 import BeautifulSoup, NavigableString, Tag

from wagtail.models import Locale

from wagtail_localize.machine_translators.base import BaseMachineTranslator
from wagtail_localize.strings import INLINE_TAGS, StringValue

from wagtail_localize_ai.models import AITranslatorSettings, TranslationLog
from wagtail_localize_ai.utils import get_llm_client, get_provider_display_name, normalize_model_identifier


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

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor_results = list(
                executor.map(translate_text, strings, [source_language] * len(strings), [target_language] * len(strings))
            )
        
        results = {}
        error = ""
        for result in executor_results:
            if "usage" in result:
                translation_log.input_tokens += result["usage"]["input_tokens"]
                translation_log.output_tokens += result["usage"]["output_tokens"]
            if "error" in result:
                error += result["error"] + "\n"
                continue
            results.update(result["result"])

        translation_log.error = error
        translation_log.save()

        if not results and strings:
            raise RuntimeError(error.strip() or str(_("Translation failed")))

        return results

    def can_translate(self, source_locale: Locale, target_locale: Locale):
        translator_settings = AITranslatorSettings.load()
        if not translator_settings:
            return False
        has_provider = translator_settings.provider
        has_model = translator_settings.model
        not_same_language = source_locale.language_code != target_locale.language_code

        return has_provider and has_model and not_same_language

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

    try:
        client = get_llm_client(provider)
        response = client.completion(
            model=model,
            temperature=0,
            messages=messages,
        )
    except Exception as e:
        return {
            "error": str(e),
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
