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
        "You are a professional translator that translates text from "
        f"{source_language} to {target_language}.\n"
        "Reply with ONLY the translated text, no preamble, no commentary, "
        "no markdown fences, no wrapping in <p> or <html> or <body>.\n\n"
        "# Output integrity\n"
        "Preserve the exact inline HTML tags that appear in the source: "
        "the output must contain the same set of tag names and the same `id` "
        "attributes as the source, in the same order. Do not add, rename, "
        "translate, or drop any `id` attribute.\n"
        "Do NOT add any attributes the source did not have (no href, class, "
        "title, target, style, etc.). The only attribute allowed is `id` on "
        "<a> tags, and it must keep its original value verbatim.\n"
        "Do NOT introduce block-level tags (<p>, <div>, <ul>, <li>, <br>, ...). "
        "Only pass through the inline tags already present (<a>, <b>, <i>, "
        "<em>, <strong>, <code>, <abbr>, <acronym>).\n"
        "If the text is slugified, keep it slugified.\n\n"
        "# Terminology\n"
        "Keep product names, feature names, and brands untranslated exactly as "
        "written in the source (e.g. \"Breads\" stays \"Breads\", never translate "
        "to a literal equivalent like \"الخبز\"). URLs, email addresses, code "
        "identifiers, and configuration keys stay verbatim.\n\n"
        "# Language and style (apply to every target language)\n"
        "Use the natural, formal written register of the target language for "
        "UI/instructional text.\n"
        "Follow the target language's natural word order; do not mirror the "
        "source language's word order. For example, in Arabic use the noun-first "
        "construct \"زر «إضافة»\" not \"إضافة الزر\".\n"
        "Prefer the concise, formal word where alternatives exist (e.g. "
        "\"تتوفر\" over \"توجد\" in Arabic).\n"
        "Use the target language's standard typographic conventions for quoting "
        "UI labels (e.g. guillemets « » in Arabic and French; keep other "
        "languages' established convention).\n"
        "Keep one consistent term for each concept across the text; do not mix "
        "synonyms for the same UI element.\n"
        "Mark parent/child relationships with the target language's natural "
        "construction (e.g. Arabic \"تابعة لـ\").\n"
        "Do not transliterate product/feature names unless they are an "
        "established loanword in the target language.\n"
        "Preserve the exact number, case, and punctuation style of the source "
        "where the target language allows; do not add headings, lists, or "
        "commentary."
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
