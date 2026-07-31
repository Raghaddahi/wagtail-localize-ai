"""Model-agnostic extraction of the real translation from an LLM response.

Reasoning-capable models served via OpenAI-compatible gateways sometimes emit
their chain-of-thought inline in ``message.content`` (because the gateway does
not surface it in a separate ``reasoning_content`` / ``thinking`` field that
any-llm knows how to strip). This module isolates the genuine translation by
keying off the source-string contract (inline-tag ids must survive in the
translation) and the target language's script, rather than guessing how each
model formats its reasoning.
"""
import re

# Known reasoning "envelope" formats across providers (gpt-oss, DeepSeek,
# Anthropic-serialized, Qwen, Mistral, OpenAI, ...). These are stripped first;
# free-prose preamble without delimiters is handled by the id+script anchoring
# below.
_REASONING_BLOCK_PATTERNS = [
    r"",
    r"",
    r"</?think(?:ing)?>",
    r"<\|channel\|>thought.*?<\|channel\|>",
    r"<\|/?think(?:ing)?\|>",
    r"<reasoning>.*?</reasoning>",
    r"<analysis>.*?</analysis>",
    r"<reflection>.*?</reflection>",
    r"<thoughts?>.*?</thoughts?>",
    r"```(?:thinking|reasoning|thought)\b.*?```",
]
_REASONING_RE = re.compile("|".join(_REASONING_BLOCK_PATTERNS), re.DOTALL | re.IGNORECASE)

_ID_RE = re.compile(r'<[a-z]+\b[^>]*\bid="([a-zA-Z0-9_]+)"', re.IGNORECASE)
_ID_OCCURRENCE_RE = lambda id_str: re.compile(
    r'id="%s"' % re.escape(id_str), re.IGNORECASE
)


def strip_reasoning_blocks(text):
    return _REASONING_RE.sub("", text)


def char_script(ch):
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF:
        return "arabic"
    if 0x0370 <= o <= 0x03FF:
        return "greek"
    if 0x0400 <= o <= 0x04FF:
        return "cyrillic"
    if 0x0590 <= o <= 0x05FF:
        return "hebrew"
    if 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
        return "cjk"
    if 0xAC00 <= o <= 0xD7AF:
        return "hangul"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if 0x0E00 <= o <= 0x0E7F:
        return "thai"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return "other"


def _local_density(content, center, window=140):
    lo = max(0, center - window)
    hi = min(len(content), center + window)
    counts = {}
    for ch in content[lo:hi]:
        s = char_script(ch)
        if s != "other":
            counts[s] = counts.get(s, 0) + 1
    return counts


def _detect_target_script(content, anchors):
    scripts = {}
    for a in anchors:
        for s, n in _local_density(content, a, window=180).items():
            if s != "latin" and s != "other":
                scripts[s] = scripts.get(s, 0) + n
    if scripts:
        return max(scripts, key=scripts.get)
    return "latin"


def _pick_occurrence(content, id_str, target_script):
    best_pos, best_score = None, -(10**9)
    for m in _ID_OCCURRENCE_RE(id_str).finditer(content):
        c = _local_density(content, m.start())
        score = c.get(target_script, 0) - c.get("latin", 0)
        if score > best_score:
            best_pos, best_score = m.start(), score
    return best_pos


def _tokenize(content):
    tokens = []
    i, n = 0, len(content)
    while i < n:
        if content[i] == "<":
            j = content.find(">", i)
            if j == -1:
                tokens.append(("text", content[i:], i, n))
                break
            tokens.append(("tag", content[i : j + 1], i, j + 1))
            i = j + 1
        else:
            j = content.find("<", i)
            if j == -1:
                j = n
            tokens.append(("text", content[i:j], i, j))
            i = j
    return tokens


def _latin_word_count(text):
    return sum(1 for w in text.split() if sum(1 for ch in w if char_script(ch) == "latin") >= 2)


def _has_target_script(text):
    return any(char_script(ch) not in ("latin", "other") for ch in text)


def _is_reasoning_text(text):
    # Substantial Latin prose with NO target script -> reasoning preamble/epilogue.
    # Threshold >=5 latin words keeps short English UI labels kept verbatim
    # inside <b>/<i> (e.g. "Open in new tab" = 4 words) from being mistaken for
    # reasoning, while real chain-of-thought tokens (typically many words) still
    # match. Boundary tokens where reasoning and translation share a single text
    # token are handled separately by _cut_leading_reasoning regardless of length.
    if not text.strip():
        return False
    return _latin_word_count(text) >= 5 and not _has_target_script(text)


def _cut_leading_reasoning(text):
    """Trim a Latin reasoning prefix that precedes the target-script
    translation within a mixed text token (the tokenizer only splits on '<',
    so reasoning prose and the translation can share one text token).

    Only the prefix *before the first target-script character* is considered,
    and only trimmed if it is substantial Latin prose (>=3 latin words). This
    keeps single embedded English brand/feature names that sit mid-translation
    (e.g. "في نظام macOS، يمكنك...") intact: the first target char there is the
    leading Arabic, so there is no Latin prefix to trim.
    """
    first_target = -1
    for i, ch in enumerate(text):
        s = char_script(ch)
        if s != "latin" and s != "other":
            first_target = i
            break
    if first_target == -1:
        return text  # no target script in this token; leave intact
    prefix = text[:first_target]
    if _latin_word_count(prefix) >= 3:
        cut = first_target
        while cut < len(text) and char_script(text[cut]) == "other":
            cut += 1
        return text[cut:]
    return text


def _cut_trailing_reasoning(text):
    """Drop a Latin reasoning suffix from a mixed text token (text token that
    ends with substantial Latin prose after the target-script translation)."""
    last_target = -1
    for i, ch in enumerate(text):
        s = char_script(ch)
        if s != "latin" and s != "other":
            last_target = i
    if last_target == -1:
        return text
    j = last_target + 1
    while j < len(text):
        if char_script(text[j]) == "latin":
            run_start = j
            while j < len(text) and char_script(text[j]) == "latin":
                j += 1
            run = text[run_start:j]
            if _latin_word_count(run) >= 3:
                return text[:run_start].rstrip()
        else:
            j += 1
    return text


def _collapse_loops(text):
    s = text.strip()
    n = len(s)
    if n < 30:
        return s
    f = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[i] != s[k]:
            k = f[k - 1]
        if s[i] == s[k]:
            k += 1
        f[i] = k
    period = n - f[n - 1]
    if 5 <= period < n and n // period >= 3:
        # The candidate is `period`-unit repeated >=3 times. A trailing
        # partial copy of the unit (n % period != 0) is also collapsed to a
        # single unit -- the reasoning-loop failure mode often ends mid-unit.
        candidate = s[:period]
        full_len = (n // period) * period
        if n == full_len or _starts_with_repeated_unit(s, candidate, full_len):
            return candidate
    return s


def _starts_with_repeated_unit(s, unit, full_len):
    """True if s[:full_len] equals unit repeated, ignoring whitespace-only
    differences at the seam (loops are often space-or-newline separated)."""
    if not unit:
        return False
    # Build expected without consuming memory: compare in chunks.
    ulen = len(unit)
    for off in range(0, full_len, ulen):
        seg = s[off : off + ulen]
        if seg != unit:
            # allow a single whitespace mismatch at seam boundaries
            if seg.rstrip() == unit.rstrip() and off + ulen <= full_len:
                continue
            return False
    return True


def extract_translation(content, source_html):
    """Isolate the real translation from a possibly-leaky model response.

    Args:
        content: raw model ``message.content``.
        source_html: ``StringValue.get_translatable_html()`` for this segment.

    Returns:
        Cleaned translation string (may still pass through the standard
        fence-strip + sanitize + from_translated_html pipeline downstream,
        which validates structure).
    """
    if not content:
        return ""
    ids = [m.group(1) for m in _ID_RE.finditer(source_html)]
    content = strip_reasoning_blocks(content)
    if not ids:
        # No id anchor available (plain-text segment): collapse any loop the
        # model may have produced at temperature=0, then return.
        return _collapse_loops(content).strip()

    anchors = []
    for id_str in set(ids):
        for m in _ID_OCCURRENCE_RE(id_str).finditer(content):
            anchors.append(m.start())
    # Contract guard: when the source segment carries inline ids the
    # translation must echo them (so downstream render-html can restore attrs
    # like href); a translation that dropped them is not usable even if it
    # parses as valid HTML. Some models (notably MiniMax-M2.7) systematically
    # ignore the "keep HTML tags and id attributes" rule and also fixate-loop
    # their Arabic output. Rather than save tag-stripped garbage, fail the
    # segment so the operator sees the error and switches models. Any source id
    # missing from the cleaned output -> fail.
    missing = [i for i in set(ids) if i not in [m.group(1) for m in _ID_RE.finditer(content)]]
    if missing:
        return ""

    target_script = _detect_target_script(content, anchors)

    chosen = []
    for id_str in ids:
        pos = _pick_occurrence(content, id_str, target_script)
        if pos is None:
            # id absent altogether -> can't anchor; fail the segment rather
            # than save content that violates the structure contract.
            return ""
        chosen.append(pos)
    first_pos, last_pos = chosen[0], chosen[-1]

    tokens = _tokenize(content)

    def token_at(pos):
        for idx, t in enumerate(tokens):
            if t[2] <= pos < t[3]:
                return idx
        return None

    ft = token_at(first_pos)
    lt = token_at(last_pos)
    if ft is None or lt is None or lt < ft:
        return content.strip()

    # Backward walk to translation start.
    start_tok = ft
    k = ft - 1
    while k >= 0:
        t = tokens[k]
        if t[0] == "tag":
            start_tok = k
        elif _is_reasoning_text(t[1]):
            break
        else:
            start_tok = k
        k -= 1

    # Forward walk to translation end.
    end_tok = lt
    k = lt + 1
    while k < len(tokens):
        t = tokens[k]
        if t[0] == "tag":
            end_tok = k
        elif _is_reasoning_text(t[1]):
            break
        else:
            end_tok = k
        k += 1

    # Build the candidate from the selected token spans.
    # The tokenizer splits on every '<', so "<b>Snippets</b>" becomes three
    # tokens: <b>, "Snippets", </b>. Reasoning often quotes the source incl. its
    # inline tags (e.g. "keep <b>Snippets</b> as-is."), so those leading tag
    # tokens + their short Latin content belong to reasoning, not the
    # translation. We drop everything before the first text token whose
    # _cut_leading_reasoning trimmed a real Latin prefix: that prefix is the
    # reasoning prose, and the tags/text before it were attached to it.
    seg_tokens = tokens[start_tok : end_tok + 1]

    # Same-script recovery: if the backward walk broke at a reasoning text
    # token that is *immediately* followed by the first id tag (tokens[ft]),
    # the reasoning prose sometimes ends with the actual leading word(s) of
    # the translation glued to the id tag with a sentence boundary in between
    # (e.g. EN->FR "...now. Cliquez <a id="a1">ici</a>..."). When the token
    # after the reasoning one is the id tag itself, cut the reasoning token at
    # its last sentence boundary and carry the tail as the translation prefix.
    # Skipped when the token after the reasoning one is a non-id tag (e.g.
    # "keep <b>Snippets</b>" from a reasoning quote) so reasoning objects are
    # not pulled into the translation.
    leading_text_prefix = ""
    if (
        start_tok == ft
        and ft - 1 >= 0
        and tokens[ft - 1][0] == "text"
        and _is_reasoning_text(tokens[ft - 1][1])
    ):
        prev = tokens[ft - 1][1]
        cut_at = -1
        for mm in re.finditer(r"[.!?]\s+", prev):
            cut_at = mm.end()
        if cut_at == -1:
            idx = prev.rfind("\n")
            if idx != -1:
                cut_at = idx + 1
        if cut_at >= 0:
            tail = prev[cut_at:]
            if tail.strip() and _latin_word_count(tail) < 5:
                leading_text_prefix = tail

    lead_cut = None
    for i, t in enumerate(seg_tokens):
        if t[0] == "text" and _cut_leading_reasoning(t[1]) != t[1]:
            lead_cut = i
            break
    if lead_cut is not None and lead_cut > 0:
        seg_tokens = seg_tokens[lead_cut:]
    trail_cut = None
    for i in range(len(seg_tokens) - 1, -1, -1):
        t = seg_tokens[i]
        if t[0] == "text" and _cut_trailing_reasoning(t[1]) != t[1]:
            trail_cut = i
            break
    if trail_cut is not None and trail_cut < len(seg_tokens) - 1:
        seg_tokens = seg_tokens[: trail_cut + 1]

    text_indices = [i for i, t in enumerate(seg_tokens) if t[0] == "text"]
    parts = []
    for i, t in enumerate(seg_tokens):
        seg = t[1]
        if t[0] == "text":
            seg = _cut_leading_reasoning(seg)
            if text_indices and i == text_indices[-1]:
                seg = _cut_trailing_reasoning(seg)
        parts.append(seg)
    candidate = "".join(parts)
    if leading_text_prefix:
        candidate = leading_text_prefix + candidate

    # Final targeted post-clean: a leading element like "<b>Snippets</b>." (tag
    # + sentence-ending punctuation) that survived because the next text token
    # began directly with target script (no Latin prefix to detect). Only strip
    # when the remainder truly begins with the target script so legitimate
    # leading UI labels like "<b>Publish</b> هذه الصفحة" are preserved.
    candidate = _strip_leading_reasoning_tag(candidate, target_script)
    candidate = _strip_trailing_reasoning_tag(candidate, target_script)

    return _collapse_loops(candidate).strip()


_LEADING_REASONING_TAG_RE = re.compile(r"^(<[^>]+>)\s*[.!?]?\s*")
_TRAILING_REASONING_TAG_RE = re.compile(r"\s*[.!?]?\s*(<[^>]+>)\s*$")


def _strip_leading_reasoning_tag(text, target_script):
    if target_script == "latin" or not text:
        return text
    while True:
        m = _LEADING_REASONING_TAG_RE.match(text)
        if not m:
            break
        rest = text[m.end() :]
        stripped = rest.lstrip()
        if not stripped:
            text = rest
            continue
        if ' id="' in m.group(1):
            break
        if char_script(stripped[0]) == target_script:
            text = rest
        else:
            break
    return text


def _strip_trailing_reasoning_tag(text, target_script):
    if target_script == "latin" or not text:
        return text
    while True:
        m = _TRAILING_REASONING_TAG_RE.search(text)
        if not m:
            break
        head = text[: m.start()]
        stripped = head.rstrip()
        if not stripped:
            break
        if ' id="' in m.group(1):
            break
        if char_script(stripped[-1]) == target_script:
            text = head
        else:
            break
    return text