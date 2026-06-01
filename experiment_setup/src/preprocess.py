from __future__ import annotations

import re
from dataclasses import dataclass


URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
WHITESPACE_RE = re.compile(r'\s+')
_EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001F5FF'
    '\U0001F600-\U0001F64F'
    '\U0001F680-\U0001F6FF'
    '\U0001F700-\U0001F77F'
    '\U0001F780-\U0001F7FF'
    '\U0001F800-\U0001F8FF'
    '\U0001F900-\U0001F9FF'
    '\U0001FA00-\U0001FAFF'
    '\U00002700-\U000027BF'
    '\U00002600-\U000026FF'
    ']+',
    flags=re.UNICODE,
)

_BASIC_NORMALIZER = None
_EMOJI_HANDLER = None
_LEX_NORMALIZER = None
_VISONORM_READY = None


@dataclass
class TextVariants:
    raw_text: str
    emoji_removed_text: str
    preprocessed_text: str
    preprocessed_emoji_removed_text: str


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(' ', text).strip()


def _init_visonorm(model_repo: str | None = None):
    global _BASIC_NORMALIZER, _EMOJI_HANDLER, _LEX_NORMALIZER, _VISONORM_READY
    if _VISONORM_READY is not None:
        return _VISONORM_READY

    try:
        from visonorm import BasicNormalizer, EmojiHandler, ViSoLexNormalizer

        _BASIC_NORMALIZER = BasicNormalizer()
        _EMOJI_HANDLER = EmojiHandler()
        if model_repo:
            _LEX_NORMALIZER = ViSoLexNormalizer(model_repo=model_repo)
        else:
            _LEX_NORMALIZER = ViSoLexNormalizer()
        _VISONORM_READY = True
    except Exception:
        _BASIC_NORMALIZER = None
        _EMOJI_HANDLER = None
        _LEX_NORMALIZER = None
        _VISONORM_READY = False
    return _VISONORM_READY


def _apply_basic_cleanup(text: str, settings: dict) -> str:
    value = str(text or '')
    if settings.get('strip_urls'):
        value = URL_RE.sub('', value)
    if settings.get('strip_mentions'):
        value = MENTION_RE.sub('', value)
    if settings.get('strip_hashtags'):
        value = HASHTAG_RE.sub(r'\1', value)
    return value


def _remove_emoji_only(text: str, settings: dict) -> str:
    model_repo = settings.get('visonorm_model_repo')
    split_emoji = settings.get('visonorm_split_emoji', True)
    if _init_visonorm(model_repo=model_repo):
        value = text
        if split_emoji:
            value = _EMOJI_HANDLER.split_emoji_text(value)
        try:
            value = _EMOJI_HANDLER.remove_emojis(value)
        except Exception:
            value = _EMOJI_RE.sub(' ', value)
        return normalize_whitespace(value)
    return normalize_whitespace(_EMOJI_RE.sub(' ', text))


def _preprocess_with_visonorm(text: str, settings: dict, remove_emoji: bool) -> str:
    model_repo = settings.get('visonorm_model_repo')
    split_emoji = settings.get('visonorm_split_emoji', True)
    lowercase = settings.get('lowercase', False)

    if not _init_visonorm(model_repo=model_repo):
        fallback = text.lower() if lowercase else text
        if remove_emoji:
            fallback = _EMOJI_RE.sub(' ', fallback)
        return normalize_whitespace(fallback)

    value = text.lower() if lowercase else text
    if split_emoji:
        value = _EMOJI_HANDLER.split_emoji_text(value)

    basic_tokens = _BASIC_NORMALIZER.basic_normalizer(
        value,
        case_folding=lowercase,
        mode='lower' if lowercase else 'lower',
        remove_emoji=remove_emoji,
        split_emoji=split_emoji,
    )
    if isinstance(basic_tokens, list):
        value = ' '.join(token for token in basic_tokens if token is not None)
    else:
        value = str(basic_tokens)

    try:
        value = _LEX_NORMALIZER.normalize_sentence(value)
    except Exception:
        pass

    if remove_emoji:
        try:
            value = _EMOJI_HANDLER.remove_emojis(value)
        except Exception:
            value = _EMOJI_RE.sub(' ', value)

    return normalize_whitespace(value)


def preprocess_text(text: str, settings: dict, remove_emoji: bool = False) -> str:
    value = _apply_basic_cleanup(text, settings)
    value = _preprocess_with_visonorm(value, settings, remove_emoji=remove_emoji)
    if settings.get('normalize_whitespace', True):
        value = normalize_whitespace(value)
    return value


def compose_text(sample: dict, include_ocr: bool, ocr_template: str) -> str:
    text = str(sample.get('text', '') or '')
    if include_ocr:
        ocr_text = str(sample.get('ocr_text', '') or '').strip()
        if ocr_text:
            text = text + ocr_template.format(ocr_text=ocr_text)
    return normalize_whitespace(text)


def build_text_variants(sample: dict, config: dict) -> TextVariants:
    raw_text = compose_text(
        sample=sample,
        include_ocr=config['data'].get('include_ocr_in_text', True),
        ocr_template=config['data'].get('ocr_template', '\n\n[OCR]\n{ocr_text}'),
    )
    settings = config.get('preprocessing', {}).get('text', {})
    emoji_removed_text = _remove_emoji_only(_apply_basic_cleanup(raw_text, settings), settings)
    preprocessed_text = preprocess_text(raw_text, settings, remove_emoji=False)
    preprocessed_emoji_removed_text = preprocess_text(raw_text, settings, remove_emoji=True)
    return TextVariants(
        raw_text=raw_text,
        emoji_removed_text=emoji_removed_text,
        preprocessed_text=preprocessed_text,
        preprocessed_emoji_removed_text=preprocessed_emoji_removed_text,
    )
