from __future__ import annotations

import re
from dataclasses import dataclass


URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
WHITESPACE_RE = re.compile(r'\s+')

_BASIC_NORMALIZER = None
_EMOJI_HANDLER = None
_LEX_NORMALIZER = None
_VISONORM_READY = None


@dataclass
class TextVariants:
    raw_text: str
    preprocessed_text: str


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


def _preprocess_with_visonorm(text: str, settings: dict) -> str:
    model_repo = settings.get('visonorm_model_repo')
    split_emoji = settings.get('visonorm_split_emoji', True)
    remove_emoji = settings.get('visonorm_remove_emoji', False)
    lowercase = settings.get('lowercase', False)

    if not _init_visonorm(model_repo=model_repo):
        return normalize_whitespace(text.lower() if lowercase else text)

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
            pass

    return normalize_whitespace(value)


def preprocess_text(text: str, settings: dict) -> str:
    value = _apply_basic_cleanup(text, settings)
    value = _preprocess_with_visonorm(value, settings)
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
    processed = preprocess_text(raw_text, config.get('preprocessing', {}).get('text', {}))
    return TextVariants(raw_text=raw_text, preprocessed_text=processed)
