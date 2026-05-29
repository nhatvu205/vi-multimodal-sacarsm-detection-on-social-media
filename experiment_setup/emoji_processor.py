"""
emoji_processor.py
------------------
Module xử lý emoji và emoticon cho bài toán Multimodal Sarcasm Detection
Hỗ trợ: emoticon ASCII (:))), =(((), từ lóng tiếng Việt

Chiến lược:
- Emoji Unicode → GIỮ NGUYÊN (VLM tự tokenize và hiểu được)
- Emoticon ASCII → thay thế bằng emoji + nhãn cường độ
- Từ lóng tiếng Việt → thay thế bằng emoji + nhãn
- Text–Emoji conflict → boost sarcasm score theo độ lệch cực tính

Conflict detection:
  Nhận text_sentiment_score từ model ngoài (vd: PhoBERT) trong [-1, +1].
  Tính emoji_polarity trung bình từ SARCASM_SIGNAL_EMOJI.
  conflict_magnitude = max(0, -(text_pol × emoji_pol))
  sarcasm_score_final = min(1.0, base_score + α × conflict_magnitude)
  α mặc định = 0.5, có thể tune qua CONFLICT_ALPHA.

Cài đặt:
pip install emoji
"""

import re
from dataclasses import dataclass, field
from typing import Optional
import unicodedata

try:
    import emoji as emoji_lib
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False
    print("[WARNING] Thư viện 'emoji' chưa được cài. Chạy: pip install emoji")


# ─────────────────────────────────────────────
#  1. CẤU HÌNH
# ─────────────────────────────────────────────

# Emoticon ASCII → (emoji Unicode, nhãn ngữ nghĩa, cực tính)
# cực tính: +1 = tích cực, -1 = tiêu cực, 0 = trung tính
EMOTICON_BASE_MAP = {
    # Vui / cười
    r"[:=;8]-?[)D\]>}]+":        ("😄", "vui",          +1),
    r"[:=;]-?[pP]+":              ("😛", "tinh nghịch",   +1),
    r"[:=;]-?\*+":                ("😘", "hôn",           +1),
    r"[xX]-?D+":                  ("😆", "cười lớn",      +1),
    r"\^[-_]?\^":                 ("😊", "vui nhẹ",       +1),
    r">\s*[:<]-?\s*\)":           ("😈", "tinh quái",      0),

    # Buồn / khóc
    r"[:=;]-?[\(\[{<]+":          ("😢", "buồn",          -1),
    r"[;:]-?'[\(\[]+":            ("😭", "khóc",          -1),
    r"T[-_]?T":                   ("😭", "khóc",          -1),
    r"Q[-_]?Q":                   ("😭", "khóc nhiều",    -1),

    # Ngạc nhiên / sốc
    r"[:=]-?[oO0]+":              ("😮", "ngạc nhiên",     0),
    r"[oO][._][oO]":              ("😲", "sốc",            0),

    # Tức giận / khó chịu
    r">[:=]-?[\(\[]+":            ("😠", "tức giận",      -1),
    r"[:=]-?[\/\\|]+":            ("😒", "khó chịu",      -1),

    # Lạnh lùng / thản nhiên
    r"[:=]-?\|":                  ("😐", "thản nhiên",     0),
    r"-[-_]+-":                   ("😑", "vô cảm",         0),

    # Nháy mắt
    r";-?[)D\]]+":                ("😉", "nháy mắt",      +1),

    # Mệt / ngủ
    r"-[-_]?[oO0zZ]":             ("😴", "buồn ngủ",      -1),
}

# Từ lóng / biểu cảm tiếng Việt → (nhãn, cực tính)
VIET_SLANG_MAP = {
    # Vui
    r"\b(hehe|hehehe+|hehe+)\b":        ("😄 [cười]",       +1),
    r"\b(hihi|hihihi+)\b":              ("😊 [vui nhẹ]",    +1),
    r"\b(haha|hahaha+)\b":              ("😂 [cười lớn]",   +1),
    r"\b(huhu|huhuu+|huhuhu+)\b":       ("😢 [khóc]",       -1),
    r"\b(hix|hixhix|hic+)\b":           ("😞 [buồn]",       -1),
    r"\b(uh+|ừ+h+)\b":                  ("😑 [thờ ơ]",       0),
    r"\b(oke+|oce+|okay+)\b":           ("👍 [đồng ý]",     +1),
    r"\b(wow+|woa+h?)\b":               ("😲 [ngạc nhiên]",  0),
    r"\b(ew+|eww+)\b":                  ("🤢 [ghê]",         -1),
    r"\b(ugh+)\b":                       ("😩 [chán]",        -1),
}

# Hệ số khuếch đại khi text–emoji conflict (tunable)
CONFLICT_ALPHA: float = 0.5

# Cực tính của emoji Unicode: dương = tích cực, âm = tiêu cực, 0 = trung tính
# Dùng để so sánh với text_sentiment_score từ model ngoài
EMOJI_POLARITY_MAP: dict[str, float] = {
    # Tích cực
    "😄": +0.9, "😊": +0.8, "😂": +0.8, "🥰": +0.9, "😍": +0.9,
    "😁": +0.8, "🤩": +0.9, "😃": +0.8, "😀": +0.8, "😆": +0.8,
    "🥳": +0.9, "😻": +0.9, "💕": +0.8, "❤️": +0.8, "👍": +0.7,
    "🎉": +0.8, "✨": +0.6, "💯": +0.7, "🙌": +0.7, "😘": +0.8,
    # Tiêu cực
    "😢": -0.8, "😭": -0.9, "😠": -0.8, "😡": -0.9, "😤": -0.7,
    "😒": -0.7, "🙄": -0.8, "😑": -0.6, "😬": -0.6, "🤢": -0.8,
    "🤮": -0.9, "💀": -0.7, "☠️": -0.7, "😩": -0.7, "😫": -0.7,
    "😞": -0.8, "😔": -0.7, "😟": -0.7, "🥺": -0.5, "😣": -0.7,
    # Trung tính / ngữ cảnh phụ thuộc
    "😐": 0.0, "😶": 0.0, "🤔": 0.0, "😮": 0.0, "😲": 0.0,
    "👀": 0.0, "🤷": 0.0, "😏": -0.2,  # hơi lệch tiêu cực vì hay dùng mỉa mai
    "👏": -0.1,                          # hay dùng mỉa mai
}

# Emoji có signal sarcasm cao — dùng để tính sarcasm score
# Không dùng để modify text, chỉ dùng để trích features
SARCASM_SIGNAL_EMOJI = {
    "🙄": 0.90,   # rolling eyes
    "😒": 0.85,   # unamused
    "🤡": 0.85,   # clown
    "💀": 0.75,   # skull / dead
    "😬": 0.75,   # grimacing
    "🫠": 0.80,   # melting face
    "😑": 0.70,   # expressionless
    "🤨": 0.70,   # raised eyebrow
    "😏": 0.75,   # smirking
    "👏": 0.60,   # clap (thường dùng mỉa mai)
    "🐸": 0.65,   # frog (meme Kermit)
    "☠️": 0.70,
    "💅": 0.65,
}


# ─────────────────────────────────────────────
#  2. DATACLASS KẾT QUẢ
# ─────────────────────────────────────────────

@dataclass
class ProcessedText:
    original: str
    processed: str                          # emoji gốc được giữ nguyên
    emoticon_features: dict = field(default_factory=dict)
    emoji_features: dict = field(default_factory=dict)
    sarcasm_score: float = 0.0              # 0.0 → 1.0, base từ emoji signal
    sarcasm_score_with_conflict: float = 0.0  # sarcasm_score sau khi boost conflict
    conflict_boost: float = 0.0             # phần điểm được boost do conflict
    sentiment_polarity: float = 0.0         # -1.0 → +1.0, từ emoticon + slang


# ─────────────────────────────────────────────
#  3. XỬ LÝ EMOTICON ASCII
# ─────────────────────────────────────────────

def _count_intensity(emoticon_str: str) -> int:
    """Đếm số ký tự cảm xúc lặp lại (vd: :)))) → 4)"""
    match = re.search(r"[)(\]DpP\*oO|\\\/]+$", emoticon_str)
    if match:
        return len(match.group(0))
    return 1


def _intensity_label(intensity: int, polarity: int) -> str:
    """Chuyển cường độ + cực tính → nhãn tiếng Việt"""
    if polarity > 0:
        if intensity >= 5: return "rất rất vui"
        if intensity >= 3: return "rất vui"
        return "vui"
    elif polarity < 0:
        if intensity >= 5: return "rất rất buồn"
        if intensity >= 3: return "rất buồn"
        return "buồn"
    return "trung tính"


def process_emoticons(text: str) -> "tuple[str, dict]":
    """
    Thay thế emoticon ASCII bằng emoji + nhãn có cường độ.
    Emoji Unicode trong text KHÔNG bị động đến.

    Returns:
        (text đã xử lý, dict features)
    """
    features = {
        "emoticon_count": 0,
        "max_intensity": 0,
        "polarity_sum": 0,
        "has_high_intensity": False,   # intensity >= 4
        "found": [],
    }

    result = text

    for pattern, (emoji_char, base_label, polarity) in EMOTICON_BASE_MAP.items():
        matches = re.findall(pattern, result, flags=re.IGNORECASE)
        for m in matches:
            intensity = _count_intensity(m)
            label = _intensity_label(intensity, polarity)
            replacement = f"{emoji_char} [{label}]"

            result = result.replace(m, replacement, 1)

            features["emoticon_count"] += 1
            features["polarity_sum"] += polarity * intensity
            features["max_intensity"] = max(features["max_intensity"], intensity)
            if intensity >= 4:
                features["has_high_intensity"] = True
            features["found"].append({
                "original": m,
                "replacement": replacement,
                "intensity": intensity,
                "polarity": polarity,
            })

    return result, features


# ─────────────────────────────────────────────
#  4. XỬ LÝ TỪ LÓNG TIẾNG VIỆT
# ─────────────────────────────────────────────

def process_viet_slang(text: str) -> "tuple[str, int]":
    """
    Thay thế từ lóng biểu cảm tiếng Việt.
    Emoji Unicode trong text KHÔNG bị động đến.

    Returns:
        (text đã xử lý, số lượng thay thế)
    """
    result = text
    count = 0
    for pattern, (replacement, _polarity) in VIET_SLANG_MAP.items():
        new_result, n = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
        count += n
        result = new_result
    return result, count


# ─────────────────────────────────────────────
#  5. TRÍCH FEATURES TỪ EMOJI UNICODE (không modify text)
# ─────────────────────────────────────────────

def extract_emojis(text: str) -> list[str]:
    """Trích xuất tất cả emoji Unicode trong text."""
    if not EMOJI_LIB_AVAILABLE:
        # Fallback: dùng unicode category
        return [ch for ch in text if unicodedata.category(ch) in ("So", "Sm")]
    return [ch for ch in text if ch in emoji_lib.EMOJI_DATA]


def compute_sarcasm_score(text: str) -> float:
    """
    Tính sarcasm score dựa trên emoji signal trong text GỐC.
    Score = trung bình cộng của signal các emoji được tìm thấy, max 1.0.
    Chỉ dùng để trích features, không modify text.
    """
    emojis_found = extract_emojis(text)
    if not emojis_found:
        return 0.0
    signals = [SARCASM_SIGNAL_EMOJI[e] for e in emojis_found if e in SARCASM_SIGNAL_EMOJI]
    if not signals:
        return 0.0
    return round(min(1.0, sum(signals) / len(emojis_found)), 4)


def get_emoji_features(text: str) -> dict:
    """
    Trích xuất features từ emoji Unicode trong text GỐC.
    Chỉ dùng để trích features, không modify text.
    """
    emojis = extract_emojis(text)
    sarcasm_emojis = [e for e in emojis if e in SARCASM_SIGNAL_EMOJI]
    return {
        "emoji_count": len(emojis),
        "unique_emoji_count": len(set(emojis)),
        "sarcasm_emoji_count": len(sarcasm_emojis),
        "sarcasm_emoji_list": sarcasm_emojis,
        "has_sarcasm_emoji": len(sarcasm_emojis) > 0,
    }


# ─────────────────────────────────────────────
#  6. CONFLICT DETECTION: TEXT vs EMOJI
# ─────────────────────────────────────────────

def compute_emoji_polarity(text: str) -> Optional[float]:
    """
    Tính cực tính trung bình của các emoji Unicode có trong text.
    Chỉ tính những emoji có trong EMOJI_POLARITY_MAP.

    Returns:
        float trong [-1, +1], hoặc None nếu không có emoji nào được nhận dạng.
    """
    emojis = extract_emojis(text)
    polarities = [EMOJI_POLARITY_MAP[e] for e in emojis if e in EMOJI_POLARITY_MAP]
    if not polarities:
        return None
    return round(sum(polarities) / len(polarities), 4)


def compute_conflict_boost(
    text_sentiment: float,
    emoji_polarity: Optional[float],
    base_sarcasm_score: float,
    alpha: float = CONFLICT_ALPHA,
) -> "tuple[float, float]":
    """
    Tính boost sarcasm score khi text và emoji có cực tính trái chiều.

    Logic:
        conflict_magnitude = max(0, -(text_sentiment × emoji_polarity))
        → Tích âm khi hai chiều ngược nhau; magnitude lớn = mâu thuẫn rõ hơn.
        boost = alpha × conflict_magnitude
        final_score = min(1.0, base_sarcasm_score + boost)

    Args:
        text_sentiment:    Score từ model ngoài (PhoBERT, v.v.), trong [-1, +1].
                           +1 = rất tích cực, -1 = rất tiêu cực.
        emoji_polarity:    Output của compute_emoji_polarity(), None nếu không có emoji.
        base_sarcasm_score: sarcasm_score tính từ SARCASM_SIGNAL_EMOJI.
        alpha:             Hệ số khuếch đại, mặc định CONFLICT_ALPHA = 0.5.

    Returns:
        (sarcasm_score_final, conflict_boost)
        Cả hai đều là 0 nếu emoji_polarity là None (không có emoji để so sánh).

    Ví dụ:
        text="vui quá" → text_sentiment=+0.8
        emoji=🙄       → emoji_polarity=-0.8
        conflict_magnitude = max(0, -(0.8 × -0.8)) = 0.64
        boost = 0.5 × 0.64 = 0.32
    """
    if emoji_polarity is None:
        return base_sarcasm_score, 0.0

    conflict_magnitude = max(0.0, -(text_sentiment * emoji_polarity))
    boost = round(alpha * conflict_magnitude, 4)
    final_score = round(min(1.0, base_sarcasm_score + boost), 4)
    return final_score, boost



def process(
    text: str,
    text_sentiment: Optional[float] = None,
    conflict_alpha: float = CONFLICT_ALPHA,
    keep_intensity_label: bool = True,
) -> ProcessedText:
    """
    Pipeline: emoticon ASCII → slang tiếng Việt → trích emoji features → conflict detection.
    Emoji Unicode được GIỮ NGUYÊN để VLM tự tokenize và hiểu.

    Args:
        text:                 Văn bản đầu vào.
        text_sentiment:       Sentiment score từ model ngoài (PhoBERT, v.v.),
                              trong [-1, +1]. Nếu None, bỏ qua conflict detection.
        conflict_alpha:       Hệ số khuếch đại conflict boost (mặc định CONFLICT_ALPHA).
        keep_intensity_label: Giữ nhãn cường độ trong text (vd: [rất vui]).

    Returns:
        ProcessedText với text đã xử lý, các features, và sarcasm_score_with_conflict.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Trích emoji features từ text GỐC (trước khi bất kỳ xử lý nào)
    emoji_features = get_emoji_features(text)
    base_sarcasm_score = compute_sarcasm_score(text)

    # 2. Xử lý emoticon ASCII → emoji + nhãn
    processed, emoticon_features = process_emoticons(text)

    # 3. Xử lý từ lóng tiếng Việt → emoji + nhãn
    processed, slang_count = process_viet_slang(processed)
    emoticon_features["viet_slang_count"] = slang_count

    # Emoji Unicode trong `processed` vẫn giữ nguyên từ bước 2 và 3,
    # vì process_emoticons và process_viet_slang chỉ match ASCII patterns.

    # 4. Tính sentiment polarity tổng hợp (từ emoticon + slang, không tính emoji)
    polarity = emoticon_features.get("polarity_sum", 0)
    normalized_polarity = max(-1.0, min(1.0, polarity / 10.0))

    # 5. Conflict detection: text_sentiment (model ngoài) vs emoji polarity
    conflict_boost = 0.0
    sarcasm_score_with_conflict = base_sarcasm_score
    if text_sentiment is not None:
        emoji_pol = compute_emoji_polarity(text)
        sarcasm_score_with_conflict, conflict_boost = compute_conflict_boost(
            text_sentiment=text_sentiment,
            emoji_polarity=emoji_pol,
            base_sarcasm_score=base_sarcasm_score,
            alpha=conflict_alpha,
        )

    return ProcessedText(
        original=text,
        processed=processed,
        emoticon_features=emoticon_features,
        emoji_features=emoji_features,
        sarcasm_score=base_sarcasm_score,
        sarcasm_score_with_conflict=sarcasm_score_with_conflict,
        conflict_boost=conflict_boost,
        sentiment_polarity=normalized_polarity,
    )


def batch_process(texts: list[str], **kwargs) -> list[ProcessedText]:
    """Xử lý một batch văn bản."""
    return [process(t, **kwargs) for t in texts]


def get_feature_vector(result: ProcessedText) -> dict:
    """
    Tổng hợp tất cả features thành một dict phẳng
    để đưa vào classifier hoặc concat với embedding.

    Lưu ý: sarcasm_score_final = sarcasm_score_with_conflict nếu đã truyền
    text_sentiment vào process(), ngược lại bằng sarcasm_score (base).
    """
    return {
        # Emoticon ASCII features
        "emoticon_count":            result.emoticon_features.get("emoticon_count", 0),
        "emoticon_max_intensity":    result.emoticon_features.get("max_intensity", 0),
        "emoticon_polarity_sum":     result.emoticon_features.get("polarity_sum", 0),
        "emoticon_high_intensity":   int(result.emoticon_features.get("has_high_intensity", False)),
        "viet_slang_count":          result.emoticon_features.get("viet_slang_count", 0),

        # Emoji Unicode features (trích từ text gốc)
        "emoji_count":               result.emoji_features.get("emoji_count", 0),
        "unique_emoji_count":        result.emoji_features.get("unique_emoji_count", 0),
        "sarcasm_emoji_count":       result.emoji_features.get("sarcasm_emoji_count", 0),
        "has_sarcasm_emoji":         int(result.emoji_features.get("has_sarcasm_emoji", False)),

        # Sarcasm scores
        "sarcasm_score":             result.sarcasm_score,
        "conflict_boost":            result.conflict_boost,
        "has_conflict":              int(result.conflict_boost > 0),
        "sarcasm_score_final":       result.sarcasm_score_with_conflict,

        # Polarity
        "sentiment_polarity":        result.sentiment_polarity,
    }


# ─────────────────────────────────────────────
#  7. DEMO / TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # (text, text_sentiment giả lập từ PhoBERT)
    test_cases = [
        ("Hay thật :)))) hôm nay đi làm vui ghê",   None),    # không có emoji → no conflict
        ("Ừ đúng rồi =(((( mệt quá huhu",            None),
        ("Tuyệt vời 🙄 đúng là thiên tài",           +0.85),   # text tích cực + 🙄 tiêu cực → CONFLICT
        ("Wow haha vui quá đi ;)",                    +0.70),   # không có emoji trong POLARITY_MAP
        ("Oke đẹp lắm 😒👏 cảm ơn nhiều nhé",        +0.75),   # text tích cực + 😒 tiêu cực → CONFLICT
        ("Thật sự T_T hix không biết nói gì nữa",    -0.80),
        ("Bình thường thôi -_- chả có gì hay",        -0.20),
        ("Buồn quá 😄😊 thật sự không chịu được",    -0.80),   # text tiêu cực + emoji tích cực → CONFLICT
    ]

    print("=" * 70)
    print("        EMOJI & EMOTICON PROCESSOR — DEMO")
    print("  (Emoji Unicode giữ nguyên | Conflict detection bật khi có text_sentiment)")
    print("=" * 70)

    for text, text_sentiment in test_cases:
        result = process(text, text_sentiment=text_sentiment)
        features = get_feature_vector(result)

        conflict_tag = f" ⚡ CONFLICT +{result.conflict_boost:.2f}" if result.conflict_boost > 0 else ""
        sentiment_tag = f"{text_sentiment:+.2f}" if text_sentiment is not None else "N/A "

        print(f"\n📥 GỐC        : {result.original}")
        print(f"📤 XỬ LÝ      : {result.processed}")
        print(f"🧠 Text sent. : {sentiment_tag}  |  "
              f"Base sarcasm: {result.sarcasm_score:.2f}  →  "
              f"Final: {result.sarcasm_score_with_conflict:.2f}{conflict_tag}")
        print(f"📊 Features   : emoticon={features['emoticon_count']}, "
              f"emoji={features['emoji_count']}, "
              f"sarcasm_emoji={features['sarcasm_emoji_count']}, "
              f"has_conflict={features['has_conflict']}, "
              f"slang={features['viet_slang_count']}")
        print("-" * 70)