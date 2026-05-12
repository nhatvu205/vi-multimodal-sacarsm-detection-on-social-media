from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class InputRecord(BaseModel):
    id: int
    text: str
    image_path: str = ""
    image_paths: Optional[List[str]] = None
    ocr_text: Optional[str] = None

    @field_validator("image_paths", mode="before")
    @classmethod
    def coerce_image_paths(cls, v: Optional[list]) -> Optional[list]:
        if v is not None and len(v) == 0:
            return None
        return v


class LLMJudgeRecord(BaseModel):
    id: int = -1
    label_llm2: Union[int, Literal["INVALID"]] = "INVALID"
    
    # Fine-grained fields (Round 2)
    T: Optional[int] = None
    I: Optional[int] = None
    MM: Optional[int] = None
    KI: Optional[Literal["YES", "NO", "NULL"]] = None

    # ✅ THÊM MỚI — intermediate fields từ Turn 1
    T_confidence: Optional[str] = None
    T_signals: Optional[List[Any]] = None
    T_reason: Optional[str] = None
    T_overridden: bool = False

    # ✅ THÊM MỚI — intermediate fields từ Turn 2
    I_confidence: Optional[str] = None
    I_category: Optional[str] = None
    I_description: Optional[str] = None
    I_reason: Optional[str] = None
    I_overridden: bool = False

    # ✅ THÊM MỚI — từ Turn 3
    MM_pattern: Optional[str] = None

    has_emoji: Optional[int] = None
    needs_human_check: Optional[int] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    parse_error: bool = False
    image_missing: bool = False


class Round1OutputRecord(BaseModel):
    """
    Output record cho Round 2 (vẫn giữ tên class để ít thay đổi pipeline nhất).
    Đã bổ sung các field fine-grained.
    """
    id: int
    text: str
    image_path: str

    label_llm2: Union[int, Literal["INVALID"]]

    # Fine-grained fields
    T: Optional[int] = None
    I: Optional[int] = None
    MM: Optional[int] = None
    KI: Optional[Literal["YES", "NO", "NULL"]] = None

    has_emoji: Optional[int] = None
    needs_human_check: Optional[int] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)

    # Routing fields
    round1_label: Literal["sarcastic", "non_sarcastic", "invalid"] = "invalid"
    need_review: bool = True
    route_reason: Literal[
        "high_conf",
        "low_conf",
        "uncertain",
        "invalid_json",
        "missing_image",
        "audit_sampled",
    ] = "uncertain"
    
    timestamp_utc: str = ""


# Optional: Alias cho rõ ràng hơn ở Round 2
Round2OutputRecord = Round1OutputRecord
