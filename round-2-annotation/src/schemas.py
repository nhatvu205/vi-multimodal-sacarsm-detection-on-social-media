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
    """
    Structured output của LLM Judge cho Round 2 (Fine-grained).
    """
    id: int = -1
    
    # Core label
    label_llm2: Union[int, Literal["INVALID"]] = "INVALID"
    
    # Fine-grained fields (Round 2)
    T: Optional[int] = None          # Text-Only
    I: Optional[int] = None          # Image-Only
    MM: Optional[int] = None         # Multimodal
    KI: Optional[Literal["YES", "NO", "NULL"]] = None

    # Old fields (giữ tương thích)
    has_emoji: Optional[int] = None
    needs_human_check: Optional[int] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)

    # Internal flags
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
