from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class InputRecord(BaseModel):
    id: int
    text: str
    image_path: str
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
    Structured output of the LLM judge using the current prompt schema.

    label_llm1        : 0 (non-sarcastic), 1 (sarcastic), or "INVALID"
    has_emoji         : 1 nếu bài đăng có emoji, 0 nếu không
    needs_human_check : 0 nếu LLM tự tin, 1 nếu cần human kiểm chứng (dùng để routing)
    notes             : free-form notes from the model
    reasoning         : full nested reasoning dict from the model response
    parse_error       : True when the model output could not be parsed (routing hint)
    image_missing     : True when expected images were not found on disk (routing hint)
    """

    id: int
    label_llm1: Union[int, Literal["INVALID"]]
    has_emoji: Optional[int] = None
    needs_human_check: Optional[int] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    parse_error: bool = False
    image_missing: bool = False


class Round1OutputRecord(BaseModel):
    id: int
    text: str
    image_path: str
    label_llm1: Union[int, Literal["INVALID"]]
    has_emoji: Optional[int]
    needs_human_check: Optional[int]
    notes: str
    reasoning: Dict[str, Any]
    round1_label: Literal["sarcastic", "non_sarcastic", "invalid"]
    need_review: bool
    route_reason: Literal[
        "high_conf",
        "low_conf",
        "uncertain",
        "invalid_json",
        "missing_image",
        "audit_sampled",
    ]
    timestamp_utc: str
