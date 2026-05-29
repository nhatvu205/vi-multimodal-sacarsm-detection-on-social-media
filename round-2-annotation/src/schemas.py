from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InputRecord(BaseModel):
    id: int
    text: str
    image_path: str = ""
    image_paths: Optional[List[str]] = None
    ocr_text: Optional[str] = None
    label_round_1: Optional[int] = None

    @field_validator("image_paths", mode="before")
    @classmethod
    def coerce_image_paths(cls, v: Optional[list]) -> Optional[list]:
        if v is not None and len(v) == 0:
            return None
        return v

    @field_validator("label_round_1", mode="before")
    @classmethod
    def coerce_label_round_1(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned in {"", "INVALID", "-1", "NULL", "NONE"}:
                return None
            if cleaned in {"0", "1"}:
                return int(cleaned)
            return None
        if v in (0, 1):
            return int(v)
        return None


class LLMJudgeRecord(BaseModel):
    """
    Structured output of the LLM judge for round 2.

    label_llm2        : 0 (non-sarcastic), 1 (sarcastic), -1 (failed after retries), or "INVALID"
    T / I / MM        : fine-grained modality labels
    KI                : whether the image is necessary for the sarcastic reading
    has_emoji         : 1 nếu bài đăng có emoji, 0 nếu không
    needs_human_check : 0 nếu LLM tự tin, 1 nếu cần human kiểm chứng
    notes             : free-form notes from the model
    reasoning         : full nested reasoning dict from the model response
    parse_error       : True when the model output could not be parsed / request failed
    image_missing     : True when expected images were not found on disk
    raw_response      : raw response/debug payload for parse-error cases (internal only)
    """

    id: int
    label_llm2: Union[int, Literal["INVALID"]]
    final_label: Optional[Union[int, Literal["INVALID"]]] = None
    T: Optional[int] = None
    I: Optional[int] = None
    MM: Optional[int] = None
    KI: Optional[Literal["YES", "NO", "NULL"]] = None
    has_emoji: Optional[int] = None
    needs_human_check: Optional[int] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    parse_error: bool = False
    image_missing: bool = False
    raw_response: Optional[Any] = Field(default=None, exclude=True)


class Round2OutputRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: int
    text: str
    image_path: str
    ocr_text: Optional[str] = None
    label_round_1: Optional[int] = None
    label_llm2: Union[int, Literal["INVALID"]]
    final_label: Optional[Union[int, Literal["INVALID"]]] = None
    T: Optional[int] = None
    I: Optional[int] = None
    MM: Optional[int] = None
    KI: Optional[Literal["YES", "NO", "NULL"]] = None
    has_emoji: Optional[int]
    needs_human_check: Optional[int] = Field(default=None, exclude=True)
    notes: str
    reasoning: Dict[str, Any]
    round2_label: Literal["sarcastic", "non_sarcastic", "invalid"]
    need_review: bool = Field(exclude=True)
    route_reason: Literal[
        "high_conf",
        "low_conf",
        "uncertain",
        "invalid_json",
        "missing_image",
        "audit_sampled",
    ] = Field(exclude=True)
    parse_error: bool = False
    image_missing: bool = False
    timestamp_utc: str
