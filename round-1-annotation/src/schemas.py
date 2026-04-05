from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class InputRecord(BaseModel):
    id: int
    text: str
    image_path: str
    image_paths: Optional[List[str]] = None

    @field_validator("image_paths", mode="before")
    @classmethod
    def coerce_image_paths(cls, v: Optional[list]) -> Optional[list]:
        if v is not None and len(v) == 0:
            return None
        return v


class LLMJudgeRecord(BaseModel):
    """
    Structured output of the LLM judge using the new prompt schema.

    label_llm1    : 0 (non-sarcastic), 1 (sarcastic), or "INVALID"
    text_only     : label when only text+emoji is considered (0/1/null)
    imageset_only : label when only images are considered (0/1/null)
    key_images    : 1-based indices of images that caused conflict (empty when label != 1)
    difficulty    : "Easy" | "Hard" (null when INVALID)
    notes         : free-form notes from the model
    reasoning     : full nested reasoning dict from the model response
    parse_error   : True when the model output could not be parsed (routing hint)
    image_missing : True when expected images were not found on disk (routing hint)
    """

    id: int
    label_llm1: Union[int, Literal["INVALID"]]
    text_only: Optional[int] = None
    imageset_only: Optional[int] = None
    key_images: List[int] = Field(default_factory=list)
    difficulty: Optional[Literal["Easy", "Hard"]] = None
    notes: str = ""
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    parse_error: bool = False
    image_missing: bool = False


class Round1OutputRecord(BaseModel):
    id: int
    text: str
    image_path: str
    label_llm1: Union[int, Literal["INVALID"]]
    text_only: Optional[int]
    imageset_only: Optional[int]
    key_images: List[int]
    difficulty: Optional[Literal["Easy", "Hard"]]
    notes: str
    round1_label: Literal["sarcastic", "non_sarcastic", "needs_human_review"]
    route_reason: Literal[
        "high_conf",
        "low_conf",
        "uncertain",
        "invalid_json",
        "missing_image",
        "audit_sampled",
    ]
    timestamp_utc: str
