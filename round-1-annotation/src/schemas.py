from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class InputRecord(BaseModel):
    id: int
    text: str
    image_path: str
    image_paths: Optional[list[str]] = None

    @field_validator("image_paths", mode="before")
    @classmethod
    def coerce_image_paths(cls, v: Optional[list]) -> Optional[list]:
        if v is not None and len(v) == 0:
            return None
        return v


class LLMJudgeRecord(BaseModel):
    id: int
    llm_pred_label: Literal["sarcastic", "non_sarcastic", "uncertain"]
    llm_prob_sarcastic: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    llm_rationale_short: str


class Round1OutputRecord(BaseModel):
    id: int
    text: str
    image_path: str
    llm_pred_label: str
    llm_prob_sarcastic: float
    llm_confidence: float
    round1_label: Literal["sarcastic", "non_sarcastic", "needs_human_review"]
    route_reason: Literal[
        "high_conf",
        "low_conf",
        "uncertain",
        "invalid_json",
        "missing_image",
        "audit_sampled",
    ]
    llm_rationale_short: str
    timestamp_utc: str
