from src.llm_judge import _validate


def test_validate_accepts_final_label_and_nested_labels():
    rec = _validate(
        {
            "labels": {"T": 1, "I": 0, "MM": 0},
            "final_label": 0,
            "reasoning": {
                "text_only": "x",
                "image_only": "y",
                "multimodal": "z",
                "verdict": "v",
            },
            "has_emoji": 1,
        }
    )

    assert rec.label_llm2 == 0
    assert rec.final_label == 0
    assert rec.T == 1
    assert rec.I == 0
    assert rec.MM == 0


def test_validate_derives_final_label_when_missing():
    rec = _validate(
        {
            "labels": {"T": 0, "I": 1, "MM": 0},
            "reasoning": {"verdict": "v"},
            "has_emoji": 0,
        }
    )

    assert rec.label_llm2 == 0
    assert rec.final_label == 0
