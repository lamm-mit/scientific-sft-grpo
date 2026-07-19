from types import SimpleNamespace

import pytest

from science_course.judge import (
    ScientificDesignJudge,
    configure_scientific_design_judge,
    scientific_design_reward,
)
from science_course.schemas import (
    ScientificDesignJudgment,
    ScientificDesignJudgmentBatch,
)


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **_):
        self.calls += 1
        return SimpleNamespace(
            output_parsed=ScientificDesignJudgmentBatch(
                judgments=[
                    ScientificDesignJudgment(
                        item_index=0,
                        brainstorm=3,
                        principles=4,
                        synthesis=3,
                        answer=4,
                        brief_reason="Several plausible ideas and a coherent final design.",
                    ),
                    ScientificDesignJudgment(
                        item_index=1,
                        brainstorm=1,
                        principles=2,
                        synthesis=1,
                        answer=2,
                        brief_reason="Ideas are repetitive and the synthesis is incomplete.",
                    ),
                ]
            ),
            id="response-test",
            model="gpt-5.6-luna",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def _item(response: str):
    return {
        "task": "Design a reversible scientific system under explicit constraints.",
        "required_constraints": ["must work in water", "must recover autonomously"],
        "evaluation_criteria": ["plausibility", "constraint satisfaction"],
        "acceptable_alternatives": ["host-guest binding", "dynamic ionic domains"],
        "failure_modes": ["irreversible scission"],
        "student_response": response,
    }


def test_judge_batches_normalizes_and_reuses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    judge = ScientificDesignJudge(
        model="gpt-5.6-luna",
        cache_path=tmp_path / "judge.jsonl",
        client=client,
    )
    items = [_item("A strong response."), _item("A weak response.")]

    assert judge.score(items) == [0.875, 0.375]
    assert judge.score(items) == [0.875, 0.375]
    assert responses.calls == 1


class OneJudgmentResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **_):
        self.calls += 1
        return SimpleNamespace(
            output_parsed=ScientificDesignJudgmentBatch(
                judgments=[
                    ScientificDesignJudgment(
                        item_index=0,
                        brainstorm=3,
                        principles=4,
                        synthesis=3,
                        answer=4,
                        brief_reason="Strong structured solution.",
                    )
                ]
            ),
            id="response-one",
            model="gpt-5.6-luna",
            usage=SimpleNamespace(input_tokens=8, output_tokens=4, total_tokens=12),
        )


def test_combined_reward_skips_luna_for_invalid_format(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses = OneJudgmentResponses()
    configure_scientific_design_judge(
        model="gpt-5.6-luna",
        cache_path=tmp_path / "combined.jsonl",
        client=SimpleNamespace(responses=responses),
        format_base_reward=0.10,
        semantic_reward_weight=0.90,
    )
    valid = """<brainstorm>Three distinct ideas.</brainstorm>
<principles>Scientific constraints.</principles>
<synthesis>Compare and combine the ideas.</synthesis>
<answer>Give the final design.</answer>"""
    invalid = "<answer>Missing the other sections.</answer>"
    shared = {
        "task": ["Task one", "Task two"],
        "required_constraints": [["constraint"], ["constraint"]],
        "evaluation_criteria": [["criterion"], ["criterion"]],
        "acceptable_alternatives": [["alternative"], ["alternative"]],
        "failure_modes": [["failure"], ["failure"]],
    }

    scores = scientific_design_reward([valid, invalid], **shared)

    assert scores == pytest.approx([0.8875, 0.0])
    assert responses.calls == 1
