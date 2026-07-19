from types import SimpleNamespace

from science_course.judge import MechanismJudge
from science_course.schemas import MechanismJudgment, MechanismJudgmentBatch


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **_):
        self.calls += 1
        return SimpleNamespace(
            output_parsed=MechanismJudgmentBatch(
                judgments=[
                    MechanismJudgment(
                        item_index=0,
                        causal_correctness=0.9,
                        completeness=0.8,
                        evidence_use=1.0,
                        overall_score=0.88,
                        concise_feedback="Sound causal chain.",
                    ),
                    MechanismJudgment(
                        item_index=1,
                        causal_correctness=0.4,
                        completeness=0.5,
                        evidence_use=0.0,
                        overall_score=0.35,
                        concise_feedback="Reverses a causal link.",
                    ),
                ]
            ),
            id="response-test",
            model="gpt-5.6-terra",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def test_judge_batches_and_reuses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    judge = MechanismJudge(
        model="gpt-5.6-terra",
        cache_path=tmp_path / "judge.jsonl",
        client=client,
    )
    items = [
        {
            "task": "Explain how reversible interactions enable recovery.",
            "reference_reasoning": "Interactions break, chains move, interactions reform.",
            "reference_answer": "Reformation restores cohesion.",
            "mechanism_steps": ["interactions break", "chains move", "interactions reform"],
            "causal_links": [{"cause": "bond disruption", "effect": "chain motion"}],
            "student_response": "A sound response.",
        },
        {
            "task": "Explain why a catalyst accelerates a reaction.",
            "reference_reasoning": "It provides a lower-barrier pathway.",
            "reference_answer": "A lower barrier increases successful reaction events.",
            "mechanism_steps": ["alternate pathway", "lower barrier"],
            "causal_links": [{"cause": "lower barrier", "effect": "faster reaction"}],
            "student_response": "An incomplete response.",
        },
    ]

    assert judge.score(items) == [0.88, 0.35]
    assert judge.score(items) == [0.88, 0.35]
    assert responses.calls == 1
