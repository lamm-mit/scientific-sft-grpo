from science_course.data import (
    assign_paper_split,
    canonical_to_grpo,
    canonical_to_sft,
    normalize_license,
)


def _record():
    return {
        "task_id": "task-1",
        "paper_id": "paper-1",
        "task": (
            "A polymer contains chains connected by reversible hydrogen bonds. Heating "
            "temporarily disrupts these bonds and increases chain mobility; cooling restores "
            "the bonds. Explain why the material can be reshaped and then recover strength."
        ),
        "reasoning": "Heating breaks reversible bonds; cooling reforms them.",
        "evidence": "cooling restores the bonds",
        "answer": (
            "Reversible bond breaking enables reshaping and bond reformation "
            "restores strength."
        ),
        "mechanism_steps": ["heating disrupts bonds", "chains move", "cooling reforms bonds"],
        "causal_links": [{"cause": "bond disruption", "effect": "chain mobility"}],
        "required_concepts": ["reversible bonds", "chain mobility"],
        "source_license": "CC-BY-4.0",
        "source_url": "https://example.test/paper",
        "source_text": "Private source material used only by the teacher.",
    }


def test_license_normalization():
    assert normalize_license("CC-BY-4.0") == "CCBY40"


def test_split_is_deterministic():
    assert assign_paper_split("abc", 17) == assign_paper_split("abc", 17)


def test_sft_projection_has_completion():
    row = canonical_to_sft(_record())
    assert row["prompt"][0]["role"] == "system"
    assert "<answer>Reversible bond breaking" in row["completion"][0]["content"]
    assert "source_text" not in row


def test_grpo_projection_hides_completion():
    row = canonical_to_grpo(_record())
    assert "completion" not in row
    assert row["reference_answer"].startswith("Reversible bond breaking")
    assert "passage" not in str(row["prompt"]).casefold()
    assert "source_text" not in row
