from science_course.data import TASK_FAMILIES, read_jsonl
from science_course.schemas import ScientificDesignTaskDraft
from science_course.teacher import generate_canonical_tasks, validate_draft


def _draft(task: str) -> ScientificDesignTaskDraft:
    return ScientificDesignTaskDraft(
        usable=True,
        rejection_reason="",
        task_family="mechanism_guided_design",
        task=task,
        brainstorm=[
            "Use reversible host-guest crosslinks.",
            "Use dynamic ionic domains.",
            "Use protected hydrogen-bonding motifs.",
        ],
        principles=[
            "Crosslinks must dissociate reversibly.",
            "Chains need local mobility.",
            "Interactions must reform in water.",
        ],
        synthesis=(
            "Combine a shape-preserving flexible network with reversible host-guest "
            "crosslinks that dissipate energy and reform after strain."
        ),
        answer=(
            "Use a flexible permanent network reinforced by reversible host-guest "
            "crosslinks that spontaneously reassociate in water."
        ),
        required_constraints=["works in water", "requires no heating"],
        evaluation_criteria=["reversible dissipation", "spontaneous recovery"],
        acceptable_alternatives=["dynamic ionic clusters"],
        failure_modes=["irreversible bond scission"],
    )


def test_accepts_self_contained_problem_solving_task():
    task = (
        "Design a hydrogel that recovers after repeated deformation in water without "
        "external heating. Develop several mechanistic approaches, identify the governing "
        "design principles, synthesize the strongest direction, and recommend a final design."
    )
    assert (
        validate_draft(
            _draft(task),
            requested_family="mechanism_guided_design",
        )
        == []
    )


def test_rejects_numerical_target_task():
    task = (
        "Design a reversible hydrogel and calculate how many crosslinks remain after "
        "deformation. Develop several ideas and recommend a final material architecture."
    )
    errors = validate_draft(
        _draft(task),
        requested_family="mechanism_guided_design",
    )
    assert any("numerical result" in error for error in errors)


def test_rejects_source_reference():
    task = (
        "According to the passage, design a self-healing hydrogel. Develop several "
        "mechanistic approaches, identify principles, synthesize them, and answer."
    )
    errors = validate_draft(
        _draft(task),
        requested_family="mechanism_guided_design",
    )
    assert any("unavailable source" in error for error in errors)


def test_unusable_source_can_leave_fields_null():
    draft = ScientificDesignTaskDraft(
        usable=False,
        rejection_reason="The source cannot support a self-contained design task.",
        task_family=None,
        task=None,
        brainstorm=None,
        principles=None,
        synthesis=None,
        answer=None,
        required_constraints=None,
        evaluation_criteria=None,
        acceptable_alternatives=None,
        failure_modes=None,
    )
    assert validate_draft(
        draft,
        requested_family="experimental_design",
    ) == ["The source cannot support a self-contained design task."]


def test_quota_generation_is_exact_resumable_and_post_validation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("science_course.teacher.OpenAI", lambda **_: object())

    calls = []

    def fake_generate(
        source,
        *,
        split,
        task_family,
        client,
        teacher_model,
        critic_model,
    ):
        del client, teacher_model, critic_model
        calls.append(source["paper_id"])
        return (
            {
                "task_id": f"task-{source['paper_id']}",
                **source,
                "split": split,
                "task_family": task_family,
            },
            None,
        )

    monkeypatch.setattr("science_course.teacher.generate_task", fake_generate)
    sources = [
        {
            "paper_id": f"paper-{index}",
            "source_text": "source",
            "source_sha256": f"sha-{index}",
        }
        for index in range(10)
    ]
    targets = {"sft_train": 5}
    weights = {family: 1.0 for family in TASK_FAMILIES}
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"

    first = generate_canonical_tasks(
        sources,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        split_targets=targets,
        task_family_weights=weights,
        concurrency=2,
        max_attempts=10,
    )
    call_count = len(calls)
    second = generate_canonical_tasks(
        sources,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        split_targets=targets,
        task_family_weights=weights,
        concurrency=2,
        max_attempts=10,
    )

    assert len(first) == len(second) == 5
    assert len(read_jsonl(accepted_path)) == 5
    assert len(calls) == call_count
    assert {row["task_family"] for row in first} == set(TASK_FAMILIES)


def test_quota_generation_journals_api_errors_and_continues(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("science_course.teacher.OpenAI", lambda **_: object())

    def fake_generate(
        source,
        *,
        split,
        task_family,
        client,
        teacher_model,
        critic_model,
    ):
        del client, teacher_model, critic_model
        if source["paper_id"] == "blocked":
            raise RuntimeError("source rejected by API")
        return (
            {
                "task_id": f"task-{source['paper_id']}",
                **source,
                "split": split,
                "task_family": task_family,
            },
            None,
        )

    monkeypatch.setattr("science_course.teacher.generate_task", fake_generate)
    sources = [
        {
            "paper_id": paper_id,
            "source_text": "source",
            "source_sha256": f"sha-{paper_id}",
        }
        for paper_id in ("blocked", "good-1", "good-2")
    ]
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    result = generate_canonical_tasks(
        sources,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        split_targets={"sft_train": 1},
        task_family_weights={
            family: 1.0 if family == "mechanism_guided_design" else 0.0
            for family in TASK_FAMILIES
        },
        concurrency=1,
        max_attempts=3,
    )

    rejected = read_jsonl(rejected_path)
    assert len(result) == 1
    assert rejected[0]["paper_id"] == "blocked"
    assert rejected[0]["stage"] == "api_error"
    assert rejected[0]["requested_split"] == "sft_train"
    assert rejected[0]["requested_task_family"] == "mechanism_guided_design"
    assert "RuntimeError: source rejected by API" in rejected[0]["reasons"]
