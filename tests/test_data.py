from science_course.data import (
    TASK_FAMILIES,
    allocate_task_quotas,
    canonical_to_grpo,
    canonical_to_sft,
    normalize_license,
    render_completion,
    task_quota_status,
    trim_records_to_quotas,
)


def _record():
    return {
        "task_id": "task-1",
        "paper_id": "paper-1",
        "split": "sft_train",
        "task_family": "mechanism_guided_design",
        "task": (
            "Design a hydrogel that recovers after deformation in water without external "
            "heating. Develop several mechanisms, identify design principles, synthesize "
            "the strongest approach, and recommend a final design."
        ),
        "brainstorm": [
            "Use reversible host-guest crosslinks.",
            "Use protected hydrogen-bonding domains.",
            "Use dynamic ionic clusters in a flexible network.",
        ],
        "principles": [
            "Crosslinks must break reversibly under strain.",
            "Chains need enough mobility to rearrange.",
            "Binding partners must reassociate in water.",
        ],
        "synthesis": (
            "A permanent flexible network can preserve shape while reversible host-guest "
            "crosslinks dissipate energy and reform after deformation."
        ),
        "answer": (
            "Use a flexible permanent network reinforced with reversible host-guest "
            "crosslinks that dissociate under strain and spontaneously reassociate in water."
        ),
        "required_constraints": [
            "operates in water",
            "requires no external heating",
        ],
        "evaluation_criteria": [
            "reversible energy dissipation",
            "spontaneous recovery",
        ],
        "acceptable_alternatives": [
            "dynamic ionic domains",
            "water-compatible hydrogen-bonding motifs",
        ],
        "failure_modes": [
            "irreversible bond scission",
            "crosslinks that cannot reform in water",
        ],
        "source_license": "CC-BY-4.0",
        "source_url": "https://example.test/paper",
        "source_text": "Private source material used only by the teacher.",
    }


def test_license_normalization():
    assert normalize_license("CC-BY-4.0") == "CCBY40"


def test_allocate_task_quotas_is_exact_and_deterministic():
    split_targets = {"sft_train": 11, "sft_validation": 3}
    weights = {
        "mechanism_guided_design": 0.25,
        "experimental_design": 0.25,
        "troubleshooting": 0.20,
        "hypothesis_development": 0.15,
        "cross_domain_synthesis": 0.15,
    }
    first = allocate_task_quotas(split_targets, weights)
    second = allocate_task_quotas(split_targets, weights)

    assert first == second
    assert set(family for _, family in first) == set(TASK_FAMILIES)
    assert sum(value for (split, _), value in first.items() if split == "sft_train") == 11
    assert (
        sum(value for (split, _), value in first.items() if split == "sft_validation")
        == 3
    )


def test_trim_and_status_use_post_validation_quotas():
    targets = {"sft_train": 5}
    weights = {family: 1.0 for family in TASK_FAMILIES}
    records = [
        {
            "task_id": f"{family}-{index}",
            "paper_id": f"{family}-{index}",
            "split": "sft_train",
            "task_family": family,
        }
        for family in TASK_FAMILIES
        for index in range(2)
    ]
    selected = trim_records_to_quotas(records, targets, weights)
    status = task_quota_status(selected, targets, weights)

    assert len(selected) == 5
    assert status["complete"] is True
    assert not status["deficits"]


def test_render_and_sft_projection_have_four_sections():
    record = _record()
    rendered = render_completion(record)
    row = canonical_to_sft(record)

    assert rendered.startswith("<brainstorm>")
    assert "<principles>" in rendered
    assert "<synthesis>" in rendered
    assert rendered.rstrip().endswith("</answer>")
    assert row["prompt"][0]["role"] == "system"
    assert row["completion"][0]["content"] == rendered
    assert "source_text" not in row
    assert "required_constraints" not in row


def test_grpo_projection_hides_reference_completion():
    record = _record()
    record["split"] = "grpo_train"
    row = canonical_to_grpo(record)

    assert "completion" not in row
    assert "brainstorm" not in row
    assert "principles" not in row
    assert "synthesis" not in row
    assert "answer" not in row
    assert "source_text" not in row
    assert row["required_constraints"] == record["required_constraints"]
    assert "passage" not in str(row["prompt"]).casefold()
