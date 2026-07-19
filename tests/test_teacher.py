from science_course.schemas import CausalLink, ScientificTaskDraft
from science_course.teacher import validate_draft


def _draft(task: str) -> ScientificTaskDraft:
    return ScientificTaskDraft(
        usable=True,
        rejection_reason="",
        task=task,
        reasoning=(
            "The perturbation weakens reversible interactions, which permits chain motion; "
            "removing it restores those interactions and the material's cohesion."
        ),
        evidence="removing the perturbation restores the interactions",
        answer=(
            "Temporary bond disruption permits rearrangement, while reformation restores "
            "cohesion."
        ),
        mechanism_steps=[
            "the perturbation disrupts reversible interactions",
            "chains rearrange",
            "interactions reform",
        ],
        causal_links=[
            CausalLink(
                cause="reversible interaction disruption",
                effect="chain rearrangement",
            )
        ],
        required_concepts=["reversible interactions", "chain rearrangement"],
    )


def test_accepts_self_contained_mechanism_task():
    task = (
        "A network is held together by reversible interactions. A perturbation weakens "
        "those interactions and lets chains rearrange; removing the perturbation restores "
        "the interactions. Explain why this sequence permits reshaping and recovery. "
        "The key observation is that removing the perturbation restores the interactions."
    )
    assert validate_draft(_draft(task)) == []


def test_rejects_numerical_target_task():
    task = (
        "A network is held together by reversible interactions. A perturbation weakens "
        "those interactions, and removing it restores the interactions. Calculate how many "
        "interactions remain and explain the result. The key observation is that removing "
        "the perturbation restores the interactions."
    )
    errors = validate_draft(_draft(task))
    assert any("numerical result" in error for error in errors)


def test_unusable_source_can_leave_task_fields_null():
    draft = ScientificTaskDraft(
        usable=False,
        rejection_reason="The source reports no causal or mechanistic finding.",
        task=None,
        reasoning=None,
        evidence=None,
        answer=None,
        mechanism_steps=None,
        causal_links=None,
        required_concepts=None,
    )
    assert validate_draft(draft) == [
        "The source reports no causal or mechanistic finding."
    ]
