from science_course.rewards import (
    answer_similarity_reward,
    component_scores,
    concept_coverage_reward,
    evidence_grounding_reward,
    format_reward,
    parse_completion,
    token_f1,
)

GOOD = """<reasoning>Heating disrupts reversible bonds, allowing chains to move;
cooling reforms the bonds.</reasoning>
<evidence>cooling restores the bonds</evidence>
<answer>Reversible bond breaking enables reshaping, while bond reformation
restores strength.</answer>"""

TASK = (
    "A polymer contains chains connected by reversible hydrogen bonds. Heating temporarily "
    "disrupts these bonds and increases chain mobility; cooling restores the bonds. Explain "
    "why the material can be reshaped and then recover strength."
)


def test_parse_completion():
    assert parse_completion(GOOD)["answer"].startswith("Reversible bond")
    assert parse_completion("<answer>A</answer>") is None


def test_reward_components():
    assert format_reward([GOOD]) == [1.0]
    assert evidence_grounding_reward([GOOD], [TASK]) == [1.0]
    assert concept_coverage_reward(
        [GOOD], [["reversible bonds", "chains"]]
    ) == [1.0]
    assert answer_similarity_reward(
        [GOOD],
        [
            "Reversible bond breaking enables reshaping, while bond reformation "
            "restores strength."
        ],
    ) == [1.0]


def test_fabricated_evidence_gets_no_grounding_reward():
    bad = GOOD.replace("cooling restores the bonds", "cooling creates covalent bonds", 1)
    assert evidence_grounding_reward([bad], [TASK]) == [0.0]


def test_component_scores_and_f1_bounds():
    scores = component_scores(
        GOOD,
        task=TASK,
        reference_answer=(
            "Reversible bond breaking enables reshaping and reformation restores strength."
        ),
        required_concepts=["reversible bonds", "chain mobility"],
    )
    assert set(scores) == {"format", "evidence", "concepts", "similarity"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert 0.0 < token_f1("bonds reform", "reversible bonds reform on cooling") < 1.0
