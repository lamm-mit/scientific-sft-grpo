import pytest

from science_course.rewards import (
    combined_reward,
    format_reward,
    parse_completion,
    semantic_score_from_judgment,
)

GOOD = """<brainstorm>
- Use reversible host-guest crosslinks.
- Use dynamic ionic clusters.
- Use protected hydrogen-bonding domains.
</brainstorm>
<principles>
- Crosslinks must break reversibly.
- Chains need mobility.
- Binding partners must reassociate in water.
</principles>
<synthesis>
Use a flexible permanent network with reversible host-guest crosslinks.
</synthesis>
<answer>
The reversible crosslinks dissipate energy and reform after strain.
</answer>"""


def test_parse_completion_requires_four_nonempty_ordered_sections():
    parsed = parse_completion(GOOD)
    assert parsed is not None
    assert parsed["answer"].startswith("The reversible")
    assert parse_completion("<answer>A</answer>") is None
    assert parse_completion(GOOD.replace("Chains need mobility.", "")) is not None
    assert parse_completion(GOOD.replace("Use a flexible permanent network", "")) is not None
    assert parse_completion(GOOD.replace("<synthesis>", "<synthesis></synthesis>")) is None


def test_format_reward_is_binary():
    assert format_reward([GOOD, "<answer>wrong format</answer>"]) == [1.0, 0.0]


def test_semantic_score_is_exact_average_of_four_zero_to_four_scores():
    score = semantic_score_from_judgment(
        {
            "brainstorm": 3,
            "principles": 4,
            "synthesis": 3,
            "answer": 4,
        }
    )
    assert score == 0.875


def test_combined_reward_matches_documented_formula():
    assert combined_reward(1.0, 0.875) == pytest.approx(0.8875)
    assert combined_reward(0.0, 1.0) == 0.0
    with pytest.raises(ValueError, match="sum to 1"):
        combined_reward(
            1.0,
            0.5,
            format_base_reward=0.2,
            semantic_reward_weight=0.9,
        )
