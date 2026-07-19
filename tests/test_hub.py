import pytest

from science_course.hub import require_hf_namespace


class FakeApi:
    def whoami(self):
        return {
            "name": "student",
            "orgs": [{"name": "lamm-mit"}],
        }


def test_hf_namespace_accepts_user_organization():
    identity = require_hf_namespace(
        "lamm-mit/scientific-sft-grpo-sft",
        api=FakeApi(),
    )
    assert identity["name"] == "student"


def test_hf_namespace_rejects_unavailable_organization():
    with pytest.raises(RuntimeError, match="cannot publish"):
        require_hf_namespace("other-org/model", api=FakeApi())


def test_hf_repo_id_requires_namespace():
    with pytest.raises(ValueError, match="namespace/name"):
        require_hf_namespace("model-only", api=FakeApi())
