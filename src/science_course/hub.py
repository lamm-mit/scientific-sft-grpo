from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi


def require_hf_namespace(
    repo_id: str,
    *,
    token: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Fail early unless the authenticated user can publish to a Hub namespace."""

    if "/" not in repo_id:
        raise ValueError("Hugging Face repo IDs must use the form 'namespace/name'.")
    namespace = repo_id.split("/", 1)[0]
    client = api or HfApi(token=token)
    try:
        identity = client.whoami()
    except Exception as error:
        raise RuntimeError(
            "Hugging Face authentication is required. Run `hf auth login`, or "
            "uncomment the optional HF_TOKEN line in the notebook configuration."
        ) from error

    allowed = {str(identity.get("name", ""))}
    allowed.update(
        str(organization.get("name", ""))
        for organization in identity.get("orgs", [])
        if isinstance(organization, dict)
    )
    if namespace not in allowed:
        raise RuntimeError(
            f"The current Hugging Face identity cannot publish to '{namespace}'. "
            f"Available namespaces: {sorted(item for item in allowed if item)}"
        )
    return identity
