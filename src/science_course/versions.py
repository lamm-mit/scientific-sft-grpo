from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

PINNED_STACK = {
    "transformers": "5.14.1",
    "trl": "1.8.0",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "openai": "2.46.0",
}


def installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in PINNED_STACK:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def require_training_stack() -> dict[str, str]:
    """Fail early when Transformers or TRL predates the notebook API."""

    observed = installed_versions()
    errors = []
    for package in ("transformers", "trl", "peft", "accelerate", "datasets"):
        current = observed[package]
        required = PINNED_STACK[package]
        if current == "not installed" or Version(current) < Version(required):
            errors.append(f"{package}>={required} required; found {current}")
    if errors:
        raise RuntimeError(
            "The training environment is older than the tested notebook stack:\n- "
            + "\n- ".join(errors)
            + '\nRun: python -m pip install -e "."'
        )
    return observed
