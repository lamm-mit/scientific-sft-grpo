from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cli_config import (
    GRPOJobConfig,
    SFTJobConfig,
    config_as_dict,
    load_grpo_config,
    load_sft_config,
    with_cli_overrides,
)


def _training_parser(subparsers: Any, stage: str, help_text: str) -> None:
    parser = subparsers.add_parser(stage, help=help_text)
    parser.add_argument(
        "--config",
        default=f"configs/{stage}.toml",
        help=f"TOML configuration (default: configs/{stage}.toml)",
    )
    parser.add_argument(
        "--resume",
        help="'none', 'auto', or a local checkpoint directory; overrides TOML",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny dataset, run one step, write to *-smoke, and never push",
    )
    push = parser.add_mutually_exclusive_group()
    push.add_argument(
        "--push",
        dest="push",
        action="store_true",
        help="Publish checkpoints and the final adapter to the configured Hub repo",
    )
    push.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Disable all Hub writes",
    )
    parser.set_defaults(push=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scientific-sft-grpo",
        description=(
            "Train the scientific task-only Gemma 4 adapters from the same "
            "settings used by the teaching notebooks."
        ),
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser(
        "doctor",
        help="Run read-only environment, data, credential, and resume checks",
    )
    doctor.add_argument("--stage", choices=("sft", "grpo"), required=True)
    doctor.add_argument(
        "--config",
        help="TOML configuration; defaults to configs/<stage>.toml",
    )
    _training_parser(subparsers, "sft", "Run completion-only LoRA SFT")
    _training_parser(subparsers, "grpo", "Continue the SFT adapter with GRPO")
    show = subparsers.add_parser(
        "show-config",
        help="Parse, validate, and print a resolved configuration",
    )
    show.add_argument("--stage", choices=("sft", "grpo"), required=True)
    show.add_argument("--config", help="Defaults to configs/<stage>.toml")
    return parser


def _load(stage: str, path: str | None) -> SFTJobConfig | GRPOJobConfig:
    config_path = Path(path or f"configs/{stage}.toml")
    return (
        load_sft_config(config_path)
        if stage == "sft"
        else load_grpo_config(config_path)
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"doctor", "show-config"}:
            config = _load(args.stage, args.config)
            if args.command == "show-config":
                _print_json(config_as_dict(config))
                return 0
            from .doctor import run_doctor

            result = run_doctor(config, stage=args.stage)
            _print_json(result)
            return 0 if result["passed"] else 1

        config = _load(args.command, args.config)
        config = with_cli_overrides(
            config,
            resume=args.resume,
            push=args.push,
            smoke_test=args.smoke_test,
        )
        if args.command == "sft":
            from .train_sft import run_sft

            result = run_sft(config, smoke_test=args.smoke_test)
        else:
            if args.smoke_test:
                print(
                    "warning: the GRPO smoke test still makes real judge API calls",
                    file=sys.stderr,
                )
            from .train_grpo import run_grpo

            result = run_grpo(config, smoke_test=args.smoke_test)
        _print_json(result)
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
