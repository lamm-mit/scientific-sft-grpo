from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from .devices import RuntimeDevice


def load_tokenizer(model_id: str, *, token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(
    model_id: str,
    runtime: RuntimeDevice,
    *,
    token: str | None = None,
):
    """Load Gemma 4's text decoder only, then place it on the selected device."""

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=runtime.dtype,
        low_cpu_mem_usage=True,
        token=token,
    )
    model.config.use_cache = False
    return model.to(runtime.device)


def teaching_lora_config(
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: str | list[str] = "all-linear",
) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_sft_completion(tokenizer: Any, completion: list[dict[str, str]]) -> str:
    content = completion[-1]["content"]
    return content + (tokenizer.eos_token or "")


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    runtime: RuntimeDevice,
    *,
    max_new_tokens: int = 220,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(runtime.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    outputs = model.generate(**inputs, **generation_kwargs)
    continuation = outputs[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True)
