"""Prompt-source utilities for E1 text prototype enhancement."""

from .templates import text_prompts, manual_3d_prompts
from llm.e1_dynamic_prompt_generator import generate_llm_prompts


def get_prompt_template(cfg):
    """Return prompt template according to cfg.prompt_source.

    Supported in the current step:
    - manual_full: original Point-Cache manual prompt ensemble.
    - manual_3d: point-cloud-aware subset filtered from manual_full.

    LLM-related prompt sources will be implemented in the next steps.
    """
    prompt_source = getattr(cfg, "prompt_source", "manual_full")

    if prompt_source == "manual_full":
        return text_prompts

    if prompt_source == "manual_3d":
        return manual_3d_prompts

    if prompt_source in {
        "llm_static",
        "llm_dynamic_init",
        "manual3d_llm_dynamic_init",
    }:
        raise NotImplementedError(
            f"Prompt source '{prompt_source}' is planned for E1 but not implemented yet."
        )

    raise ValueError(f"Unknown prompt source: {prompt_source}")
