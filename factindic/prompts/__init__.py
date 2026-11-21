"""Prompt definitions and builders for FactIndic summarization experiments."""

from factindic.prompts.templates import (
    PROMPT_TEMPLATE_VERSION,
    PromptStrategy,
    build_summarization_prompt,
    PROMPT_A_INSTRUCTION,
    PROMPT_B_INSTRUCTION,
)

__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "PromptStrategy",
    "build_summarization_prompt",
    "PROMPT_A_INSTRUCTION",
    "PROMPT_B_INSTRUCTION",
]
