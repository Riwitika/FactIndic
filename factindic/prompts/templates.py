"""Prompt templates and builders for FactIndic summarization experiments."""

from enum import Enum
from typing import Dict, Optional

PROMPT_TEMPLATE_VERSION = "v1.0"

PROMPT_A_INSTRUCTION = (
    "Generate a concise summary of the supplied article while preserving its main information."
)

PROMPT_B_INSTRUCTION = (
    "Generate a concise summary using only information supported by the supplied article. "
    "Preserve important entities, numbers, dates, events, and relationships. "
    "Do not introduce unsupported details or external knowledge. "
    "If the source is ambiguous, do not invent a resolution."
)


class PromptStrategy(str, Enum):
    """Enumeration of evaluation prompting strategies."""

    PROMPT_A = "prompt_a"  # Standard Summarization
    PROMPT_B = "prompt_b"  # Factuality-Aware Summarization


def build_summarization_prompt(
    source_text: str,
    strategy: PromptStrategy,
    language: str = "en",
) -> str:
    """Construct an instruction prompt for summary generation.

    Args:
        source_text: Cleaned body text of the article to summarize.
        strategy: PromptStrategy.PROMPT_A or PromptStrategy.PROMPT_B.
        language: Canonical language code ('en' or 'hi').

    Returns:
        Formatted prompt string.
    """
    if isinstance(strategy, str):
        try:
            strategy = PromptStrategy(strategy)
        except ValueError:
            valid_vals = [s.value for s in PromptStrategy]
            raise ValueError(f"Unknown prompt strategy '{strategy}'. Valid options: {valid_vals}")

    instruction = PROMPT_A_INSTRUCTION if strategy == PromptStrategy.PROMPT_A else PROMPT_B_INSTRUCTION

    lang_lower = language.lower().strip()
    if lang_lower in ("hi", "hindi"):
        lang_directive = "Output the summary in Hindi."
    else:
        lang_directive = "Output the summary in English."

    prompt = (
        f"You are a professional abstractive summarization assistant.\n\n"
        f"Task:\n{instruction}\n{lang_directive}\n\n"
        f"Article:\n{source_text.strip()}\n\n"
        f"Summary:"
    )

    return prompt
