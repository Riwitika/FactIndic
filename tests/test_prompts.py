"""Unit tests for prompt construction and strategy formatting."""

import pytest
from factindic.prompts.templates import (
    PROMPT_A_INSTRUCTION,
    PROMPT_B_INSTRUCTION,
    PROMPT_TEMPLATE_VERSION,
    PromptStrategy,
    build_summarization_prompt,
)


def test_prompt_template_version_constant():
    assert PROMPT_TEMPLATE_VERSION == "v1.0"


def test_build_prompt_a_english():
    text = "The Euclid telescope has sent back its first full-colour images of the cosmos."
    prompt = build_summarization_prompt(text, strategy=PromptStrategy.PROMPT_A, language="en")

    assert PROMPT_A_INSTRUCTION in prompt
    assert PROMPT_B_INSTRUCTION not in prompt
    assert "Output the summary in English." in prompt
    assert text in prompt


def test_build_prompt_b_hindi():
    text = "इसरो ने चंद्रमा के दक्षिणी ध्रुव पर वैज्ञानिक प्रयोग जारी रखे हैं।"
    prompt = build_summarization_prompt(text, strategy=PromptStrategy.PROMPT_B, language="hi")

    assert PROMPT_B_INSTRUCTION in prompt
    assert PROMPT_A_INSTRUCTION not in prompt
    assert "Output the summary in Hindi." in prompt
    assert text in prompt


def test_prompt_strategy_invalid():
    with pytest.raises(ValueError, match="Unknown prompt strategy"):
        build_summarization_prompt("Some text", strategy="prompt_z")
