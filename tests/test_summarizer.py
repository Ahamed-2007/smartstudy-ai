import re

from summarizer import generate_fallback

SOURCE = """Artificial intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and problem solving. AI can be classified into narrow AI, general AI, and superintelligent AI. Applications include healthcare, finance, education, robotics, and customer support. Advantages include automation, speed, and personalization. Limitations include bias, data dependence, and job displacement. In conclusion, AI is transforming modern society and requires careful governance. This paragraph provides a practical example of the core ideas behind AI, its different types, and its role in modern life."""


def test_2_mark_answer_is_short_and_structured():
    result = generate_fallback(SOURCE, "2-Mark Answer")
    assert "## Introduction" in result["answer"]
    assert "## Short Summary" in result["answer"]
    assert result["summary_words"] < 120


def test_10_mark_answer_keeps_detailed_sections():
    result = generate_fallback(SOURCE, "10-Mark Exam Answer")
    text = result["answer"]
    required = [
        "## Introduction",
        "## Definition",
        "## Main Concepts",
        "## Examples",
        "## Applications",
        "## Advantages",
        "## Limitations",
        "## Conclusion",
        "## Short Summary",
        "## What Was Preserved",
        "## What Was Shortened or Removed",
        "## Quick Revision",
    ]
    for heading in required:
        assert heading in text, f"missing heading: {heading}"


def test_word_count_and_compression_are_reasonable():
    result = generate_fallback(SOURCE, "5-Mark Exam Answer")
    original_words = len(re.findall(r"\b\w+\b", SOURCE))
    generated_words = result["summary_words"]
    assert generated_words > 0
    assert original_words > 0
    assert abs(result["compression"] - ((original_words - generated_words) / original_words) * 100) < 0.01
    assert "\\n" not in result["answer"]
