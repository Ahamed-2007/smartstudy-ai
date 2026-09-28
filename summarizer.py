import io
import os
import re
from collections import Counter

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# NLTK resources are downloaded quietly when needed.
for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "the", "is", "a", "an", "and", "or", "of", "to", "in", "for",
        "on", "with", "as", "by", "from", "that", "this", "are", "was",
        "were", "be", "it", "at", "which", "can", "also", "has", "have",
    }


def extract_text_from_upload(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError("Unsupported file type.")


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def normalize_markdown(text):
    if text is None:
        return ""
    return str(text).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r", "\n")


def analyze_text(text):
    sentences = split_sentences(text)
    words = re.findall(r"\b[a-zA-Z][a-zA-Z'-]*\b", text.lower())

    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(filtered)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(sentences if len(sentences) > 1 else [text])
        features = vectorizer.get_feature_names_out()
        scores = matrix.sum(axis=0).A1
        ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
        keywords = [x[0] for x in ranked[:12]]
    except Exception:
        keywords = [w for w, _ in counts.most_common(12)]

    return {
        "original_words": len(words),
        "sentences": len(sentences),
        "keywords": keywords,
        "top_words": counts.most_common(10),
    }


def build_prompt(source, purpose):
    purpose_rules = {
        "Understand the topic": """
Explain the topic clearly for a student who is learning it for the first time.
Use headings, subheadings, definitions, examples, applications, and a final short recap.
Keep the language simple but accurate and structured.
""",
        "2-Mark Answer": """
Create a very short answer suitable for a 2-mark question. Include a simple introduction,
one clear idea or definition, and one supporting point. Keep it brief and direct.
""",
        "5-Mark Exam Answer": """
Create a concise exam-ready 5-mark answer. Use a short introduction, definition if present,
3-5 important points or subtopics, one or two relevant examples, and a brief conclusion.
Do not over-expand.
""",
        "10-Mark Exam Answer": """
Create a detailed exam-ready 10-mark answer. Preserve the key ideas from the source,
including definition, main concepts, classifications or types, examples, applications,
advantages, limitations, and a conclusion. Use a clear sectioned structure and end with
short revision points.
""",
        "Assignment": """
Create a polished academic assignment from the source. Use an introduction, definition or
background where appropriate, logical subtopics, examples, applications, advantages and
limitations where relevant, a conclusion, and a short summary. Do not invent facts.
""",
        "Quick Revision": """
Create high-value revision notes. Use headings, very short bullets, definitions, key terms,
examples, and a final recap. Keep the answer concise and easy to memorize.
""",
        "Seminar / Presentation": """
Turn the source into presentation-ready content. Organize it into slide-friendly sections:
introduction, main concepts, classifications, examples, applications, advantages and
limitations, and conclusion. Use brief bullet point sections with clear headings.
""",
    }

    instruction = purpose_rules.get(purpose, purpose_rules["Understand the topic"])

    return f"""
You are SmartStudy AI, a purpose-aware educational content organizer.

SOURCE MATERIAL:
----------------
{source}
----------------

USER PURPOSE:
{purpose}

TASK:
{instruction}

STRICT CONTENT RULES:
1. Do not merely summarize into one paragraph.
2. Preserve the important concepts and relationships in the source.
3. Organize information into meaningful headings and subheadings.
4. Keep definitions, examples, steps, classifications, applications, advantages,
   limitations, and conclusions when they exist in the source.
5. Remove repetition, filler, and information that is not useful for the selected purpose.
6. Do not claim that removed information was unimportant; say it was shortened or omitted
   for the selected format.
7. Do not fabricate citations, statistics, quotations, names, or facts.
8. If the source is incomplete, say what is missing instead of inventing it.
9. Make the result easy for a student to read, study, and reproduce.
10. At the end include these exact sections:
   ## Short Summary
   ## What Was Preserved
   ## What Was Shortened or Removed
   ## Quick Revision

Return clean Markdown only.
"""


def generate_with_openai(source, purpose, model="gpt-4.1-mini"):
    if OpenAI is None:
        raise RuntimeError("The OpenAI package is not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it to .streamlit/secrets.toml or the OPENAI_API_KEY environment variable."
        )

    requested_model = (model or "gpt-4.1-mini").strip()
    if requested_model in {"gpt-5.6-luna", "gpt-5.6", "gpt-5"}:
        requested_model = "gpt-4.1-mini"

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=requested_model,
            input=build_prompt(source, purpose),
        )
    except Exception as exc:
        message = str(exc)
        if "model" in message.lower():
            raise RuntimeError(
                f"OpenAI model configuration failed: {message}. Use a supported model such as gpt-4.1-mini."
            ) from exc
        raise RuntimeError(f"OpenAI request failed: {message}") from exc

    answer = normalize_markdown(getattr(response, "output_text", "")).strip()
    if not answer:
        raise RuntimeError("OpenAI returned an empty response.")

    summary_words = len(re.findall(r"\b\w+\b", answer))
    original_words = len(re.findall(r"\b\w+\b", source))
    compression = ((original_words - summary_words) / original_words) * 100 if original_words else 0.0

    preserved = extract_section(
        answer,
        ["## What Was Preserved", "## What was preserved"],
        ["## What Was Shortened or Removed", "## What was shortened or removed"],
    )
    removed = extract_section(
        answer,
        ["## What Was Shortened or Removed", "## What was shortened or removed"],
        ["## Quick Revision"],
    )
    revision = extract_section(
        answer,
        ["## Quick Revision"],
        [],
    )

    return {
        "answer": answer,
        "summary_words": summary_words,
        "compression": compression,
        "preserved": preserved or "See the generated answer above.",
        "removed": removed or "Repetition and purpose-irrelevant detail were shortened.",
        "revision": revision or "Review the headings and key terms in the generated answer.",
        "model_used": requested_model,
    }


def extract_section(text, starts, ends):
    start_pos = -1
    chosen = None
    for s in starts:
        p = text.lower().find(s.lower())
        if p != -1 and (start_pos == -1 or p < start_pos):
            start_pos = p
            chosen = s

    if start_pos == -1:
        return ""

    body_start = start_pos + len(chosen)
    end_pos = len(text)
    for e in ends:
        p = text.lower().find(e.lower(), body_start)
        if p != -1:
            end_pos = min(end_pos, p)

    return text[body_start:end_pos].strip()


def extractive_sentences(text, n=5):
    sentences = split_sentences(text)
    if len(sentences) <= n:
        return sentences

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(sentences)
        scores = matrix.sum(axis=1).A1
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n]
        return [sentences[i] for i in sorted(ranked)]
    except Exception:
        return sentences[:n]


def generate_fallback(source, purpose):
    """
    Local NLP generation mode.

    This function does not use the OpenAI API.
    It creates purpose-aware educational content from the
    source text using extractive NLP techniques.
    """

    import re

    # ---------------------------------------------------------
    # Basic sentence extraction
    # ---------------------------------------------------------

    sentences = split_sentences(source)

    if not sentences:
        sentences = [source.strip()]

    selected = extractive_sentences(
        source,
        min(8, max(4, len(sentences)))
    )

    if not selected:
        selected = sentences[:8]

    # Remove duplicate sentences while preserving order
    clean_selected = []

    for sentence in selected:
        sentence = sentence.strip()

        if sentence and sentence not in clean_selected:
            clean_selected.append(sentence)

    selected = clean_selected

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------

    def clean_text(text):
        text = str(text).replace("\\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def bullet_list(items):
        return "\n".join(
            f"- {clean_text(item)}"
            for item in items
            if clean_text(item)
        )

    def numbered_list(items):
        return "\n".join(
            f"{i}. {clean_text(item)}"
            for i, item in enumerate(items, 1)
            if clean_text(item)
        )

    def keyword_list(items):
        return "\n".join(
            f"- **{clean_text(item)}**"
            for item in items
            if clean_text(item)
        )

    # ---------------------------------------------------------
    # Analyze source
    # ---------------------------------------------------------

    stats = analyze_text(source)

    raw_keywords = stats.get("keywords", [])

    # Filter weak/generic keywords
    stop_words = {
        "does", "make", "makes", "made", "try", "tries",
        "using", "used", "use", "thing", "things",
        "important", "main", "also", "many", "much",
        "people", "person", "example", "examples",
        "work", "works", "way", "ways", "good", "better",
        "need", "needs", "can", "could", "would",
        "should", "will", "may", "might"
    }

    keywords = []

    for word in raw_keywords:

        word = clean_text(word).lower()

        if not word:
            continue

        if word in stop_words:
            continue

        if len(word) < 3:
            continue

        if word not in keywords:
            keywords.append(word)

    keywords = keywords[:8]

    # If keyword extraction is weak, use meaningful words
    if not keywords:

        words = re.findall(
            r"\b[a-zA-Z][a-zA-Z-]{3,}\b",
            source.lower()
        )

        frequency = {}

        for word in words:

            if word in stop_words:
                continue

            frequency[word] = frequency.get(word, 0) + 1

        keywords = [
            word
            for word, count in sorted(
                frequency.items(),
                key=lambda x: (-x[1], x[0])
            )[:8]
        ]

    # ---------------------------------------------------------
    # Important sentences
    # ---------------------------------------------------------

    first = selected[0] if selected else source.strip()

    middle = selected[1:-1] if len(selected) > 2 else selected

    last = selected[-1] if selected else source.strip()

    important_points = selected[:6]

    short = " ".join(selected[:3])

    # ---------------------------------------------------------
    # Purpose-specific generation
    # ---------------------------------------------------------

    if purpose == "Understand the Topic":

        answer = f"""# Understanding the Topic

## Introduction

{first}

## Main Concepts

{bullet_list(middle[:5])}

## Key Terms

{keyword_list(keywords)}

## Explanation

The topic can be understood through the main ideas presented in the source. The important concepts are connected through the information given above.

## Conclusion

{last}

## Short Summary

{short}

## What Was Preserved

- Main concepts and important statements.
- Key terminology from the source.
- Important supporting information.

## What Was Shortened or Removed

- Repeated statements.
- Less important supporting details.

## Quick Revision

{bullet_list(important_points[:5])}
"""

    elif purpose == "2-Mark Answer":

        answer = f"""# 2-Mark Answer

## Answer

{short}

## Key Point

{first}

## Key Terms

{", ".join(keywords)}

## Quick Revision

- Remember the main definition or idea.
- Remember the key terms.
- Focus on the central concept.
"""

    elif purpose == "5-Mark Exam Answer":

        answer = f"""# 5-Mark Exam Answer

## 1. Introduction

{first}

## 2. Main Points

{bullet_list(middle[:4])}

## 3. Key Terms

{keyword_list(keywords)}

## 4. Conclusion

{last}

## Exam Tip

Remember the main concept, supporting points, and important terminology.

## Short Summary

{short}

## What Was Preserved

- Main concepts.
- Important supporting points.
- Key terminology.

## What Was Shortened or Removed

- Repeated wording.
- Less important details.

## Quick Revision

{bullet_list(important_points[:5])}
"""

    elif purpose == "10-Mark Exam Answer":

        main_sections = []

        for i, sentence in enumerate(middle[:6], 1):
            main_sections.append(
                f"### 2.{i} Key Point\n\n{clean_text(sentence)}"
            )

        main_content = "\n\n".join(main_sections)

        answer = f"""# 10-Mark Exam Answer

## 1. Introduction

{first}

## 2. Detailed Explanation

{main_content}

## 3. Key Terms

{keyword_list(keywords)}

## 4. Important Points

{bullet_list(important_points)}

## 5. Conclusion

{last}

## Short Summary

{short}

## What Was Preserved

- Main concepts and supporting explanations.
- Important terminology.
- Important examples or statements present in the source.

## What Was Shortened or Removed

- Repeated information.
- Less relevant wording.
- Redundant explanations.

## Quick Revision

{bullet_list(important_points[:6])}
"""

    elif purpose == "Assignment":

        main_sections = []

        for i, sentence in enumerate(middle[:6], 1):
            main_sections.append(
                f"### 2.{i} Important Point\n\n{clean_text(sentence)}"
            )

        main_content = "\n\n".join(main_sections)

        answer = f"""# Assignment

## 1. Introduction

{first}

## 2. Main Content

{main_content}

## 3. Key Concepts

{bullet_list(important_points[:6])}

## 4. Key Terms

{keyword_list(keywords)}

## 5. Conclusion

{last}

## Short Summary

{short}

## What Was Preserved

- Central ideas from the source.
- Important concepts and terminology.
- Major supporting statements.

## What Was Shortened or Removed

- Repeated wording.
- Less informative sentences.
- Redundant details.

## Quick Revision

{bullet_list(important_points[:6])}
"""

    elif purpose == "Quick Revision":

        answer = f"""# Quick Revision Notes

## Key Terms

{keyword_list(keywords)}

## Important Points

{bullet_list(important_points)}

## One-Minute Summary

{short}

## Remember

- Focus on the main concept.
- Remember the important supporting points.
- Revise the key terms before the examination.

## Short Summary

{short}

## What Was Preserved

- Core concepts.
- Key terminology.
- Important points.

## What Was Shortened or Removed

- Repeated information.
- Less central details.

## Quick Revision

{bullet_list(important_points[:5])}
"""

    elif purpose == "Seminar / Presentation":

        answer = f"""# Seminar / Presentation

## 1. Introduction

{first}

## 2. Main Points

{bullet_list(middle[:5])}

## 3. Key Terms

{keyword_list(keywords)}

## 4. Important Takeaways

{bullet_list(important_points[:5])}

## 5. Conclusion

{last}

## Presentation Summary

{short}

## What Was Preserved

- Main concepts.
- Important supporting points.
- Key terminology.

## What Was Shortened or Removed

- Repeated statements.
- Less important details.

## Quick Revision

{bullet_list(important_points[:5])}
"""

    else:

        answer = f"""# {purpose}

## Introduction

{first}

## Main Ideas

{bullet_list(middle[:6])}

## Key Terms

{keyword_list(keywords)}

## Conclusion

{last}

## Short Summary

{short}

## What Was Preserved

- Main ideas.
- Important concepts.
- Key terminology.

## What Was Shortened or Removed

- Repeated wording.
- Less important information.

## Quick Revision

{bullet_list(important_points[:5])}
"""

    # ---------------------------------------------------------
    # Calculate generated word count
    # ---------------------------------------------------------

    generated_words = len(
        re.findall(
            r"\b[\w'-]+\b",
            answer
        )
    )

    original_words = len(
        re.findall(
            r"\b[\w'-]+\b",
            source
        )
    )

    # Positive = compression
    # Negative = expansion
    if original_words > 0:

        compression = (
            (original_words - generated_words)
            / original_words
        ) * 100

    else:

        compression = 0.0

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------

    return {
        "answer": answer,
        "summary_words": generated_words,
        "compression": compression,
        "preserved": (
            "- Main concepts and important information were preserved.\n"
            "- Key terminology and major supporting points were retained."
        ),
        "removed": (
            "- Repeated wording was reduced.\n"
            "- Less important or redundant information was shortened."
        ),
        "revision": bullet_list(important_points[:5]),
        "model_used": "Local NLP",
    }