# 🎓 SmartStudy AI

**Purpose-Aware NLP Study & Assignment Assistant**

SmartStudy AI takes study material and restructures it according to the student's purpose.

## Main modes

- Understand the topic
- 5-Mark Exam Answer
- 10-Mark Exam Answer
- Assignment
- Quick Revision
- Seminar / Presentation

The output is intentionally **not just a paragraph**. It can contain:

- Introduction
- Definitions
- Main topics
- Subtopics
- Examples
- Applications
- Advantages / limitations when supported
- Conclusion
- Short Summary
- What was preserved
- What was shortened or removed
- Quick revision points

## Technology

- Python
- Streamlit
- NLTK
- Scikit-learn / TF-IDF
- OpenAI Responses API (optional)
- pypdf
- python-docx

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

For AI generation, create:

`.streamlit/secrets.toml`

with:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

Then run:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload this project.
3. Set the main file to `app.py`.
4. In Streamlit Cloud, open **Settings → Secrets**.
5. Add:

```toml
OPENAI_API_KEY = "your_api_key_here"
```

6. Deploy.

## Without an API key

The application still works using a local TF-IDF/NLP fallback. The fallback is intentionally extractive and is not as strong as the AI mode for detailed assignment writing.

## Suggested college-project title

**Purpose-Aware Educational Text Summarization and Content Generation Using NLP**

## Suggested modules for your report

1. User Input and Document Processing
2. Text Extraction
3. NLP Preprocessing
4. Keyword and Sentence Importance Analysis
5. Purpose Selection
6. Purpose-Aware Content Generation
7. Structured Summary Generation
8. Content Preservation / Removal Analysis
9. Quick Revision Generation
10. Streamlit User Interface

## Important academic note

Generated answers should be checked against the original textbook or lecturer material before submission or examination. The system is designed to organize and rewrite source material, not to guarantee factual correctness.
