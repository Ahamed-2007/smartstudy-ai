import streamlit as st

from summarizer import (
    extract_text_from_upload,
    analyze_text,
    generate_fallback,
)


st.set_page_config(
    page_title="SmartStudy AI",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------
# Page Header
# ---------------------------------------------------------

st.title("🎓 SmartStudy AI")
st.caption("Purpose-Aware Educational Text Summarization and Content Generation Using NLP")

st.info(
    "🧠 Local NLP mode is active. "
    "This version does not require an OpenAI API key or paid API credits."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("🎯 Choose Your Purpose")

    purpose = st.selectbox(
        "What do you need this content for?",
        [
            "Understand the Topic",
            "2-Mark Answer",
            "5-Mark Exam Answer",
            "10-Mark Exam Answer",
            "Assignment",
            "Quick Revision",
            "Seminar / Presentation",
        ],
    )

    st.divider()

    st.subheader("⚙️ Generation Mode")

    st.success("Local NLP Generation")

    st.caption(
        "Your study material is processed using the local NLP "
        "system. No OpenAI API credits are required."
    )


# ---------------------------------------------------------
# Main Instructions
# ---------------------------------------------------------

st.markdown(
    """
### 📚 Enter Your Study Material

Paste your textbook material, class notes, article, or assignment
source below.

SmartStudy AI restructures the material according to your selected
purpose instead of producing only a simple paragraph summary.
"""
)


# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------

uploaded = st.file_uploader(
    "📂 Upload a TXT, PDF, or DOCX file (optional)",
    type=["txt", "pdf", "docx"],
)


# ---------------------------------------------------------
# Text Input
# ---------------------------------------------------------

text_input = st.text_area(
    "📖 Source Content",
    height=300,
    placeholder=(
        "Paste your study material here...\n\n"
        "Example: Machine Learning is a branch of Artificial "
        "Intelligence that enables computers to learn patterns "
        "from data..."
    ),
)


source_text = text_input.strip()


# ---------------------------------------------------------
# Uploaded File Processing
# ---------------------------------------------------------

if uploaded is not None:
    try:
        source_text = extract_text_from_upload(uploaded)

        st.success(f"✅ Loaded: {uploaded.name}")

        with st.expander("👀 Preview Extracted Text"):
            st.write(source_text[:5000])

    except Exception as e:
        st.error(f"❌ Could not read the file: {e}")


# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    generate = st.button(
        "✨ Generate",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear = st.button(
        "🧹 Clear",
        use_container_width=True,
    )


if clear:
    st.rerun()


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

if generate:

    if len(source_text.split()) < 30:
        st.warning(
            "⚠️ Please provide at least 30 words of source content."
        )
        st.stop()

    with st.spinner(
        "🧠 Analyzing and structuring your content using Local NLP..."
    ):

        # Analyze original text
        stats = analyze_text(source_text)

        # Generate using LOCAL NLP only
        result = generate_fallback(
            source_text,
            purpose,
        )

        # Store results
        st.session_state["result"] = result
        st.session_state["stats"] = stats
        st.session_state["used_ai"] = False
        st.session_state["ai_model_used"] = "Local NLP"
        st.session_state["ai_error"] = None


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

if "result" in st.session_state:

    st.divider()

    stats = st.session_state["stats"]
    result = st.session_state["result"]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    st.subheader("📊 Content Metrics")

    a, b, c, d = st.columns(4)

    a.metric(
        "Original Words",
        stats.get("original_words", "—"),
    )

    b.metric(
        "Generated Words",
        result.get("summary_words", "—"),
    )

    c.metric(
        "Original Sentences",
        stats.get("sentences", "—"),
    )


    # Correctly handle compression / expansion
    compression_value = float(
        result.get("compression", 0.0) or 0.0
    )

    if compression_value >= 0:

        d.metric(
            "Compression",
            f"{compression_value:.1f}%",
        )

    else:

        d.metric(
            "Expansion",
            f"{abs(compression_value):.1f}%",
        )


    # -----------------------------------------------------
    # Generation Status
    # -----------------------------------------------------

    st.success(
        "✅ Generated using Local NLP — no OpenAI API required."
    )


    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------

    tab1, tab2, tab3 = st.tabs(
        [
            "📖 Final Answer",
            "🔎 Content Analysis",
            "⚡ Quick Revision",
        ]
    )


    # -----------------------------------------------------
    # Final Answer
    # -----------------------------------------------------

    with tab1:

        st.subheader(
            f"📝 {purpose}"
        )

        st.markdown(
            result.get(
                "answer",
                "No answer was generated.",
            )
        )


        st.download_button(
            "⬇️ Download as Markdown",
            data=result.get("answer", ""),
            file_name="smartstudy_answer.md",
            mime="text/markdown",
        )


    # -----------------------------------------------------
    # Content Analysis
    # -----------------------------------------------------

    with tab2:

        st.subheader("✅ What Was Preserved")

        st.markdown(
            result.get(
                "preserved",
                "Main concepts and important information were preserved.",
            )
        )


        st.subheader("✂️ What Was Shortened or Removed")

        st.markdown(
            result.get(
                "removed",
                "Repeated or less important information was shortened.",
            )
        )


        st.subheader("🔑 Detected Key Topics")

        keywords = stats.get("keywords", [])

        if keywords:

            st.write(
                ", ".join(keywords)
            )

        else:

            st.write(
                "No major keywords detected."
            )


    # -----------------------------------------------------
    # Quick Revision
    # -----------------------------------------------------

    with tab3:

        st.subheader("⚡ Quick Revision")

        st.markdown(
            result.get(
                "revision",
                "Revision points are not available.",
            )
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "SmartStudy AI is an educational NLP assistant. "
    "Always compare generated answers with your textbook, "
    "lecturer notes, and required academic format."
)