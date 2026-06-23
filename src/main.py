"""
streamlit_app.py - RAG Document Q&A Bot: Streamlit Web Interface

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import tempfile
import os

from rag import answer_question

# ── Assume this backend function exists (from rag.py) ─────────────────────────
# from rag import answer_question
#
# For development/demo without the backend, a mock is included below.
# Remove the mock and uncomment the import above when connecting the real backend.




# ── Page configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A Bot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Page background ── */
    .stApp {
        background-color: #0F1117;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Title block ── */
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        font-weight: 400;
        color: #F0F2F6;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        text-align: center;
        color: #8B8FA8;
        font-size: 0.95rem;
        margin-bottom: 2.5rem;
        letter-spacing: 0.01em;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #1A1D27;
        border: 1.5px dashed #2E3147;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #5B6BF8;
    }

    /* ── Text input ── */
    [data-testid="stTextInput"] input {
        background: #1A1D27 !important;
        border: 1.5px solid #2E3147 !important;
        border-radius: 10px !important;
        color: #F0F2F6 !important;
        font-size: 0.97rem !important;
        padding: 0.65rem 1rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #5B6BF8 !important;
        box-shadow: 0 0 0 3px rgba(91,107,248,0.15) !important;
    }

    /* ── Primary button ── */
    .stButton > button {
        background: linear-gradient(135deg, #5B6BF8 0%, #7C3AED 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        width: 100%;
        transition: opacity 0.15s, transform 0.1s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Answer card ── */
    .answer-card {
        background: #1A1D27;
        border: 1px solid #2E3147;
        border-left: 4px solid #5B6BF8;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        color: #E2E4EF;
        font-size: 1rem;
        line-height: 1.75;
        margin-top: 0.5rem;
    }

    /* ── Source badge ── */
    .source-badge {
        display: inline-block;
        background: #1E2235;
        border: 1px solid #2E3147;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        color: #8B8FA8;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }
    .source-badge span {
        color: #5B6BF8;
        font-weight: 600;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5B6BF8;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #13151F !important;
        border-right: 1px solid #1E2235;
    }
    .sidebar-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-bottom: 1.1rem;
    }
    .step-num {
        background: #1E2235;
        color: #5B6BF8;
        font-weight: 700;
        font-size: 0.8rem;
        border-radius: 50%;
        min-width: 26px;
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 1px;
    }
    .step-text {
        color: #9BA1B8;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .step-text strong {
        color: #C8CADB;
        display: block;
        margin-bottom: 0.1rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #1E2235;
        margin: 1.4rem 0;
    }
    .sidebar-tip {
        background: #1A1D27;
        border: 1px solid #2E3147;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #7B8096;
        font-size: 0.82rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<p style='color:#5B6BF8; font-weight:700; font-size:0.8rem;"
        "text-transform:uppercase; letter-spacing:0.1em;'>How it works</p>",
        unsafe_allow_html=True,
    )

    steps = [
        ("Upload", "Add a PDF or TXT file using the uploader on the main page."),
        ("Ask",    "Type your question in plain English — no special syntax needed."),
        ("Review", "Read the answer and check the cited source pages for verification."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-num">{i}</div>
            <div class="step-text"><strong>{title}</strong>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#5B6BF8; font-weight:700; font-size:0.8rem;"
        "text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.6rem;'>Tips</p>",
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="sidebar-tip">
        • Ask specific questions for more precise answers.<br>
        • The bot answers <em>only</em> from the uploaded document — it will not guess.<br>
        • Supported formats: <strong>PDF</strong> and <strong>TXT</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#3D4060; font-size:0.75rem; text-align:center;'>"
        "Powered by ChromaDB · HuggingFace · Gemini"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Main layout ────────────────────────────────────────────────────────────────

# Centre-column layout: narrow gutters on each side
gutter_l, col_main, gutter_r = st.columns([0.5, 9, 0.5])

with col_main:

    # ── Hero title ─────────────────────────────────────────────────────────────
    st.markdown('<h1 class="hero-title">📄 RAG Document Q&A Bot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Upload a document · Ask a question · Get a grounded answer</p>',
        unsafe_allow_html=True,
    )

    # ── Two-column input layout ────────────────────────────────────────────────
    col_upload, col_question = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.markdown('<p class="section-label">Document</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="Upload a PDF or TXT file",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            help="Supported: PDF, TXT",
        )
        if uploaded_file:
            st.markdown(
                f"<p style='color:#5B6BF8; font-size:0.82rem; margin-top:0.4rem;'>"
                f"✓ &nbsp;<strong>{uploaded_file.name}</strong> ready</p>",
                unsafe_allow_html=True,
            )

    with col_question:
        st.markdown('<p class="section-label">Your Question</p>', unsafe_allow_html=True)
        question = st.text_input(
            label="Question",
            placeholder="e.g. What are the key findings of this report?",
            label_visibility="collapsed",
        )
        run_button = st.button("Get Answer →", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Validation & execution ─────────────────────────────────────────────────
    if run_button:
        if not uploaded_file:
            st.warning("⚠️  Please upload a document before asking a question.", icon=None)
        elif not question.strip():
            st.warning("⚠️  Please type a question to continue.", icon=None)
        else:
            # Save upload to a temp file so the backend can read it
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                with st.spinner("Searching your document and generating an answer …"):
                    result = answer_question(file_path=tmp_path, question=question)

                answer_text = result.get("answer", "No answer returned.")
                sources      = result.get("sources", [])

                # ── Answer card ────────────────────────────────────────────────
                st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-card">{answer_text}</div>',
                    unsafe_allow_html=True,
                )

                # ── Source citations ───────────────────────────────────────────
                if sources:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<p class="section-label">Sources</p>', unsafe_allow_html=True)
                    badges = "".join(
                        f'<span class="source-badge">'
                        f'📄 {s["source"]} &nbsp;·&nbsp; <span>p. {s["page"]}</span>'
                        f'</span>'
                        for s in sources
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<p style='color:#5C6070; font-size:0.83rem; margin-top:0.8rem;'>"
                        "No source citations returned.</p>",
                        unsafe_allow_html=True,
                    )

            except Exception as err:
                st.error(
                    f"Something went wrong while generating an answer: {err}\n\n"
                    "Check that your API key is set and the document is readable."
                )
            finally:
                # Always clean up the temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # ── Empty state hint (before any interaction) ──────────────────────────────
    elif not run_button and not uploaded_file:
        st.markdown(
            "<p style='color:#2E3147; font-size:0.88rem; text-align:center;"
            "margin-top:1.5rem;'>"
            "Upload a document and ask a question to get started.</p>",
            unsafe_allow_html=True,
        )