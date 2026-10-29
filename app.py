
import re

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Parsing & Job Match Recommendation")
st.write(
    "Upload a PDF resume and compare it with job descriptions "
    "using NLP, TF-IDF, and cosine similarity."
)


def extract_resume_text(pdf_file):
    """Extract selectable text from an uploaded PDF."""
    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()


def clean_text(text):
    """Normalize text before comparing documents."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return text.strip()


resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
)

job_text = st.text_area(
    "Paste a job description",
    height=220,
    placeholder=(
        "Example: We are looking for a Python developer "
        "with SQL, NLP, and machine learning skills..."
    ),
)

if resume_file is not None:
    try:
        resume_text = extract_resume_text(resume_file)

        if not resume_text:
            st.error(
                "No selectable text was found. "
                "This PDF may be scanned or image-based."
            )
        else:
            st.subheader("Extracted Resume Text")
            st.text_area(
                "Review the extracted text",
                value=resume_text,
                height=200,
            )

            if job_text.strip():
                if st.button("Calculate Job Match"):
                    documents = [
                        clean_text(resume_text),
                        clean_text(job_text),
                    ]

                    if not all(documents):
                        st.warning(
                            "Please provide text in both fields."
                        )
                    else:
                        vectorizer = TfidfVectorizer(
                            stop_words="english"
                        )
                        vectors = vectorizer.fit_transform(documents)

                        score = cosine_similarity(
                            vectors[0:1],
                            vectors[1:2],
                        )[0][0] * 100

                        st.subheader("Job Match Results")
                        st.metric(
                            "Text Similarity Score",
                            f"{score:.2f}%",
                        )

                        resume_words = set(
                            documents[0].split()
                        )
                        job_words = set(
                            documents[1].split()
                        )

                        shared_words = sorted(
                            resume_words & job_words
                        )
                        missing_words = sorted(
                            job_words - resume_words
                        )

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Shared keywords**")
                            st.write(
                                ", ".join(shared_words[:40])
                                or "No shared keywords found."
                            )

                        with col2:
                            st.write("**Job keywords not found in resume**")
                            st.write(
                                ", ".join(missing_words[:40])
                                or "No missing keywords found."
                            )

                        st.caption(
                            "This score measures textual similarity, "
                            "not the probability of getting hired. "
                            "Keyword differences do not necessarily "
                            "mean a candidate lacks a skill."
                        )

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")

else:
    st.info("Upload a PDF resume to get started.") ;