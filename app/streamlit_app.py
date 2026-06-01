import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import pandas as pd

from src.document_loader import load_document
from src.text_cleaner import clean_text
from src.llm_chains import summarize_document, extract_clauses
from src.risk_analyzer import analyze_risks


st.set_page_config(
    page_title="Legal Document Intelligence Tool",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document Intelligence Tool")
st.write(
    "Upload a legal or policy document to generate a summary, extract key clauses, "
    "and identify potential risks."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT document",
    type=["pdf", "txt"]
)

analysis_button = st.sidebar.button("Run Analysis")

if uploaded_file is None:
    st.info("Please upload a PDF or TXT document to begin.")
    st.stop()

with st.expander("Document Preview", expanded=True):
    raw_text = load_document(uploaded_file)
    cleaned_text = clean_text(raw_text)

    st.text_area(
        "Extracted Text",
        cleaned_text[:5000],
        height=300
    )

if analysis_button:
    with st.spinner("Analyzing document..."):
        summary = summarize_document(cleaned_text)
        clauses = extract_clauses(cleaned_text)
        risks = analyze_risks(cleaned_text, clauses)

    st.subheader("1. Document Summary")
    st.write(summary)

    st.subheader("2. Key Clause Extraction")
    st.write(clauses)

    st.subheader("3. Risk Table")
    risk_df = pd.DataFrame(risks)
    st.dataframe(risk_df, use_container_width=True)

    csv = risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Risk Table as CSV",
        data=csv,
        file_name="risk_table.csv",
        mime="text/csv"
    )