
import streamlit as st

from pdf_processor import extract_pdf_text
from rag_pipeline import create_vector_store
from lease_analyzer import analyze_lease_question


st.set_page_config(
    page_title="LeaseGuard AI",
    page_icon="🏠",
    layout="wide"
)

st.title("LeaseGuard AI")
st.write("Understand your lease with AI-powered analysis.")

st.caption(
    "Document review assistance only. Not legal advice. "
    "Uploaded text is sent to an external AI provider."
)

uploaded_file = st.file_uploader(
    "Upload your lease agreement",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        pages = extract_pdf_text(uploaded_file)

        st.success(
            f"Successfully processed {len(pages)} pages."
        )

        with st.spinner("Preparing lease document..."):
            vector_store, chunks = create_vector_store(pages)

        question = st.text_input(
            "Ask a question about your lease",
            placeholder=(
                "What happens if I terminate my lease early?"
            )
        )

        if st.button("Analyze Lease", type="primary"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Analyzing your lease..."):
                    result = analyze_lease_question(
                        vector_store,
                        question
                    )

                st.subheader("AI Explanation")
                st.markdown(result["answer"])

    except Exception as error:
        st.error(
            "Unable to process the lease. "
            "Please check the PDF and try again."
        )
        st.exception(error)
