# Final app that calls functions from backend

import os
import streamlit as st
from dotenv import load_dotenv

from backend.document_loader import load_pdf, split_docs
from backend.vectorstore_utils import create_vectorstore
from backend.rag_pipeline import build_rag_chain
from backend.qa_runner import run_qa, show_sources

def load_api_key():
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(page_title="PDF QA with GPT-4", layout="centered")
    st.title("📄 Ask Questions About Your PDF")

    api_key = load_api_key()
    if not api_key:
        st.warning("Please set your OpenAI API key in a .env file or Streamlit secrets.")
        return

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        question = st.text_input("What would you like to ask about this PDF?")
        if question:
            with st.spinner("🔄 Processing..."):
                documents = load_pdf(uploaded_file)
                splits = split_docs(documents)
                vectorstore = create_vectorstore(splits)
                chain, retriever = build_rag_chain(vectorstore)
                response = run_qa(chain, question)

                st.subheader("💬 GPT-4's Answer")
                st.write(response.content)
                show_sources(retriever, question)

if __name__ == "__main__":
    main()

