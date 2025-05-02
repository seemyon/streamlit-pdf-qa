import streamlit as st

def run_qa(chain, question):
    return chain.invoke(question)

def show_sources(retriever, question):
    docs = retriever.invoke(question)
    st.subheader("📚 Source Chunks Used:")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:300].replace("\n", " ") + "..."
        st.markdown(f"**🔹 Source #{i} — Page {page}**")
        st.markdown(f"`{snippet}`")


