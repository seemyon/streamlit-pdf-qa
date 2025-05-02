from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.from_documents(splits, embeddings)

