from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    template = """
    You are a helpful assistant. Use the following context to answer the user's question.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain, retriever

