import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def rag_simple(query, retrievr, model=None, top_k=5):
    """Retrieve relevant context and generate an answer with the configured LLM."""
    if model is None:
        if not groq_api_key:
            return "GROQ_API_KEY is not set. Please configure it in your environment."
        model = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
        )

    retrieved_docs = retrievr.retrieve(query=query, top_k=top_k)

    if not retrieved_docs:
        return "No relevant documents found."

    context = "\n\n".join(
        [f"Document ID: {doc['id']}\nContent: {doc['document']}" for doc in retrieved_docs]
    )
    prompt = (
        "Answer the following question based on the provided context:\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = model.invoke([prompt])

    return response