from src.data_loader import process_documents, split_documents
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore, RagRetriever
from src.search import rag_simple

if __name__ == "__main__":
    # Path to the PDF documents used for the RAG pipeline
    doc_dir = "data/docs"

    # Step 1: Load PDF documents and keep them as LangChain documents
    processed_documents = process_documents(doc_dir)

    # Step 2: Split long documents into smaller chunks for retrieval
    chunked_documents = split_documents(processed_documents)

    # Step 3: Create embeddings for each text chunk
    embeddings = EmbeddingManager().generate_embeddings(
        [doc.page_content for doc in chunked_documents]
    )

    # Step 4: Store the chunks and embeddings in ChromaDB
    vector_store = VectorStore()
    vector_store.add_embeddings(chunked_documents, embeddings)
    print(f"Stored {len(chunked_documents)} chunks in the vector store.")

    # Step 5: Build a retriever and answer a sample question
    embedding_manager = EmbeddingManager()
    retriever = RagRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    query = "what is the attention mechanism"
    answer = rag_simple(query, retriever)
    print(answer)
