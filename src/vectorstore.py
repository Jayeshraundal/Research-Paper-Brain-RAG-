import os
import uuid
from pathlib import Path
from typing import Any, List, Dict
import numpy as np
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding import EmbeddingManager

class VectorStore:
    """Create and manage a persistent ChromaDB vector store for document embeddings."""

    def __init__(self, collection_name: str = "pdf_collection", persist_directory: str | None = None):
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = str(Path(__file__).resolve().parents[1] / "data" / "vectorestore" / "chroma_db")
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB client and create or open the target collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Collection of PDF document embeddings for RAG"})
            print(f"Vector store initialized at {self.persist_directory} with collection '{self.collection_name}'.")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e

    def add_embeddings(self, documents: List[Any], embeddings: np.ndarray):
        """Store document chunks and their embeddings in the vector database."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")


        ids =[]
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            metadatas.append(doc.metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())


        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Added {len(documents)} documents to the vector store.")

        except Exception as e:
            print(f"Error adding embeddings to vector store: {e}")
            raise e    

class RagRetriever:
    """Retrieve the most relevant document chunks for a user query."""

    def __init__(self, vector_store:VectorStore, embedding_manager: EmbeddingManager):
        
        self.vector_store = vector_store
        self.embedding_manager= embedding_manager

    def retrieve (self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Embed the query, search the vector store, and return relevant chunks."""
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["metadatas", "documents", "embeddings"]
            )

            retrieved_docs = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                document_text = results["documents"][0][i]
                embedding = np.array(results["embeddings"][0][i])

                similarity_score = cosine_similarity([query_embedding], [embedding])[0][0]

                if similarity_score >= similarity_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "document": document_text,
                        "similarity_score": similarity_score
                    })

            return retrieved_docs
        except Exception as e:
            print(f"Error occurred while retrieving documents: {e}")
            return []
