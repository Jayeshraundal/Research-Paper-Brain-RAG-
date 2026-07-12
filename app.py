from src.data_loader import process_documents, split_documents


if __name__ == "__main__":

    doc_dir = "data/docs"
    processed_documents = process_documents(doc_dir)
    chunked_documents = split_documents(processed_documents, chunk_size=1000, chunk_overlap=200)
    print(f"Processed {len(processed_documents)} documents.")