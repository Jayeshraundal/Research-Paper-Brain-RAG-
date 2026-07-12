import tqdm
import os
from typing import List, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


"""
this function processes PDF documents in a specified directory, extracts their content, and adds metadata to
each document. It returns a list of all processed documents.

"""


def process_documents(doc_dir):

    """Load all PDF files from a directory and return them as processed documents."""
    
    all_documents = []

    input_path = Path(doc_dir)
    if not input_path.is_absolute():
        input_path = (Path(__file__).resolve().parents[1] / input_path).resolve()

    pdf_directory = input_path
    pdf_files = list(pdf_directory.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files in {pdf_directory}")

    for pdf in pdf_files:
        print(f"Processing {pdf}...")

        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()

            ##adding the metadata to the documents

            for doc in documents:
                doc.metadata["source"] = pdf.name

            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {pdf}: {e}")   

    print(f"Total documents processed: {len(all_documents)}")
    return all_documents  

def split_documents(documents):

    """Split long documents into smaller chunks suitable for embedding and retrieval."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
