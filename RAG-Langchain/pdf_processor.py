import os
from pathlib import Path
from vector_store import create_vector_store

def process_all_pdfs_in_folder(folder_path: str):
    """
    Creates vector stores for all PDFs in a folder. Returns retrievers list.
    """
    retrievers = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found.")
        return []

    for pdf in pdf_files:
        file_path = os.path.join(folder_path, pdf)
        collection_name = f"collection_{Path(pdf).stem.lower().replace(' ', '_')}"
        retriever = create_vector_store(file_path, collection_name)
        retrievers.append(retriever)

    return retrievers
