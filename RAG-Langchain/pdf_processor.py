import os
from pathlib import Path
from vector_store import create_vector_store
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

# Load env vars
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def process_all_pdfs_in_folder(pdf_root_folder: Path, recreate:bool, vector_size:int, embedder):
    """
    Creates vector stores for all PDFs in a folder. Returns retrievers list.
    """

    retrievers = []

    if not pdf_root_folder.exists():
        print(f"❌ PDF root folder not found: {pdf_root_folder}")
        return []
    
    subfolders = [f for f in pdf_root_folder.iterdir() if f.is_dir()]

    if not subfolders:
        print("❌ No subfolders (collections) found inside PDF root.")
        return []
    
    for subfolder in subfolders:
        collection_name = subfolder.name
        pdf_files = sorted([f for f in subfolder.iterdir() if f.suffix == ".pdf"])

        if not pdf_files:
            print(f"⚠️ No PDFs found in subfolder: {collection_name}")
            continue

        print(f"\n📂 Creating collection: {collection_name} ({len(pdf_files)} PDFs)")

        for pdf_file in pdf_files:
            create_vector_store(
                file_path=str(pdf_file),
                collection_name=collection_name,
                recreate=recreate,
                vector_size=vector_size,
                embedder=embedder
            )

        retriever = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=embedder,
            url=QDRANT_URL
        )
        retrievers.append(retriever)

    return retrievers
