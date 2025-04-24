import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from dotenv import load_dotenv

# Load env vars
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Global constants
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Initialize once
embedder = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_vector_store(file_path: str, collection_name: str):
    """
    Creates a Qdrant collection and adds embedded chunks from a PDF.
    """

    # Step 1: Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Step 2: Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    print(f"\n📘 Processing: {os.path.basename(file_path)}")
    print(f" - Pages: {len(pages)} | Chunks: {len(chunks)}")

    # Step 3: Recreate collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=1536,  # Based on text-embedding-3-small
            distance=qdrant_models.Distance.COSINE,
        )
    )

    # Step 4: Create vectorstore and insert documents
    vectorstore = QdrantVectorStore(
        embedding=embedder,
        collection_name=collection_name,
        client=client
    )
    vectorstore.add_documents(chunks)

    print(f"✅ Collection '{collection_name}' ready and populated.")

    # Step 5: Return retriever for this collection
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embedder,
        url=QDRANT_URL
    )
    return retriever
