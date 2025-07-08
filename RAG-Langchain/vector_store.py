import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qdrant_models
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# Load env vars
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_vector_store(file_path: str, collection_name: str, recreate:bool, vector_size:int,embedder):
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

    # Step 3: Create or Recreate collection

    existing_collections = client.get_collections().collections
    collection_names = [col.name for col in existing_collections]


    if recreate and collection_name in collection_names:
        client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,  
            distance=qdrant_models.Distance.COSINE,
        )
         )  
    elif(collection_name not in collection_names):
        client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE
        )
        )
    else:
        print(f"📂 Adding to existing collection: {collection_name}")

    # Step 4: Create vectorstore and insert documents
    vectorstore = QdrantVectorStore(
    embedding=embedder,
    collection_name=collection_name,
    client=client
        )

    vectorstore.add_documents(chunks)

    print(f"✅ PDF at'{file_path}' processed.")

