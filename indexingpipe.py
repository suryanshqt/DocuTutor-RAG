import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = "http://localhost:6333"
# Flat file that tracks whichever collection is currently active
ACTIVE_COLLECTION_FILE = Path(__file__).parent / ".active_collection"


def get_active_collection() -> str | None:
    """Returns the currently active collection name, or None if none exists."""
    if ACTIVE_COLLECTION_FILE.exists():
        return ACTIVE_COLLECTION_FILE.read_text().strip() or None
    return None


def set_active_collection(name: str):
    """Persists the active collection name to disk."""
    ACTIVE_COLLECTION_FILE.write_text(name)


def make_collection_name(filename: str) -> str:
    """
    Converts a PDF filename into a safe Qdrant collection name.
    e.g. 'Node.js Guide (2024).pdf' -> 'nodejs_guide_2024'
    """
    stem = Path(filename).stem                     # strip .pdf extension
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem)    # replace special chars with _
    slug = slug.strip("_").lower()[:60]            # trim, lowercase, max 60 chars
    return slug or "document"


def index_document(file_name: str = "Nodejs.pdf") -> str:
    """
    Loads a PDF, chunks it, and saves it to a NEW Qdrant collection.
    Deletes the previous collection so only one PDF is active at a time.
    Returns the new collection name.
    """
    pdf_path = Path(file_name)
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / file_name

    print(f"[INFO] Initializing indexing process for: {pdf_path}")

    client = QdrantClient(url=QDRANT_URL)
    new_collection = make_collection_name(pdf_path.name)

    try:
        # 1. Delete the previous collection if one exists
        previous = get_active_collection()
        if previous:
            existing = [c.name for c in client.get_collections().collections]
            if previous in existing:
                print(f"[INFO] Deleting previous collection: '{previous}'")
                client.delete_collection(previous)

        # 2. Load the PDF
        loader = PyPDFLoader(file_path=str(pdf_path))
        docs = loader.load()
        print(f"[INFO] Loaded {len(docs)} pages from {pdf_path.name}")

        # 3. Split into chunks
        print("[INFO] Executing text splitting sequence...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(documents=docs)
        print(f"[INFO] Created {len(chunks)} chunks.")

        # 4. Embed and store in the new collection
        print(f"[INFO] Writing to new Qdrant collection: '{new_collection}'")
        embedding_models = OpenAIEmbeddings(model="text-embedding-3-large")
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_models,
            url=QDRANT_URL,
            collection_name=new_collection
        )

        # 5. Persist the new active collection name
        set_active_collection(new_collection)
        print(f"[SUCCESS] Indexing complete. Active collection is now '{new_collection}'.")
        return new_collection

    except FileNotFoundError:
        print(f"[ERROR] Could not locate {pdf_path}. Verify the file exists.")
        raise
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    index_document()