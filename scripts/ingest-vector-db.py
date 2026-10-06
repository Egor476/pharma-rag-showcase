import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Path Anchors
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuration parameters with fallbacks matching .env.example
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/dita-processed")
CHROMA_PERSIST_DIRECTORY = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma-db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pharma_dita_topics")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def build_chunk_documents(json_data: dict) -> list[Document]:
    """
    Transforms a preprocessed DITA topic JSON into LangChain Documents.
    Each section is converted into an individual chunk with rich taxonomy metadata.
    """
    documents = []
    
    doc_id = json_data.get("doc_id", "")
    title = json_data.get("title", "")
    shortdesc = json_data.get("shortdesc", "")
    metadata_base = json_data.get("metadata", {})
    sections = json_data.get("sections", [])

    for sec_idx, section in enumerate(sections):
        section_title = section.get("section_title", "General")
        section_content = section.get("content", "").strip()

        if not section_content:
            continue

        page_content = f"Document: {title}\nSection: {section_title}\n\n{section_content}"

        chunk_metadata = {
            "doc_id": doc_id,
            "title": title,
            "shortdesc": shortdesc,
            "section_title": section_title,
            "section_index": sec_idx,
            "source_file": metadata_base.get("source_file", ""),
            "category": metadata_base.get("category", ""),
            "product": metadata_base.get("product", ""),
            "audience": metadata_base.get("audience", ""),
            "otherprops": metadata_base.get("otherprops", ""),
            "audience_list": ",".join(metadata_base.get("audience_list", [])),
            "otherprops_list": ",".join(metadata_base.get("otherprops_list", [])),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=chunk_metadata,
            )
        )

    return documents


def load_all_processed_topics(processed_dir: Path) -> list[Document]:
    """Recursively loads all processed DITA JSON files and builds chunks."""
    all_docs = []
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}. Run dita-preprocessor.py first.")

    json_files = list(processed_dir.rglob("*.json"))
    print(f"Found {len(json_files)} processed JSON files in {processed_dir}.\n")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                docs = build_chunk_documents(data)
                all_docs.extend(docs)
                print(f"  [+] Loaded {len(docs)} section chunk(s) from {json_file.name}")
        except Exception as e:
            print(f"  [!] Error processing {json_file.name}: {e}")

    return all_docs


def ingest_to_chroma():
    """Embeds chunked DITA documents into persistent ChromaDB storage."""
    print("==================================================")
    print(" PHARMA RAG: VECTOR DB INGESTION ")
    print("==================================================")

    # 1. Load documents
    print(f"Loading preprocessed DITA JSON files from:\n -> {DATA_PROCESSED_DIR}\n")
    documents = load_all_processed_topics(DATA_PROCESSED_DIR)

    if not documents:
        print("No valid section chunks found to ingest. Exiting.")
        return

    print(f"\nTotal section chunks prepared for embedding: {len(documents)}")

    # 2. Initialize Embeddings Engine
    print(f"\nInitializing SentenceTransformerEmbeddings ({EMBEDDING_MODEL_NAME})...")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 3. Create or Reset Chroma Vector Store
    persist_path = Path(CHROMA_PERSIST_DIRECTORY)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Persisting to ChromaDB directory: {persist_path}")
    print(f"Collection Name: '{CHROMA_COLLECTION_NAME}'\n")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_path),
    )

    vectorstore.persist()
    print("✓ Vector database successfully created and persisted.")

    # 4. Smoke Test Query & Metadata Validation
    print("\n--------------------------------------------------")
    print(" INGESTION SANITY CHECK & METADATA FILTER TEST ")
    print("--------------------------------------------------")
    
    test_query = "What are the common adverse reactions for Product A?"
    print(f"Test Query: '{test_query}'")
    
    test_filter = {"category": "oncology"}
    results = vectorstore.similarity_search(test_query, k=2, filter=test_filter)

    for idx, doc in enumerate(results, 1):
        print(f"\n[Result {idx}]")
        print(f"  Title: {doc.metadata.get('title')}")
        print(f"  Section: {doc.metadata.get('section_title')}")
        print(f"  Category: {doc.metadata.get('category')}")
        print(f"  Product: {doc.metadata.get('product')}")
        print(f"  Content Preview: {doc.page_content[:150].strip()}...")

    print("\n✓ Ingestion & sanity check complete!")


if __name__ == "__main__":
    ingest_to_chroma()