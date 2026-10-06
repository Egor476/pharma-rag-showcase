import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Ingestion configuration
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/dita-processed")
CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma-db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pharma_dita_topics")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")