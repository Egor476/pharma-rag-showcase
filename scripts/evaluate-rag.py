import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Model & Evaluation Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))

# Database & Path References
CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma-db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pharma_dita_topics")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
PROMPT_LIBRARY_DIR = PROJECT_ROOT / os.getenv("PROMPT_LIBRARY_DIR", "prompt-library")
SCHEMAS_DIR = PROJECT_ROOT / os.getenv("SCHEMAS_DIR", "schemas")
TEST_DATASET_PATH = PROJECT_ROOT / os.getenv("TEST_DATASET_PATH", "data/test-dataset.json")

DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", 4))
METADATA_FILTER_ENABLED = os.getenv("METADATA_FILTER_ENABLED", "true").lower() == "true"