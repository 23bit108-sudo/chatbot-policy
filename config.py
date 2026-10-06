import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _setting(name: str, default: str = "") -> str:
    """Read a setting from environment/.env or Streamlit Cloud secrets."""
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        # Streamlit secrets are unavailable in local scripts unless configured.
        return default

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "policies"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EVALUATION_DIR = BASE_DIR / "evaluation"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"

# Embedding Model
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Chunking Configuration
CHUNK_SIZE = 900          # Characters target per chunk (page/section aligned)
CHUNK_OVERLAP = 150       # Overlap between contiguous chunks within same section/page

# Retrieval Configuration
TOP_K = 5
SCORE_THRESHOLD = 0.40    # FAISS inner-product / cosine similarity threshold

# LLM Configuration
# Can be configured via environment variables or Streamlit secrets:
# Providers supported by rag.py: "groq", "openai", "ollama"
DEFAULT_LLM_PROVIDER = _setting("LLM_PROVIDER", "groq")
GROQ_API_KEY = _setting("GROQ_API_KEY")
GROQ_MODEL = _setting("GROQ_MODEL", "qwen/qwen3.8-27b")

OPENAI_API_KEY = _setting("OPENAI_API_KEY")
OPENAI_MODEL = _setting("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = _setting("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _setting("OLLAMA_MODEL", "llama3.2")
