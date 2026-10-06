"""
Ingestion Pipeline for Internal Policy Question-Answering Assistant
- Extracts text page-by-page from PDFs using PyMuPDF (fitz)
- Detects chapters, sections, and document metadata (status, version, dates)
- Splits into section-aware, page-preserved chunks
- Computes dense embeddings using BGE-M3 (BAAI/bge-m3)
- Stores vectors in FAISS index (IndexFlatIP with normalized vectors for cosine similarity)
- Stores metadata in JSON format alongside the FAISS index
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR,
    INDEX_PATH,
    METADATA_PATH,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_document_header_info(first_pages_text: str, filename: str) -> Dict[str, Any]:
    """
    Extracts policy version, status, and effective date from document header / cover.
    Detects if marked as draft, expired, superseded, or approved.
    """
    text_lower = first_pages_text.lower()
    
    # Status detection
    if "superseded" in text_lower or "expired" in text_lower or "outdated" in text_lower:
        status = "expired"
    elif "draft" in text_lower or "unapproved" in text_lower:
        status = "draft"
    elif "approved" in text_lower or "manual" in text_lower or "official" in text_lower:
        status = "approved"
    else:
        status = "unknown"

    # Year / Version detection
    version_match = re.search(r"\b(20\d\d)\b", first_pages_text)
    version = version_match.group(1) if version_match else "Unknown"

    # Effective Date detection
    effective_match = re.search(r"effective(?:\s+date)?[:\s]+([0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})", first_pages_text, re.IGNORECASE)
    effective_date = effective_match.group(1) if effective_match else None

    # Clean display name from filename
    clean_name = filename.replace(".pdf", "").replace("_", " ")

    return {
        "document_name": clean_name,
        "filename": filename,
        "policy_version": version,
        "status": status,
        "effective_date": effective_date,
    }


def detect_section_title(text: str, current_section: str) -> str:
    """
    Identifies chapter or section headers in the page text to preserve context.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines[:8]:
        # Chapter pattern e.g. "CHAPTER 6 LEAVE AND ATTENDANCE"
        if re.match(r"^(?:CHAPTER|Chapter)\s+\d+", line):
            return line
        # Section pattern e.g. "Section: Casual Leave Amendment" or "5.1 LEAVE TYPE 1: CASUAL LEAVE"
        if re.match(r"^(?:Section[:\s]|\d+\.\d+\s+[A-Z\s]{3,})", line, re.IGNORECASE):
            return line
        # All caps heading of reasonable length
        if line.isupper() and 4 < len(line) < 60 and not line.startswith("IIMA"):
            return line
            
    return current_section


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Page-aware text chunking. Breaks text on paragraph/newline boundaries rather than arbitrary slices.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if not current_chunk:
            current_chunk = para
        elif len(current_chunk) + len(para) + 1 <= chunk_size:
            current_chunk += "\n\n" + para
        else:
            chunks.append(current_chunk)
            # Retain overlap from end of current chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                overlap_text = current_chunk[-chunk_overlap:]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def process_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF, detects section headers, and generates chunks with full metadata.
    """
    doc = fitz.open(str(pdf_path))
    filename = pdf_path.name
    
    # Read first 3 pages to extract doc-level metadata (version, status, dates)
    header_sample = ""
    for p_num in range(min(3, len(doc))):
        header_sample += doc[p_num].get_text() + "\n"
        
    doc_meta = extract_document_header_info(header_sample, filename)
    logger.info(f"Processing '{filename}' | Status: {doc_meta['status']} | Version: {doc_meta['policy_version']} | Pages: {len(doc)}")

    chunks_data = []
    current_section = "General"

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1  # 1-indexed for citations
        page_text = page.get_text()

        # Clean noise/excess whitespace
        page_text_clean = re.sub(r"[ \t]+", " ", page_text).strip()
        if not page_text_clean or len(page_text_clean) < 30:
            continue  # Skip blank pages or decorative title pages

        # Track section title
        current_section = detect_section_title(page_text_clean, current_section)

        # Generate chunks for this page
        page_chunks = chunk_text(page_text_clean)

        for chunk_idx, chunk_str in enumerate(page_chunks):
            # Prepend contextual header to assist embedding retrieval
            context_header = f"[{doc_meta['document_name']} | Page {page_num} | Section: {current_section}]\n"
            searchable_text = context_header + chunk_str

            chunk_entry = {
                "chunk_id": f"{filename}_p{page_num}_c{chunk_idx}",
                "text": searchable_text,
                "content": chunk_str,
                "document_name": doc_meta["document_name"],
                "filename": filename,
                "page": page_num,
                "section": current_section,
                "policy_version": doc_meta["policy_version"],
                "status": doc_meta["status"],
                "effective_date": doc_meta["effective_date"],
            }
            chunks_data.append(chunk_entry)

    doc.close()
    return chunks_data


def build_index(chunks: List[Dict[str, Any]], model_name: str = EMBEDDING_MODEL_NAME):
    """
    Generates BGE-M3 embeddings, normalizes them, and builds a FAISS IndexFlatIP (Cosine Similarity).
    """
    logger.info(f"Loading embedding model: {model_name}...")
    embed_model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info(f"Computing embeddings for {len(texts)} chunks...")
    
    # Compute embeddings in batches
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}. Building FAISS IndexFlatIP...")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"Saved FAISS index to {INDEX_PATH}")

    # Save Metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata ({len(chunks)} entries) to {METADATA_PATH}")


def ingest_new_pdf(pdf_path: Path):
    """
    Ingests a newly added PDF and appends its chunks & embeddings into the existing FAISS index.
    """
    logger.info(f"Ingesting new PDF document: {pdf_path}")
    new_chunks = process_pdf(pdf_path)
    if not new_chunks:
        logger.warning(f"No chunks extracted from {pdf_path}")
        return len(new_chunks)

    # Load existing metadata if available
    existing_chunks = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)

    # Generate embeddings for new chunks
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in new_chunks]
    new_embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    new_embeddings = np.array(new_embeddings, dtype=np.float32)

    # Load or create index
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(new_embeddings.shape[1])

    index.add(new_embeddings)

    # Save updated FAISS index & metadata
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    
    all_chunks = existing_chunks + new_chunks
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully added {len(new_chunks)} new chunks. Total indexed chunks: {len(all_chunks)}")
    return len(new_chunks)


def run_ingestion():
    """Build a fresh vectorstore from every PDF in the policy directory."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No policy PDFs found in {DATA_DIR}")

    all_chunks = []
    for pdf_path in pdf_paths:
        logger.info(f"Extracting policy document: {pdf_path.name}")
        all_chunks.extend(process_pdf(pdf_path))

    if not all_chunks:
        raise ValueError("No searchable text was extracted from the policy PDFs.")

    build_index(all_chunks)
    logger.info(f"Ingestion complete: {len(pdf_paths)} PDFs, {len(all_chunks)} chunks.")
    return len(all_chunks)


if __name__ == "__main__":
    run_ingestion()
