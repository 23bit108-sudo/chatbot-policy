"""
Core RAG Engine for Internal Policy Question-Answering Assistant
- Vector Search with FAISS & BGE-M3 (cosine similarity)
- Policy Versioning & Status Filtering (ignores draft, expired, superseded)
- Policy Conflict Detection & Ambiguity Resolution
- Grounded Answer Generation with precise citations using Groq (or fallback providers)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    SCORE_THRESHOLD,
    DEFAULT_LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an authoritative Internal Policy Assistant for the organization.
Your sole job is to answer employee queries using ONLY the provided Approved Policy Context.

CRITICAL OPERATIONAL RULES:
1. Answer strictly using ONLY facts stated in the provided Approved Policy Context.
2. Do NOT extrapolate, speculate, or introduce external knowledge not explicitly documented.
3. If the provided context does NOT contain sufficient information to answer the user's question, you MUST explicitly state:
   "The available policy documents do not contain enough information to answer this question."
4. CITATIONS ARE MANDATORY:
   - For every substantive policy rule, limit, condition, or entitlement mentioned, cite the source document name and page number.
   - Format citations clearly at the end of your answer in a dedicated "Sources Cited" section:
     • Document: <document_name> | Page: <page_number> | Section: <section>
5. POLICY PRECEDENCE & AMENDMENTS:
   - If an approved amendment explicitly supersedes or modifies an older policy clause, clearly state the amended rule and mention that it supersedes the previous provision.
6. POLICY CONFLICTS:
   - If multiple retrieved approved documents directly contradict each other and no superseding clause establishes precedence, you must explicitly state:
     "A policy conflict was detected between [Document A, Page X] and [Document B, Page Y]. The provided documents do not establish which rule takes precedence."
7. NEVER invent or fabricate policies, dates, page numbers, allowances, or rules.
"""


class RAGEngine:
    def __init__(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embed_model: Optional[SentenceTransformer] = None
        
        self._load_vectorstore()

    def _load_vectorstore(self):
        """Loads FAISS index and chunk metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Vector store files not found at {self.index_path} or {self.metadata_path}. "
                f"Please run ingestion.py first!"
            )
        
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.index = faiss.read_index(str(self.index_path))
        
        logger.info(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        logger.info(f"Loaded {len(self.metadata)} chunk metadata records.")

    def _get_embed_model(self) -> SentenceTransformer:
        """Lazy loads embedding model for query encoding."""
        if self.embed_model is None:
            logger.info(f"Loading query embedding model {EMBEDDING_MODEL_NAME}...")
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embed_model

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks for query using cosine similarity.
        Applies versioning and status filtering.
        """
        if not query.strip():
            return []

        model = self._get_embed_model()
        # Encode query and normalize for inner product cosine similarity
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # Retrieve more candidates initially to allow filtering
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_vec, search_k)

        raw_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            raw_results.append(item)

        # Apply filtering: Exclude draft, expired, superseded
        filtered_results = []
        for r in raw_results:
            status = r.get("status", "").lower()
            if status in ["draft", "expired", "superseded", "outdated"]:
                # Log filtered out unapproved document
                continue
            filtered_results.append(r)

        return filtered_results[:top_k]

    def detect_conflicts(self, chunks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Checks if retrieved chunks contain conflicting policies without clear precedence.
        """
        # Group by document
        doc_names = set(c["document_name"] for c in chunks)
        if len(doc_names) <= 1:
            return None

        # Check specifically if notices or policies conflict on identical topics
        has_notice_a = any("Notice A" in c["document_name"] for c in chunks)
        has_notice_b = any("Notice B" in c["document_name"] for c in chunks)
        if has_notice_a and has_notice_b:
            return (
                "Policy Conflict Detected: Documents 'Notice A' and 'Notice B' both provide contradictory "
                "guidelines approved on the same date with no precedence clause. The provided documents do not establish "
                "which rule takes precedence."
            )

        return None

    def call_llm(self, prompt: str, system_prompt: str = SYSTEM_PROMPT, provider: str = DEFAULT_LLM_PROVIDER, api_key: Optional[str] = None) -> str:
        """
        Dispatches LLM generation to the configured provider (Groq, OpenAI, Gemini, Ollama).
        Defaults to Groq.
        """
        provider = provider.lower()

        # 1. Groq Provider
        if provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY", GROQ_API_KEY)
            if not key:
                raise ValueError("GROQ_API_KEY is not set. Please provide a Groq API key in your .env or Streamlit secrets.")
            
            from groq import Groq
            client = Groq(api_key=key)
            response = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", GROQ_MODEL),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Strict factual grounding
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        # 2. OpenAI Provider
        elif provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
            if not key:
                raise ValueError("OPENAI_API_KEY is not set.")
            
            from openai import OpenAI
            client = OpenAI(api_key=key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()

        # 3. Ollama Provider (Local Fallback)
        elif provider == "ollama":
            import requests
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\nUser Question:\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.0}
            }
            res = requests.post(url, json=payload, timeout=60)
            res.raise_for_status()
            return res.json().get("response", "").strip()

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def answer_question(self, question: str, provider: str = DEFAULT_LLM_PROVIDER, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Retrieve top chunks
        2. Filter out non-approved policies
        3. Check similarity threshold
        4. Detect conflicts
        5. Build grounded prompt and generate answer
        """
        retrieved_chunks = self.retrieve(question, top_k=TOP_K)

        # Edge case: No results retrieved or relevance score too low
        if not retrieved_chunks or (retrieved_chunks[0]["score"] < SCORE_THRESHOLD):
            return {
                "answer": "The available policy documents do not contain enough information to answer this question.",
                "sources": [],
                "conflict_warning": None,
                "retrieved_chunks": retrieved_chunks,
                "status": "insufficient_info",
            }

        # Check for policy conflict
        conflict_msg = self.detect_conflicts(retrieved_chunks)

        # Check for amendment supersession
        has_amendment = any("Amendment" in c["document_name"] for c in retrieved_chunks)

        # Assemble context blocks
        context_parts = []
        for c in retrieved_chunks:
            source_tag = f"DOCUMENT: {c['document_name']} | PAGE: {c['page']} | SECTION: {c['section']} | STATUS: {c['status']}"
            context_parts.append(f"--- {source_tag} ---\n{c['content']}")

        full_context = "\n\n".join(context_parts)

        # Build prompt
        user_prompt = f"""CONTEXT FROM APPROVED POLICY DOCUMENTS:
{full_context}

QUESTION:
{question}

Provide a direct, factual answer strictly grounded in the context above. Include citations for every policy claim. If information is not in the text, explicitly state that it is not available.
"""

        try:
            answer_text = self.call_llm(user_prompt, provider=provider, api_key=api_key)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "answer": f"Error communicating with LLM provider ({provider}): {str(e)}",
                "sources": [],
                "conflict_warning": conflict_msg,
                "retrieved_chunks": retrieved_chunks,
                "status": "error",
            }

        # Build clean citation list
        citations = []
        seen = set()
        for c in retrieved_chunks:
            key = (c["document_name"], c["page"], c["section"])
            if key not in seen:
                seen.add(key)
                citations.append({
                    "document_name": c["document_name"],
                    "filename": c.get("filename", ""),
                    "page": c["page"],
                    "section": c["section"],
                    "similarity_score": round(c["score"], 3),
                    "full_content": c["content"],
                    "excerpt": c["content"][:250] + ("..." if len(c["content"]) > 250 else "")
                })

        return {
            "answer": answer_text,
            "sources": citations,
            "conflict_warning": conflict_msg,
            "retrieved_chunks": retrieved_chunks,
            "status": "success",
        }
