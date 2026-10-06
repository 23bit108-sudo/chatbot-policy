# Architecture

## Overview

The project keeps the retrieval pipeline explicit across a few Python files. PDF ingestion is an offline/update operation; Streamlit loads the saved vectorstore for interactive questions.

```text
                           BUILD / UPDATE
data/policies/*.pdf
        │
        ▼
ingestion.py ── PyMuPDF extraction by page
        │
        ├── section-aware, page-preserving chunks + metadata
        │      (document name, filename, page, section, status, version, date)
        ▼
BGE-M3 embeddings (normalized)
        │
        ├── vectorstore/index.faiss
        └── vectorstore/metadata.json  (same row order as FAISS vectors)

                           QUERY
Question ──► BGE-M3 query embedding ──► FAISS inner-product search (cosine)
                                                  │
                                                  ▼
                                  status filtering and Top-K selection
                                                  │
                                                  ▼
                                  retrieved policy context + question
                                                  │
                                                  ▼
                                  configured provider: Groq, OpenAI,
                                  or local Ollama
                                                  │
                                                  ▼
                                  Streamlit answer and citations
```

## Main components

- **`ingestion.py`** reads each PDF page with PyMuPDF, identifies likely headings, and chunks text within each page. It embeds chunk text with `BAAI/bge-m3`, normalizes vectors, and writes an `IndexFlatIP` FAISS index. Each metadata record is stored at the same position as its vector; this ordering is required for citations and document removal.
- **`rag.py`** loads the index and metadata, embeds each question, retrieves the nearest passages, filters by the detected status field, checks the retrieval threshold, constructs a prompt from retrieved context, calls the configured LLM, and formats source metadata.
- **`config.py`** defines project-relative data and vectorstore paths, embedding/retrieval settings, provider defaults, and credentials read from environment variables, `.env`, or Streamlit secrets.
- **`app.py`** provides the Streamlit interface, PDF upload and removal, question entry, answer rendering, citations, and full-page text lookup.

## Chunk and metadata model

Chunking runs separately for each PDF page, so a chunk does not cross page boundaries. It groups paragraphs up to the configured character target (`CHUNK_SIZE`, default 900), retaining a trailing overlap (`CHUNK_OVERLAP`, default 150) when moving to the next chunk on the same page. Each record retains the PDF filename, display name, one-based page number, detected section, version, status, and effective date when found.

Status detection is a lightweight text scan of the first three pages. It is useful for this sample corpus but is not a robust document approval workflow. A reviewer should confirm status metadata before relying on the filtering behavior for real policies.

## Retrieval and answer generation

Questions and policy chunks use the same BGE-M3 embedding model. The normalized vectors are searched with FAISS inner product, which yields cosine similarity for normalized vectors. The engine retrieves up to three times the configured `TOP_K` candidates (bounded by index size), removes chunks with statuses `draft`, `expired`, `superseded`, or `outdated`, and returns up to `TOP_K` remaining results. If the top result is below `SCORE_THRESHOLD`, the app abstains.

The LLM receives only the retrieved policy context and the question, along with instructions to answer from that context and cite sources. Supported providers in `rag.py` are Groq, OpenAI, and Ollama. The model does not perform retrieval and should not be treated as the policy source of truth.

## Precedence and conflicts

The current implementation filters known non-current statuses and includes amendment guidance in the answer-generation prompt. It does not implement a general policy-version graph or automatically prove that one arbitrary policy supersedes another. Conflict detection is currently a narrow check for the two included gym timing notices. These rules should be expanded and covered by tests before using the system as an operational HR authority.

## Document updates and persistence

Adding a PDF extracts and appends its chunks and vectors. Removing a document removes its vector rows and corresponding metadata records, then deletes the matching local PDF. The index and metadata must remain aligned; the removal handler checks their counts before writing. Streamlit Community Cloud's local filesystem is not durable across all restarts/rebuilds, so UI-based document changes should be committed back into the repository's source PDFs and vectorstore when they need to persist.

## Evaluation

`evaluation/evaluation.json` provides a small set of examples spanning answerable, unanswerable, amended, and conflicting policy queries. `evaluation/evaluate.py` calls the live configured LLM and checks expected answer keywords, abstention/conflict behavior, and expected document names. The current runner does not assert expected page or section values; results are a lightweight smoke check rather than a statistically meaningful benchmark.
