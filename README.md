# Internal Policy Assistant

A small Streamlit RAG application for asking questions about the policy PDFs in `data/policies/`. It retrieves policy passages with BGE-M3 and FAISS, then asks a configured LLM to answer from those passages with document, page, and section citations.

## What it does

- Extracts text from PDFs page by page and creates page-aware chunks.
- Stores chunk text and document metadata beside a FAISS vector index.
- Filters retrieved chunks whose detected status is draft, expired, superseded, or outdated.
- Shows answers, citations, retrieved excerpts, and the original PDF page text.
- Lets a user add a PDF or remove a document and its indexed chunks.
- Includes a small JSON evaluation set for answerable, unanswerable, amended-policy, and conflicting-policy questions.

This is a portfolio prototype. See [ARCHITECTURE.md](ARCHITECTURE.md) for the implemented pipeline and its current limitations.

## Project files

```text
app.py                    Streamlit UI, upload/removal controls, page previews
rag.py                    Retrieval, status filtering, conflict check, LLM call
ingestion.py              PDF extraction, chunking, embedding, FAISS persistence
config.py                 Paths, model and retrieval settings
data/policies/            Source policy PDFs
vectorstore/              FAISS index and aligned chunk metadata
evaluation/                Evaluation examples, runner, and generated report
.streamlit/config.toml     Streamlit theme and server settings
.env.example               Local environment-variable template
```

## Run locally

Requirements: Python 3.10 or newer. Install dependencies in a virtual environment:

```bash
git clone <repository-url>
cd <repository-directory>
python -m venv .venv
```

Activate the environment (`.venv\\Scripts\\activate` in PowerShell, or `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add a key for the selected cloud provider, and leave `.env` untracked:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

Supported providers in the current code are `groq`, `openai`, and `ollama`. Ollama requires a locally running Ollama server. Google Gemini settings are not currently implemented in the RAG provider code.

The checked-in FAISS index and metadata can be used as-is. To rebuild them from all PDFs in `data/policies/`:

```bash
python ingestion.py
```

Then start the app:

```bash
streamlit run app.py
```

The first run downloads the BGE-M3 model if it is not already cached. An LLM API key is required for Groq or OpenAI answers.

## Evaluation

Run the lightweight evaluation script with:

```bash
python evaluation/evaluate.py
```

It makes live LLM calls using the configured provider and writes `evaluation/evaluation_report.json`. The current checks are keyword-based answer checks, a simple conflict/abstention check, and expected-document matching; the expected page and section fields in the dataset are descriptive and are not currently asserted by the runner. Treat the report as a smoke check, not a comprehensive RAG benchmark.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. Add the provider and credentials under the app's **Secrets** settings using TOML, for example:

   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-key-here"
   GROQ_MODEL = "qwen/qwen3.8-27b"
   ```

   Alternatively, configure equivalent environment variables where supported.

The prebuilt vectorstore is included, so deployment does not need to run ingestion. The embedding model is downloaded at runtime. Files added or removed through the UI are stored on the app's local filesystem; Community Cloud does not provide durable storage for those runtime changes across rebuilds or restarts. Rebuild and push the vectorstore and PDFs to make repository changes durable.

## Before publishing

- Never commit `.env`, API keys, or `.streamlit/secrets.toml`; `.gitignore` excludes them.
- Confirm you have permission to publish every PDF under `data/policies/` and the generated vectorstore, which contains extracted policy text in `metadata.json`.
- Review `evaluation/evaluation_report.json` before publishing if it contains outputs you do not want public.

