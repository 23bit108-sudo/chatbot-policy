You are a senior Python/AI engineer helping me build a small but production-quality portfolio project.

PROJECT:
Internal Policy Question-Answering Assistant

GOAL:
Build a cloud-hosted RAG application that allows employees to ask questions about provided company/HR policy documents and receive accurate, grounded answers with document, page, and section citations.

This is both an assignment and a portfolio project. The final result should be a clean, working, publicly deployable Streamlit application that demonstrates strong understanding of RAG, retrieval, embeddings, vector search, document metadata, policy versioning, conflict handling, hallucination prevention, and evaluation.

IMPORTANT:
Keep the codebase SIMPLE AND SWEET.

Do NOT overengineer the project.
Prefer a small number of readable Python files over complicated abstractions.
Do not introduce frameworks or libraries unless they provide clear value.
Avoid unnecessary classes, interfaces, design patterns, dependency injection, microservices, background workers, databases, Docker, Kubernetes, LangChain, LlamaIndex, etc.

The project should be easy for a student/interviewer to understand and explain line-by-line.

==================================================
TECH STACK
==================================================

Use:

- Python
- Streamlit for the web application and deployment
- FAISS for vector search
- BGE-M3 for embeddings
- An LLM accessible from the cloud for final answer generation
- PyMuPDF for PDF extraction
- GitHub for source control
- Streamlit Community Cloud for deployment

Use Ollama only as an optional local-development LLM if useful.

IMPORTANT:
The deployed application must NOT depend on Ollama running on my laptop.

The architecture must allow the LLM provider to be changed through configuration/environment variables without changing the RAG logic.

Do not build a separate FastAPI service initially unless there is a concrete reason to do so. The primary deliverable is the Streamlit application.

==================================================
HIGH-LEVEL ARCHITECTURE
==================================================

The application should follow:

PDF policy documents
        ↓
PDF extraction
        ↓
page/section-aware chunking
        ↓
metadata extraction
        ↓
BGE-M3 embeddings
        ↓
FAISS index
        ↓
user question
        ↓
query embedding
        ↓
FAISS retrieval
        ↓
policy/version filtering
        ↓
relevant context
        ↓
LLM
        ↓
grounded answer + citations
        ↓
Streamlit UI

Keep this architecture explicit and easy to follow.

==================================================
RAG REQUIREMENTS
==================================================

The assistant MUST answer questions using ONLY the supplied policy documents.

It must NOT use general world knowledge to fill gaps.

If the retrieved documents do not contain enough information to answer the question, the assistant must clearly say that the available policy documents do not provide enough information.

Never fabricate:

- policies
- rules
- dates
- page numbers
- section names
- document names
- policy versions
- citations
- employee entitlements
- approval status

If information is partially available, answer only the supported part and explicitly state what is missing.

==================================================
DOCUMENT METADATA
==================================================

Every chunk should preserve useful metadata such as:

- document_name
- page_number
- section
- policy_version
- status
- effective_date
- approval_date
- source_type

Do not discard page information during chunking.

Page-level citation is mandatory.

Where possible, section headings should also be preserved.

Example metadata:

{
    "document_name": "IIMA HR Policy Manual 2024",
    "page": 32,
    "section": "Leave Policy",
    "version": "2024",
    "status": "approved",
    "effective_date": "2024-01-01"
}

==================================================
POLICY VERSIONING
==================================================

The system must prefer the latest APPROVED policy when multiple versions exist.

Ignore documents/chunks that are clearly:

- draft
- expired
- superseded
- obsolete
- withdrawn
- deprecated

Do NOT assume a document is current merely because its filename contains a newer-looking year.

Use explicit metadata/content where available.

If approval/current status cannot be established confidently, do not silently choose a version.

Instead explain the ambiguity.

==================================================
CONFLICT HANDLING
==================================================

If multiple approved documents contain contradictory information:

1. Detect the conflict.
2. Identify the relevant documents.
3. Check whether one explicitly supersedes or amends another.
4. Prefer the latest approved policy if that precedence is supported.
5. If precedence cannot be established, clearly state that the documents conflict and that the provided material does not establish which rule should prevail.

Never silently merge contradictory policies.

Example:

"Two approved documents provide different leave limits. The 2024 Leave Amendment explicitly supersedes the older policy, so the 2024 rule is used."

If there is no evidence of precedence:

"The available approved documents contain conflicting information regarding this policy. I cannot determine which rule takes precedence from the provided documents."

==================================================
CITATIONS
==================================================

Every factual policy answer must include citations.

Citations should contain:

- document name
- page number
- section when available

Example:

Source:
IIMA HR Policy Manual 2024
Page 32
Section: Leave Policy

If multiple chunks support the answer, cite all relevant sources.

Never invent citation information.

The UI should make sources clearly visible below the answer.

==================================================
RETRIEVAL
==================================================

Use FAISS for vector similarity search.

Use BGE-M3 to create embeddings.

Keep retrieval implementation straightforward.

Prefer:

query
→ embedding
→ FAISS similarity search
→ metadata lookup
→ filtering/ranking
→ context

Do not introduce a vector database server.

Do not use ChromaDB.

Do not use LangChain unless there is an unavoidable reason.

The purpose of this project is to demonstrate that I understand how RAG works rather than hiding the entire pipeline behind a framework.

==================================================
CHUNKING
==================================================

Use simple section-aware and page-aware chunking.

Preserve:

- page boundaries
- section headings
- meaningful paragraphs
- policy clauses

Avoid blindly splitting every N characters if that destroys policy meaning.

A reasonable chunk should contain enough surrounding context to answer a question while remaining retrieval-friendly.

Keep chunking logic simple and explainable.

==================================================
LLM BEHAVIOR
==================================================

The LLM is an ANSWER GENERATOR, not the source of truth.

The retrieved policy context is the only source of truth.

The LLM must:

- answer only from supplied context
- avoid hallucination
- acknowledge insufficient information
- respect policy version filtering
- respect conflicts
- provide citations
- not add external knowledge

The system prompt sent to the LLM should strongly enforce these rules.

==================================================
STREAMLIT APPLICATION
==================================================

Build a clean, professional UI.

The application should contain:

1. Application title
   "Internal Policy Assistant"

2. Short description explaining:
   "Ask questions about the provided approved policy documents."

3. Question input/chat interface.

4. Answer section.

5. Sources section.

6. Retrieval/source information where useful.

7. Clear warning when:
   - information is insufficient
   - policy conflict is detected
   - policy version is ambiguous

Example UI:

--------------------------------------------------
Internal Policy Assistant

Ask a question about the available policies.

[ How many casual leaves are employees entitled to? ]

ANSWER
Employees are entitled to ...

SOURCES
IIMA HR Policy Manual 2024
Section: Leave Policy
Page: 32

--------------------------------------------------

For conflicts:

⚠ Policy Conflict Detected

Document A — Page 12
Document B — Page 4

The available documents do not establish which
policy takes precedence.

--------------------------------------------------

For unanswerable questions:

Information Not Found

The provided approved policy documents do not
contain enough information to answer this question.
--------------------------------------------------

The UI should look polished but should not contain excessive visual complexity.

==================================================
CLOUD DEPLOYMENT
==================================================

The primary objective is CLOUD DEPLOYMENT.

The application must work independently of my laptop after deployment.

Target deployment:

Streamlit Community Cloud

Repository:

GitHub

The project should include:

requirements.txt
README.md
.streamlit/config.toml if actually needed
.env.example if environment variables are required

Never commit API keys or secrets.

Use Streamlit secrets/environment variables for deployed credentials.

The README must contain clear deployment instructions.

==================================================
PROJECT STRUCTURE
==================================================

Keep the structure small.

Prefer something approximately like:

policy-qa/
│
├── app.py
├── rag.py
├── ingestion.py
├── config.py
│
├── data/
│   └── policies/
│
├── vectorstore/
│   ├── index.faiss
│   └── metadata.json
│
├── evaluation/
│   └── evaluation.json
│
├── requirements.txt
├── README.md
└── .gitignore

Do NOT create dozens of tiny files.

If two pieces of logic naturally belong together, keep them together.

The exact structure can be adjusted if there is a strong reason.

==================================================
INGESTION
==================================================

Create a simple ingestion process that:

1. Reads PDFs.
2. Extracts text page-by-page.
3. Detects/preserves section information where possible.
4. Creates chunks.
5. Creates metadata.
6. Generates BGE-M3 embeddings.
7. Builds the FAISS index.
8. Saves:
   - FAISS index
   - metadata

The application should load the existing index rather than rebuilding embeddings every time Streamlit starts.

Make ingestion a separate script/process.

For example:

python ingestion.py

Then:

streamlit run app.py

==================================================
EVALUATION
==================================================

Create a small evaluation dataset containing at least:

- answerable questions
- unanswerable questions
- conflicting-policy questions
- versioning questions

For each question, define the expected behavior.

Example:

{
    "question": "...",
    "type": "answerable",
    "expected_source": "...",
    "expected_page": 32
}

Evaluation should consider:

1. Retrieval correctness
2. Answer correctness
3. Citation correctness
4. Abstention correctness
5. Conflict detection
6. Version-selection correctness

Keep evaluation simple and transparent.

Do not build a complicated evaluation framework.

==================================================
ERROR HANDLING
==================================================

Handle common failures gracefully:

- missing FAISS index
- missing metadata
- missing API key
- empty question
- no relevant retrievals
- malformed PDF
- LLM API failure

Show useful user-facing messages rather than Python tracebacks.

==================================================
README
==================================================

The README should explain:

1. Problem statement
2. Features
3. Architecture
4. RAG pipeline
5. Why FAISS
6. Why BGE-M3
7. LLM choice
8. Chunking strategy
9. Policy version handling
10. Conflict handling
11. Evaluation approach
12. Local setup
13. Ingestion
14. Running the Streamlit app
15. Cloud deployment
16. Limitations
17. Example questions

Keep the README concise and professional.

==================================================
ENGINEERING PRINCIPLES
==================================================

Follow these principles throughout the project:

1. Simplicity over abstraction.
2. Readability over cleverness.
3. Explicit RAG pipeline over black-box frameworks.
4. Correctness over unnecessary features.
5. Grounded answers over fluent hallucinations.
6. Small codebase over enterprise-style architecture.
7. Reproducible setup.
8. Clear metadata and citations.
9. Cloud deployment from the beginning.
10. Every dependency must have a reason.

Before adding a library, ask:

"Can this be implemented simply with the Python standard library or a dependency we already have?"

If yes, prefer the simpler solution.

==================================================
IMPORTANT DEVELOPMENT RULE
==================================================

Do NOT generate the entire project in one giant response.

Build incrementally.

At each stage:

1. Explain what we are building.
2. Show the files being created/modified.
3. Provide complete code for those files.
4. Explain how to run/test it.
5. Wait for my confirmation/results before moving to the next stage.

When debugging, modify the minimum amount of code necessary.

Do not rewrite working code unnecessarily.

==================================================
FINAL QUALITY BAR
==================================================

The final project should feel like:

"A small, well-engineered, cloud-deployed RAG application built by someone who understands retrieval, embeddings, vector search, LLM grounding, document versioning and evaluation."

It should NOT feel like:

"A huge AI-generated project with 30 files and unnecessary frameworks."

The priority is:

WORKING > SIMPLE > CORRECT > DEPLOYABLE > POLISHED

Build accordingly.