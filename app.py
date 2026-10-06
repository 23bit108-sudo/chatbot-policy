"""
Streamlit Web Application: Internal Policy Question-Answering Assistant
A clean, grounded RAG application for employee policy Q&A with strict citations,
versioning awareness, conflict detection, PDF upload/ingestion, and clickable citations.
"""

import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

load_dotenv()

from rag import RAGEngine
from ingestion import ingest_new_pdf
import config

# Streamlit Page Configuration
st.set_page_config(
    page_title="Internal Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
        font-size: 0.95rem;
    }
    .conflict-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        margin-bottom: 15px;
        border-radius: 6px;
        color: #92400E;
    }
    .unanswerable-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 14px 18px;
        margin-bottom: 15px;
        border-radius: 6px;
        color: #1E40AF;
    }
    .page-preview-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.88rem;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Policy RAG Engine & FAISS Index...")
def get_rag_engine():
    """Initializes and caches the RAG Engine across user interactions."""
    return RAGEngine()


def extract_original_page_text(filename: str, page_number: int, document_name: str = "") -> str:
    """Extracts raw text from the original PDF file for full inspection."""
    # Older vector stores may omit filename, or contain an absolute/relative
    # path from another machine. Resolve only by the PDF's basename in DATA_DIR.
    pdf_name = Path(filename).name if filename else ""
    pdf_path = config.DATA_DIR / pdf_name if pdf_name else None
    if not pdf_path or not pdf_path.is_file():
        search_terms = [term for term in (pdf_name, filename, document_name) if term]
        matching = []
        for term in search_terms:
            matching = list(config.DATA_DIR.glob(f"*{Path(term).stem}*.pdf"))
            if matching:
                break
        if not matching:
            return "Original PDF document not found on disk."
        pdf_path = matching[0]
    try:
        doc = fitz.open(str(pdf_path))
        if 1 <= page_number <= len(doc):
            text = doc[page_number - 1].get_text()
            doc.close()
            return text if text.strip() else "[No selectable text on this page or scanned image]"
        doc.close()
        return f"Page {page_number} is out of range (Total pages: {len(doc)})."
    except Exception as e:
        return f"Error reading page {page_number}: {e}"


def set_search_query(q_text: str):
    """Callback when clicking a sample question so it populates the search bar."""
    st.session_state["query_input"] = q_text
    st.session_state["main_query_field"] = q_text


def remove_policy_document(document_name: str, metadata: list) -> int:
    """Remove a policy's vectors, metadata, and local PDF from the repository."""
    matching_rows = [i for i, item in enumerate(metadata) if item.get("document_name") == document_name]
    if not matching_rows:
        raise ValueError(f"No indexed chunks found for '{document_name}'.")

    index = faiss.read_index(str(config.INDEX_PATH))
    if index.ntotal != len(metadata):
        raise ValueError("The FAISS index and metadata are out of sync; removal was cancelled.")

    filenames = {
        Path(item.get("filename", "")).name
        for item in metadata
        if item.get("document_name") == document_name and item.get("filename")
    }
    keep_rows = [i for i in range(len(metadata)) if i not in set(matching_rows)]
    index.remove_ids(np.ascontiguousarray(matching_rows, dtype=np.int64))

    # Write both updated store files before replacing their current versions.
    index_tmp = config.INDEX_PATH.with_suffix(".faiss.tmp")
    metadata_tmp = config.METADATA_PATH.with_suffix(".json.tmp")
    try:
        faiss.write_index(index, str(index_tmp))
        with open(metadata_tmp, "w", encoding="utf-8") as f:
            json.dump([metadata[i] for i in keep_rows], f, indent=2, ensure_ascii=False)
        os.replace(index_tmp, config.INDEX_PATH)
        os.replace(metadata_tmp, config.METADATA_PATH)
    finally:
        for temp_path in (index_tmp, metadata_tmp):
            if temp_path.exists():
                temp_path.unlink()

    for filename in filenames:
        pdf_path = config.DATA_DIR / filename
        if pdf_path.is_file():
            pdf_path.unlink()
    return len(matching_rows)


def main():
    # Initialize session state for query
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = ""

    engine = get_rag_engine()

    # Sidebar
    with st.sidebar:
        
        st.title("Policy Documents")
        
        provider = config.DEFAULT_LLM_PROVIDER

        st.markdown("---")
        st.markdown("## Add New Policy Document")
        uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Ingest Document", type="secondary", use_container_width=True):
                save_dest = config.DATA_DIR / uploaded_file.name
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner(f"Processing and indexing {uploaded_file.name}..."):
                    num_chunks = ingest_new_pdf(save_dest)
                    # Clear cache to reload updated FAISS index
                    st.cache_resource.clear()
                    st.success(f"Added '{uploaded_file.name}' ({num_chunks} chunks indexed)!")
                    st.rerun()

        st.markdown("---")
        st.markdown("## Policy Repository")
        
        # Display indexed policy documents
        try:
            doc_names = sorted(list(set(c["document_name"] for c in engine.metadata)))
            st.success(f"**{len(doc_names)} Documents Active**")
            for doc in doc_names:
                st.caption(f"• {doc}")
            if doc_names:
                with st.expander("Remove a policy document"):
                    selected_doc = st.selectbox(
                        "Choose a document", doc_names, key="remove_policy_selection"
                    )
                    st.caption("This removes its PDF and indexed passages from the repository.")
                    confirmed = st.checkbox(
                        f"Confirm removal of {selected_doc}", key=f"confirm_remove_{selected_doc}"
                    )
                    if st.button("Remove document", type="secondary", disabled=not confirmed):
                        try:
                            removed_chunks = remove_policy_document(selected_doc, engine.metadata)
                            st.cache_resource.clear()
                            st.session_state.pop("latest_result", None)
                            st.session_state.pop("last_searched_query", None)
                            st.success(f"Removed '{selected_doc}' and {removed_chunks} indexed passages.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not remove '{selected_doc}': {e}")
        except Exception as e:
            st.error(f"Vector store loading error: {e}")

    # Main Area
    st.markdown('<div class="main-header">Internal Policy Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about approved organizational and HR policy documents.'
        'Every answer is strictly grounded with exact page and section citations. <b>I can also answer in Hindi.</b></div>',
        unsafe_allow_html=True
    )

    # Question Input: Tied directly to session state
    query = st.text_input(
        "Ask a policy question:",
        value=st.session_state["query_input"],
        placeholder="e.g. How many days of maternity leave can a female employee avail?",
        key="main_query_field"
    )

    st.caption("Try a sample question:")
    sample_questions = [
        "How many casual leaves are employees entitled to per calendar year?",
        "How many days of maternity leave is a female employee entitled to for pregnancy?",
        "What are the morning and evening operating hours for the SAB Gym at the New Campus?",
        "What is the reimbursement limit for transporting household pets under the relocation scheme?",
        "What is the monthly fixed reimbursement amount for Children Education Allowance per child?",
        "What is the penalty charge for overstay in Type V campus house beyond retirement?",
    ]
    sample_cols = st.columns(3)
    for idx, q in enumerate(sample_questions):
        with sample_cols[idx % 3]:
            st.button(q, key=f"btn_{idx}", on_click=set_search_query, args=(q,), use_container_width=True)

    col1, col2 = st.columns([1, 6])
    with col1:
        submit = st.button("Submit Question", type="primary", use_container_width=True)

    # If the user changed the input in the box, update session state
    if query != st.session_state["query_input"]:
        st.session_state["query_input"] = query

    active_query = query.strip()

    if submit or (active_query and active_query == st.session_state.get("last_searched_query")):
        pass

    if submit and active_query:
        st.session_state["last_searched_query"] = active_query
        
        with st.spinner("Searching approved policy documents and generating grounded answer..."):
            try:
                result = engine.answer_question(
                    question=active_query,
                    provider=provider
                )

                st.session_state["latest_result"] = result
            except Exception as e:
                st.error(f"An error occurred while answering your question: {str(e)}")

    # Display result if available
    if "latest_result" in st.session_state and st.session_state.get("last_searched_query"):
        result = st.session_state["latest_result"]
        st.markdown("---")

        # 1. Conflict Warning Banner
        if result.get("conflict_warning"):
            st.markdown(
                f"""
                <div class="conflict-box">
                    <strong>⚠️ Policy Conflict Detected:</strong><br/>
                    {result['conflict_warning']}
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. Insufficient Info Notice
        if result.get("status") == "insufficient_info":
            st.markdown(
                """
                <div class="unanswerable-box">
                    <strong>ℹ️ Information Not Found in Approved Policies:</strong><br/>
                    The provided approved policy documents do not contain enough information to answer this question.
                </div>
                """,
                unsafe_allow_html=True
            )

        # 3. Grounded Answer
        st.subheader("Answer")
        st.markdown(result["answer"])

        # 4. Verified Sources & Clickable Citations
        st.subheader("Sources & Citations")
        sources = result.get("sources", [])
        
        if sources:
            st.info("💡 **Click on any source card below to expand and view the exact retrieved paragraph or inspect the full page text from the original PDF.**")
            
            for idx, src in enumerate(sources):
                with st.expander(
                    f"📄 {src['document_name']} — Page {src['page']} (Section: {src['section']})",
                    expanded=(idx == 0)
                ):
                    tab_snippet, tab_full_page = st.tabs(["📌 Retrieved Excerpt / Paragraph", "📖 Full Page Text"])
                    
                    with tab_snippet:
                        st.markdown("**Retrieved Paragraph Chunk:**")
                        st.markdown(f"> {src.get('full_content', src['excerpt'])}")
                    
                    with tab_full_page:
                        page_text = extract_original_page_text(
                            src.get("filename", ""), src["page"], src.get("document_name", "")
                        )
                        st.markdown(f"**Original PDF Page {src['page']} Content:**")
                        st.markdown(f'<div class="page-preview-box">{page_text}</div>', unsafe_allow_html=True)
        else:
            st.caption("No approved policy citations available for this query.")


if __name__ == "__main__":
    main()
