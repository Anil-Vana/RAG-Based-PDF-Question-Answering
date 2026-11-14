
import os
import json
import hashlib
import tempfile
from datetime import datetime

import streamlit as st

from src.rag_pipeline import (
    initialize_rag,
    get_llm,
    answer_question
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
    "vector_store": None,
    "llm": None,
    "chat_history": [],
    "processed_pdf": None,
    "processed_pdf_hash": None,
    "processing_complete": False
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def reset_document():
    st.session_state.vector_store = None
    st.session_state.llm = None
    st.session_state.processed_pdf = None
    st.session_state.processed_pdf_hash = None
    st.session_state.processing_complete = False
    st.session_state.chat_history = []


def clear_chat():
    st.session_state.chat_history = []


def export_chat(format_type="txt"):
    """
    Export conversation without serializing Document objects.
    """

    export_data = []

    for message in st.session_state.chat_history:
        sources = []

        for source in message.get("sources", []):
            document = source["document"]

            sources.append({
                "page": document.metadata.get("page"),
                "source": document.metadata.get("source"),
                "content": document.page_content
            })

        export_data.append({
            "question": message["question"],
            "answer": message["answer"],
            "sources": sources
        })

    if format_type == "json":
        return json.dumps(
            export_data,
            indent=4,
            ensure_ascii=False
        )

    lines = []

    for index, message in enumerate(
        export_data,
        start=1
    ):
        lines.append(f"QUESTION {index}")
        lines.append(message["question"])
        lines.append("")
        lines.append("ANSWER")
        lines.append(message["answer"])
        lines.append("")
        lines.append("SOURCES")

        for source in message["sources"]:
            lines.append(
                f"Page {source['page']} - "
                f"{source['source']}"
            )

        lines.append("=" * 60)
        lines.append("")

    return "\n".join(lines)


def process_pdf(uploaded_file):
    """
    Process an uploaded PDF and initialize the RAG pipeline.
    """

    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    # Avoid unnecessary reprocessing
    if (
        file_hash == st.session_state.processed_pdf_hash
        and st.session_state.vector_store is not None
    ):
        st.info("This PDF is already processed.")
        return

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        with st.spinner(
            "Extracting PDF text and creating embeddings..."
        ):

            vector_store = initialize_rag(temp_path)
            llm = get_llm()

        st.session_state.vector_store = vector_store
        st.session_state.llm = llm

        st.session_state.processed_pdf = uploaded_file.name
        st.session_state.processed_pdf_hash = file_hash
        st.session_state.processing_complete = True

        st.session_state.chat_history = []

        st.success("PDF processed successfully!")

    except Exception as error:
        st.error(f"PDF processing failed: {error}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📚 PDF RAG Assistant")

    st.caption(
        "Upload documents and ask questions "
        "using their contents."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        help="Upload a text-based PDF document."
    )

    if uploaded_file is not None:

        st.markdown(
            f"**Selected:** {uploaded_file.name}"
        )

        if st.button(
            "Process PDF",
            use_container_width=True,
            type="primary"
        ):
            process_pdf(uploaded_file)

    st.divider()

    # Document status
    st.subheader("Document Status")

    if st.session_state.processing_complete:

        st.success("Document ready")

        st.write(
            f"**Name:** {st.session_state.processed_pdf}"
        )

        st.write(
            f"**Questions asked:** "
            f"{len(st.session_state.chat_history)}"
        )

    else:

        st.info("No PDF processed yet.")

    st.divider()

    # Chat controls
    st.subheader("Chat Controls")

    if st.button(
        "Clear Chat History",
        use_container_width=True
    ):
        clear_chat()
        st.rerun()

    if st.button(
        "Reset Document",
        use_container_width=True
    ):
        reset_document()
        st.rerun()

    # Export options
    if st.session_state.chat_history:

        st.divider()

        st.subheader("Export Conversation")

        txt_content = export_chat("txt")
        json_content = export_chat("json")

        st.download_button(
            "Download Chat (.txt)",
            data=txt_content,
            file_name="pdf_rag_chat.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.download_button(
            "Download Chat (.json)",
            data=json_content,
            file_name="pdf_rag_chat.json",
            mime="application/json",
            use_container_width=True
        )


# ==================================================
# MAIN APPLICATION
# ==================================================

st.title("PDF Question Answering System")

st.markdown(
    """
    Ask questions about your documents and receive
    AI-generated answers grounded in PDF content.
    """
)

st.divider()


# ==================================================
# EMPTY STATE
# ==================================================

if not st.session_state.processing_complete:

    st.info(
        "Upload a PDF using the sidebar and click "
        "'Process PDF' to begin."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📄")
        st.markdown("**PDF Processing**")
        st.caption(
            "Extract text and create searchable embeddings."
        )

    with col2:
        st.markdown("### 🔍")
        st.markdown("**Hybrid Retrieval**")
        st.caption(
            "Retrieve relevant content using semantic "
            "and keyword search."
        )

    with col3:
        st.markdown("### 💬")
        st.markdown("**Question Answering**")
        st.caption(
            "Ask questions and view supporting sources."
        )


# ==================================================
# DOCUMENT CHAT
# ==================================================

else:

    st.subheader(
        f"📄 {st.session_state.processed_pdf}"
    )

    st.caption(
        "Answers are generated using the retrieved "
        "content from your processed PDF."
    )

    st.divider()

    # Display existing conversation
    for message in st.session_state.chat_history:

        with st.chat_message("user"):
            st.markdown(message["question"])

        with st.chat_message("assistant"):

            st.markdown(message["answer"])

            sources = message.get("sources", [])

            # Display sources only when available
            if sources:

                with st.expander("View Retrieved Sources"):

                    for source in sources:

                        document = source["document"]

                        page = document.metadata.get(
                            "page",
                            "Unknown"
                        )

                        source_name = document.metadata.get(
                            "source",
                            "PDF"
                        )

                        st.markdown(
                            f"**{source_name} | Page {page}**"
                        )

                        st.text(document.page_content)

                        st.divider()

    # User question
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Retrieving relevant information..."
            ):

                try:

                    answer, sources = answer_question(
                        st.session_state.vector_store,
                        st.session_state.llm,
                        question,
                        chat_history=(
                            st.session_state.chat_history
                        )
                    )

                    st.markdown(answer)

                    # Display sources only when available
                    if sources:

                        with st.expander(
                            "View Retrieved Sources"
                        ):

                            for source in sources:

                                document = source["document"]

                                page = document.metadata.get(
                                    "page",
                                    "Unknown"
                                )

                                source_name = document.metadata.get(
                                    "source",
                                    "PDF"
                                )

                                st.markdown(
                                    f"**{source_name} | Page {page}**"
                                )

                                st.text(document.page_content)

                                st.divider()

                    # Save conversation
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

                    # UPDATED: Refresh sidebar question count
                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Answer generation failed: {error}"
                    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "PDF RAG Assistant | "
    "Streamlit · LangChain · ChromaDB · Groq"
)

