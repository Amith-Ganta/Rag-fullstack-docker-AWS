"""
RAG Fullstack — self-contained Streamlit app.

This is a single-process version of the project that runs entirely on
Streamlit Community Cloud (no separate FastAPI backend). It does the RAG work
in-process: upload a document, it gets chunked and embedded into an in-memory
Chroma store, and questions are answered by the selected LLM grounded on the
retrieved chunks (optionally augmented with web search).

API keys are read from st.secrets (Streamlit Cloud) and fall back to
environment variables for local runs.
"""

import os
import tempfile

import streamlit as st


# --------------------------------------------------------------------------- #
# Config / secrets
# --------------------------------------------------------------------------- #
def get_key(name: str) -> str:
    """Read a secret from st.secrets first, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # st.secrets raises if no secrets file exists at all (local dev).
        pass
    return os.getenv(name, "")


st.set_page_config(
    page_title="RAG Fullstack",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_vector_store():
    """One in-memory Chroma collection for the whole session/app."""
    from langchain_community.vectorstores import Chroma

    return Chroma(
        embedding_function=get_embeddings(),
        collection_name="documents",
    )


def get_llm(provider: str, temperature: float):
    """Build a chat LLM for the requested provider, or None if its key is missing."""
    provider = (provider or "groq").lower()
    try:
        if provider == "groq":
            key = get_key("GROQ_API_KEY")
            if not key:
                return None, "GROQ_API_KEY"
            from langchain_groq import ChatGroq

            return ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=key,
                temperature=temperature,
            ), None
        if provider == "deepseek":
            key = get_key("DEEPSEEK_API_KEY")
            if not key:
                return None, "DEEPSEEK_API_KEY"
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="deepseek-chat",
                api_key=key,
                base_url="https://api.deepseek.com",
                temperature=temperature,
            ), None
        if provider == "openai":
            key = get_key("OPENAI_API_KEY")
            if not key:
                return None, "OPENAI_API_KEY"
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model="gpt-4o-mini",
                api_key=key,
                temperature=temperature,
            ), None
    except Exception as e:  # noqa: BLE001
        return None, f"init error: {e}"
    return None, f"unknown provider '{provider}'"


def web_search(question: str):
    """Run a Tavily search.

    Returns (context_text, source_urls, error). On success error is None; on
    failure context/urls are empty and error is a human-readable string so the
    UI can tell the user *why* there were no web results (instead of silently
    pretending web search ran).
    """
    key = get_key("TAVILY_API_KEY")
    if not key:
        return "", [], "TAVILY_API_KEY not set"
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        resp = client.search(query=question, max_results=3)
        results = resp.get("results", []) if isinstance(resp, dict) else []
        context = "\n".join(r.get("content", "") for r in results)
        urls = [r.get("url", "") for r in results if r.get("url")]
        if not results:
            return "", [], "Tavily returned no results"
        return context, urls, None
    except Exception as e:  # noqa: BLE001
        return "", [], f"Tavily error: {e}"


def index_uploaded_file(uploaded_file) -> int:
    """Chunk + embed an uploaded file into the vector store. Returns chunk count."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(tmp_path).load()
        elif suffix == ".txt":
            from langchain_community.document_loaders import TextLoader

            docs = TextLoader(tmp_path).load()
        elif suffix == ".docx":
            from langchain_community.document_loaders import (
                UnstructuredWordDocumentLoader,
            )

            docs = UnstructuredWordDocumentLoader(tmp_path).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if not docs:
            raise ValueError("Could not extract text from the document.")

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        for c in chunks:
            c.metadata["source"] = uploaded_file.name

        get_vector_store().add_documents(chunks)
        return len(chunks)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def answer_question(question, provider, temperature, use_web):
    llm, missing = get_llm(provider, temperature)
    if llm is None:
        return None

    # Retrieve relevant chunks (vector store may be empty -> no context).
    context, doc_sources = "", []
    try:
        retriever = get_vector_store().as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        doc_sources = list({d.metadata.get("source", "unknown") for d in docs})
    except Exception:  # noqa: BLE001
        pass

    web_context, web_sources, web_error = ("", [], None)
    if use_web:
        web_context, web_sources, web_error = web_search(question)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question. If the context doesn't
        contain relevant information, use your own knowledge but say that it's from
        your training data rather than the provided documents.

        Context:
        {context}

        Web Search Results (if available):
        {web_context}

        Question: {question}

        Answer:"""
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "web_context": web_context, "question": question}
    )
    return {
        "answer": answer,
        "sources": list({*doc_sources, *web_sources})[:5],
        "model_used": provider,
        "confidence": 0.85 if context else 0.6,
        "web_error": web_error,
    }


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>🤖 RAG Fullstack Application</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["groq", "deepseek", "openai"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_web = st.checkbox("Enable Web Search", value=False)

    st.subheader("Key Status")
    for label, env in [
        ("Groq", "GROQ_API_KEY"),
        ("OpenAI", "OPENAI_API_KEY"),
        ("DeepSeek", "DEEPSEEK_API_KEY"),
        ("Tavily (web)", "TAVILY_API_KEY"),
    ]:
        st.write(f"{'✅' if get_key(env) else '⬜'} {label}")

tab_qa, tab_docs, tab_info = st.tabs(["💬 Q&A", "📄 Documents", "ℹ️ Info"])

with tab_qa:
    st.subheader("Ask a question")
    question = st.text_area(
        "Your question",
        placeholder="Ask anything — upload a document first to ground the answer.",
        height=120,
        label_visibility="collapsed",
    )
    if st.button("🔍 Search", use_container_width=False):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            llm, missing = get_llm(provider, temperature)
            if llm is None:
                st.error(
                    f"❌ {provider} is not configured. Missing **{missing}**. "
                    "Add it under app Settings → Secrets and reboot."
                )
            else:
                with st.spinner("🔄 Thinking..."):
                    result = answer_question(
                        question, provider, temperature, use_web
                    )
                if use_web and result.get("web_error"):
                    st.warning(f"🌐 Web search unavailable: {result['web_error']}")
                st.markdown("### 📌 Answer")
                st.write(result["answer"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", result["model_used"])
                c2.metric("Confidence", f"{result['confidence']:.0%}")
                c3.metric("Sources", len(result["sources"]))
                if result["sources"]:
                    st.markdown("### 📚 Sources")
                    for s in result["sources"]:
                        st.write(f"• {s}")

with tab_docs:
    st.subheader("📤 Upload a document")
    st.caption("Supported: PDF, TXT, DOCX. Indexed into an in-memory store.")
    uploaded = st.file_uploader(
        "Choose a document", type=["pdf", "txt", "docx"], label_visibility="collapsed"
    )
    if uploaded is not None and st.button("Index document"):
        if uploaded.size == 0:
            st.error("❌ The file is empty.")
        else:
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    n = index_uploaded_file(uploaded)
                    st.success(f"✓ Indexed {n} chunks from {uploaded.name}.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Could not index document: {e}")

with tab_info:
    st.markdown(
        """
        ### What is this?
        A **self-contained RAG app**: retrieval (embeddings + vector search) +
        generation (LLM), running entirely inside Streamlit.

        **How to use**
        1. Add your API keys in the app's *Settings → Secrets* (at minimum `GROQ_API_KEY`).
        2. Upload a document in the **Documents** tab.
        3. Ask questions in the **Q&A** tab — answers are grounded on your document.

        Without an uploaded document, answers come from the model's own knowledge.

        🔗 [GitHub](https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS)
        """
    )

st.divider()
st.caption("RAG Fullstack v1.0.0 · self-contained Streamlit build")
