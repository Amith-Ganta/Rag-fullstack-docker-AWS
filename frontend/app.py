"""
RAG Fullstack Streamlit Frontend
Interactive UI for document upload and RAG-based question answering
"""

import streamlit as st
import requests
import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Fullstack",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='main-header'>🤖 RAG Fullstack Application</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selection
    st.subheader("LLM Configuration")
    model = st.selectbox(
        "Select LLM Model",
        ["groq", "deepseek", "openai"],
        help="Choose which LLM to use for answering questions"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more deterministic"
    )

    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        help="Maximum length of the response"
    )

    # Search options
    st.subheader("Search Options")
    use_web_search = st.checkbox(
        "Enable Web Search",
        value=False,
        help="Include web search results in the answer"
    )

    # Health check
    st.subheader("API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.success(f"✓ API Connected")
            st.caption(f"Environment: {data.get('environment')}")
        else:
            st.error("✗ API Error")
    except requests.exceptions.RequestException:
        st.error("✗ API Offline")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📄 Documents", "ℹ️ Info"])

# Tab 1: Q&A
with tab1:
    st.markdown("<h2 class='section-header'>Ask Questions about Your Documents</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.text_area(
            "Your Question:",
            placeholder="Ask me anything about your documents...",
            height=120,
            label_visibility="collapsed"
        )

    with col2:
        submit_button = st.button(
            "🔍 Search",
            use_container_width=True,
            key="search_button"
        )

    if submit_button and question:
        with st.spinner("🔄 Processing your question..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "use_web_search": use_web_search,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.markdown("### 📌 Answer")
                    st.write(result.get("answer", "No answer generated"))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Used", result.get("model_used", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col3:
                        st.metric("Sources", len(result.get("sources", [])))

                    if result.get("sources"):
                        st.markdown("### 📚 Sources")
                        for source in result["sources"]:
                            st.write(f"• {source}")

                else:
                    st.error(f"❌ Error: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. The API is taking too long to respond.")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {str(e)}")
            except json.JSONDecodeError:
                st.error("❌ Invalid response from API")

    elif submit_button:
        st.warning("⚠️ Please enter a question")

# Tab 2: Document Management
with tab2:
    st.markdown("<h2 class='section-header'>Document Management</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Documents")
        st.write("Supported formats: PDF, TXT, DOCX")

        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "docx"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
                        response = requests.post(
                            f"{API_URL}/documents/upload",
                            files=files,
                            timeout=30
                        )

                        if response.status_code == 200:
                            st.success(f"✓ {uploaded_file.name} uploaded successfully!")
                        else:
                            st.error(f"❌ Upload failed: {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Error: {str(e)}")

    with col2:
        st.subheader("📚 Your Documents")

        try:
            response = requests.get(f"{API_URL}/documents", timeout=5)
            if response.status_code == 200:
                documents = response.json()
                if documents:
                    for doc in documents:
                        col_name, col_size, col_delete = st.columns([3, 1, 1])
                        with col_name:
                            st.write(f"📄 {doc['filename']}")
                        with col_size:
                            size_mb = doc['size_bytes'] / (1024 * 1024)
                            st.caption(f"{size_mb:.1f} MB")
                        with col_delete:
                            if st.button("🗑️", key=doc['filename']):
                                # Delete document
                                del_response = requests.delete(
                                    f"{API_URL}/documents/{doc['filename']}"
                                )
                                if del_response.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                else:
                    st.info("No documents uploaded yet.")
            else:
                st.error("Could not fetch documents")

        except requests.exceptions.RequestException:
            st.error("Could not connect to API")

# Tab 3: Information
with tab3:
    st.markdown("<h2 class='section-header'>About RAG Fullstack</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### What is RAG?
    **Retrieval-Augmented Generation (RAG)** combines:
    - 📚 **Retrieval**: Finding relevant documents
    - 🧠 **Generation**: Using LLMs to answer questions
    - 🔗 **Integration**: Connecting external knowledge with AI

    ### Features
    - ✅ Upload and index documents
    - ✅ Ask questions about your documents
    - ✅ Support for multiple LLM providers
    - ✅ Web search integration
    - ✅ REST API for programmatic access
    - ✅ Streamlit web interface

    ### Supported Models
    - **Groq**: Fast inference API
    - **DeepSeek**: Open-source LLM
    - **OpenAI**: GPT-4 and GPT-3.5

    ### Architecture
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Frontend**\nStreamlit Web UI")
    with col2:
        st.success("**API**\nFastAPI Server")
    with col3:
        st.warning("**Backend**\nRAG & LLM Logic")

    st.markdown("""
    ### Project Links
    - 🔗 [GitHub Repository](https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS)
    - 📚 [API Documentation](/docs)
    - 📧 Contact: gantaamith007@gmail.com
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 30px;'>
    <p>RAG Fullstack v1.0.0 | Made with ❤️ | Deployed on AWS EC2</p>
</div>
""", unsafe_allow_html=True)
