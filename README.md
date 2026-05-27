# RAG Fullstack Docker AWS

A complete Retrieval-Augmented Generation (RAG) application with FastAPI backend, Streamlit frontend, and LangChain integration deployed on AWS EC2.

## 🚀 Live Deployment

**Production URLs:**
- **RAG Backend API**: http://100.27.34.29:9000
- **RAG Frontend**: http://100.27.34.29:9501
- **API Documentation**: http://100.27.34.29:9000/docs

## ✨ Features

### RAG Implementation
- **Document Retrieval**: Semantic search using ChromaDB vector database
- **LLM Integration**: Groq and DeepSeek LLM support
- **Web Search**: Tavily integration for real-time web information
- **Document Processing**: PDF, TXT, DOCX file support
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)

### Backend Endpoints
- `POST /query` - Submit questions to RAG system
- `POST /documents/upload` - Upload documents for indexing
- `GET /documents` - List all uploaded documents
- `DELETE /documents/{filename}` - Remove a document
- `GET /health` - Health check
- `GET /models` - Available LLM models

### Frontend
Streamlit UI with document upload, Q&A, and document management.

## 🏗️ Architecture

FastAPI backend (port 9000) + Streamlit frontend (port 9501)
Connected to ChromaDB vector store with HuggingFace embeddings
Integration with Groq/DeepSeek LLMs and Tavily web search

## 📦 Deployment

Docker image: `amith98480/rag-fullstack-docker-aws:latest`

### EC2 Commands
```bash
docker pull amith98480/rag-fullstack-docker-aws:latest
docker run -d --name rag-backend -p 9000:8000 --restart unless-stopped \
  amith98480/rag-fullstack-docker-aws:latest \
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
docker run -d --name rag-frontend -p 9501:8501 --restart unless-stopped \
  amith98480/rag-fullstack-docker-aws:latest \
  streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔄 CI/CD

GitHub Actions automatically builds and deploys on push to main.
Workflow: .github/workflows/ci-cd.yml

## 📝 License

MIT
