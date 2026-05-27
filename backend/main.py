"""
RAG Fullstack FastAPI Application
Provides RESTful API for document retrieval and LLM-based question answering
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization flags
_embeddings = None
_vector_store = None
_llm = None
_tavily_client = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            _embeddings = False
    return _embeddings if _embeddings else None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            from langchain_community.vectorstores import Chroma
            embeddings = get_embeddings()
            if embeddings:
                persist_directory = "/tmp/chroma_db"
                os.makedirs(persist_directory, exist_ok=True)
                _vector_store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=persist_directory,
                    collection_name="documents",
                )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            _vector_store = False
    return _vector_store if _vector_store else None

def get_llm():
    global _llm
    if _llm is None:
        try:
            from langchain_groq import ChatGroq
            groq_api_key = os.getenv("GROQ_API_KEY", "")
            if groq_api_key:
                _llm = ChatGroq(model="mixtral-8x7b-32768", api_key=groq_api_key, temperature=0.7)
            else:
                _llm = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            _llm = False
    return _llm if _llm else None

def get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        try:
            from tavily import TavilyClient
            tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            if tavily_api_key:
                _tavily_client = TavilyClient(api_key=tavily_api_key)
            else:
                _tavily_client = False
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily: {e}")
            _tavily_client = False
    return _tavily_client if _tavily_client else None

# Initialize FastAPI app
app = FastAPI(
    title="RAG Fullstack API",
    description="Retrieval-Augmented Generation API with LLM Integration",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str
    use_web_search: bool = False
    model: str = "groq"
    temperature: float = 0.7
    max_tokens: int = 1000


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    question: str
    answer: str
    sources: List[str] = []
    model_used: str
    confidence: float


class DocumentInfo(BaseModel):
    """Information about uploaded documents"""
    filename: str
    size_bytes: int
    upload_timestamp: str


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    environment: str
    version: str


# Routes
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "1.0.0",
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Submit a question to the RAG system

    Returns:
        - answer: LLM-generated answer
        - sources: Relevant document sources
        - model_used: Which LLM model was used
        - confidence: Confidence score of the answer
    """
    try:
        llm = get_llm()
        vector_store = get_vector_store()

        # If no LLM configured, return a placeholder response
        if not llm:
            return QueryResponse(
                question=request.question,
                answer="LLM API key not configured. Please set GROQ_API_KEY environment variable to enable RAG functionality.",
                sources=[],
                model_used=request.model,
                confidence=0.0,
            )

        logger.info(f"Processing query: {request.question}")

        sources = []
        context = ""

        # Search for relevant documents if vector store is available
        if vector_store:
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(request.question)
                sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
                context = "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # Optionally search web
        web_context = ""
        if request.use_web_search:
            tavily_client = get_tavily_client()
            if tavily_client:
                try:
                    results = tavily_client.search(query=request.question, max_results=3)
                    web_context = "\n".join([r["content"] for r in results.get("results", [])])
                    sources.extend([r["url"] for r in results.get("results", [])])
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")

        # Build RAG prompt
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the question. If the context doesn't contain
            relevant information, use your knowledge but indicate that it's from your training data.

            Context:
            {context}

            Web Search Results (if available):
            {web_context}

            Question: {question}

            Answer:"""
        )

        # Create RAG chain
        rag_chain = (
            {
                "context": lambda x: context,
                "web_context": lambda x: web_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate answer
        answer = rag_chain.invoke(request.question)

        confidence = 0.85 if context else 0.6

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=list(set(sources))[:5],
            model_used=request.model,
            confidence=confidence,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for RAG indexing

    Supported formats: PDF, TXT, DOCX
    """
    try:
        if file.size is None or file.size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        allowed_extensions = {".pdf", ".txt", ".docx"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )

        logger.info(f"Uploading document: {file.filename}")
        vector_store = get_vector_store()

        if not vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not available. Document indexing is disabled."
            )

        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Load and extract text based on file type
            docs = []
            if file_ext == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif file_ext == ".txt":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(tmp_path)
                docs = loader.load()
            elif file_ext == ".docx":
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(tmp_path)
                docs = loader.load()

            if not docs:
                raise HTTPException(status_code=400, detail="Could not extract text from document")

            # Split into chunks
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(docs)

            # Add metadata
            for chunk in chunks:
                chunk.metadata["source"] = file.filename
                chunk.metadata["upload_date"] = datetime.now().isoformat()

            # Store in vector database
            vector_store.add_documents(chunks)
            logger.info(f"Successfully indexed {len(chunks)} chunks from {file.filename}")

            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": len(chunks),
                "message": "Document uploaded and indexed successfully",
            }

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    try:
        vector_store = get_vector_store()
        if not vector_store:
            return []

        # Get all documents from vector store
        collection = vector_store.get()
        documents = {}

        if collection and "metadatas" in collection:
            for metadata in collection["metadatas"]:
                source = metadata.get("source", "unknown")
                if source not in documents:
                    documents[source] = {
                        "filename": source,
                        "size_bytes": 0,
                        "upload_timestamp": metadata.get("upload_date", "unknown"),
                    }

        return [DocumentInfo(**doc) for doc in documents.values()]

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return []


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its embeddings"""
    try:
        vector_store = get_vector_store()
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not available")

        # Get all documents and filter by source
        collection = vector_store.get()
        if not collection or "ids" not in collection or "metadatas" not in collection:
            raise HTTPException(status_code=404, detail="Document not found")

        ids_to_delete = []
        for idx, metadata in enumerate(collection["metadatas"]):
            if metadata.get("source") == filename:
                ids_to_delete.append(collection["ids"][idx])

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

        # Delete from vector store
        vector_store.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for document: {filename}")

        return {
            "status": "success",
            "deleted": filename,
            "chunks_removed": len(ids_to_delete),
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def available_models():
    """Get list of available LLM models"""
    return {
        "available_models": [
            {"name": "groq", "description": "Groq LLM"},
            {"name": "deepseek", "description": "DeepSeek LLM"},
            {"name": "openai", "description": "OpenAI GPT"},
        ]
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "RAG Fullstack API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
