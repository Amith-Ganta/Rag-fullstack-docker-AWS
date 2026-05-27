"""
Unit tests for RAG Fullstack API
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "environment" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestQueryEndpoint:
    """Test RAG query endpoint"""

    def test_query_with_valid_request(self):
        payload = {
            "question": "What is RAG?",
            "use_web_search": False,
            "model": "groq",
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "question" in data
        assert data["question"] == "What is RAG?"

    def test_query_without_question(self):
        payload = {
            "question": "",
            "model": "groq",
        }
        response = client.post("/query", json=payload)
        # Should either accept empty or return validation error
        assert response.status_code in [200, 422]

    def test_query_with_web_search_enabled(self):
        payload = {
            "question": "What is the latest news?",
            "use_web_search": True,
            "model": "groq",
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200

    def test_query_response_structure(self):
        payload = {
            "question": "Test question",
            "model": "groq",
        }
        response = client.post("/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "question" in data
            assert "answer" in data
            assert "model_used" in data
            assert "confidence" in data
            assert "sources" in data


class TestDocumentEndpoints:
    """Test document upload and management"""

    def test_list_documents_returns_200(self):
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_upload_empty_file(self):
        # This test checks behavior with empty files
        response = client.post("/documents/upload", files={"file": ("empty.txt", b"")})
        assert response.status_code in [200, 400]

    def test_unsupported_file_type(self):
        # Try uploading unsupported file type
        response = client.post(
            "/documents/upload",
            files={"file": ("test.exe", b"some content")}
        )
        assert response.status_code in [400, 422]


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_returns_api_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestAvailableModels:
    """Test available models endpoint"""

    def test_get_available_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert len(data["available_models"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
