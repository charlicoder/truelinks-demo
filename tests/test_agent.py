"""
Tests for QCS 2024 Submittal Review Agent
Professional test suite using pytest
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_concrete_submittal():
    """Sample concrete submittal that should be approved."""
    return {
        "type": "Concrete",
        "description": "Ready-mix concrete for structural columns and beams in a commercial building.",
        "specifications": """
            Grade: C40/50
            28-day compressive strength: 40 MPa minimum
            Cement: OPC Type I conforming to ASTM C150
            Slump: 100-150mm
            Maximum aggregate size: 20mm
            Water-cement ratio: 0.45 max
            Test certificates attached
        """
    }


@pytest.fixture
def sample_rejected_submittal():
    """Sample submittal that should be rejected."""
    return {
        "type": "Concrete",
        "description": "Concrete mix for foundation work.",
        "specifications": """
            Grade: C15
            28-day strength: 15 MPa
            No test certificates available
            Mix design not provided
        """
    }


@pytest.fixture
def sample_safety_submittal():
    """Sample safety submittal that should be approved."""
    return {
        "type": "Safety & Health",
        "description": "Personal Protective Equipment (PPE) supply for construction workers.",
        "specifications": """
            Hard hats: EN 397 certified
            Safety boots: S3 rated steel toe
            High-visibility vests: Class 2 reflective, EN ISO 20471
            Safety glasses: EN 166 certified
            All items have valid CE marking
        """
    }


@pytest.fixture
def mock_knowledge_base():
    """Mock knowledge base for testing without PDF loading."""
    mock_kb = Mock()
    mock_kb.is_initialized = True
    mock_kb.chunks = [
        {"text": "QCS 2024 Section 5.6: Minimum compressive strength for structural concrete shall be 30 MPa.", "source": "05/06 Property Requirements.pdf"},
        {"text": "QCS 2024 Section 11.1: All PPE shall comply with EN standards.", "source": "01/10 Welfare Occupational Health and Safety.pdf"},
    ]
    mock_kb.get_context_for_review = Mock(return_value=[
        {"text": "Sample QCS requirement text", "source": "test.pdf", "score": 0.9}
    ])
    mock_kb.search = Mock(return_value=[
        {"text": "Sample search result", "source": "test.pdf", "score": 0.85}
    ])
    return mock_kb


# ============================================================================
# Knowledge Base Tests
# ============================================================================

class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""
    
    def test_knowledge_base_initialization(self):
        """Test that knowledge base can be instantiated."""
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        assert kb is not None
        assert kb.is_initialized == False
        assert kb.chunks == []
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        
        sample_text = " ".join(["word"] * 1000)
        chunks = kb.chunk_text(sample_text, "test_source.pdf")
        
        assert len(chunks) > 0
        assert all("source" in chunk for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
    
    def test_db_paths(self):
        """Test database path generation."""
        from knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        
        assert kb._get_index_path().endswith("faiss_index.bin")
        assert kb._get_chunks_path().endswith("chunks.json")
        assert kb._get_meta_path().endswith("metadata.json")


# ============================================================================
# Agent Tests
# ============================================================================

class TestAgent:
    """Tests for the LangGraph agent."""
    
    def test_review_state_structure(self):
        """Test that ReviewState has all required fields."""
        from agent import ReviewState
        
        # Check TypedDict has required keys
        required_keys = [
            'submittal_type', 'description', 'specifications',
            'context', 'analysis', 'decision', 'confidence',
            'explanation', 'citations', 'recommendations',
            'compliance_summary', 'key_findings', 'issues_found'
        ]
        
        for key in required_keys:
            assert key in ReviewState.__annotations__
    
    @patch('agent.get_knowledge_base')
    def test_retrieve_context(self, mock_get_kb, mock_knowledge_base):
        """Test context retrieval node."""
        from agent import retrieve_context
        
        mock_get_kb.return_value = mock_knowledge_base
        
        state = {
            "submittal_type": "Concrete",
            "description": "Test description",
            "specifications": "Test specs",
            "context": [],
            "analysis": "",
            "decision": "NEEDS_REVIEW",
            "confidence": 0.0,
            "explanation": "",
            "citations": [],
            "recommendations": [],
            "compliance_summary": "",
            "key_findings": [],
            "issues_found": []
        }
        
        result = retrieve_context(state)
        
        assert "context" in result
        mock_knowledge_base.get_context_for_review.assert_called_once()
    
    def test_format_citations(self):
        """Test citation formatting."""
        from agent import format_citations
        
        state = {
            "context": [
                {"source": "test.pdf", "text": "Sample text " * 100, "score": 0.9},
                {"source": "test2.pdf", "text": "Another sample", "score": 0.8}
            ],
            "submittal_type": "Concrete",
            "description": "",
            "specifications": "",
            "analysis": "",
            "decision": "NEEDS_REVIEW",
            "confidence": 0.0,
            "explanation": "",
            "citations": [],
            "recommendations": [],
            "compliance_summary": "",
            "key_findings": [],
            "issues_found": []
        }
        
        result = format_citations(state)
        
        assert "citations" in result
        assert len(result["citations"]) == 2
        assert all("source" in c for c in result["citations"])
        assert all("text" in c for c in result["citations"])


# ============================================================================
# API Tests
# ============================================================================

class TestAPI:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "knowledge_base_ready" in data
        assert "chunks_count" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint serves HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_css_endpoint(self, client):
        """Test CSS endpoint."""
        response = client.get("/style.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
    
    def test_js_endpoint(self, client):
        """Test JavaScript endpoint."""
        response = client.get("/app.js")
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "")
    
    def test_review_endpoint_validation(self, client):
        """Test review endpoint input validation."""
        # Missing required fields
        response = client.post("/api/review", json={})
        assert response.status_code == 422
        
        # Invalid JSON
        response = client.post("/api/review", content="invalid")
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete flow."""
    
    @pytest.mark.integration
    @patch('agent.ChatOpenAI')
    @patch('agent.get_knowledge_base')
    def test_full_review_flow(self, mock_get_kb, mock_llm, mock_knowledge_base, sample_concrete_submittal):
        """Test complete review flow with mocked LLM."""
        from agent import review_submittal
        
        # Setup mocks
        mock_get_kb.return_value = mock_knowledge_base
        
        mock_response = Mock()
        mock_response.content = json.dumps({
            "decision": "APPROVED",
            "confidence": 0.85,
            "compliance_summary": "The submittal meets QCS 2024 requirements.",
            "key_findings": ["Compressive strength meets minimum requirements"],
            "issues_found": [],
            "explanation": "All specifications comply with QCS 2024.",
            "recommendations": ["Continue monitoring quality"]
        })
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = review_submittal(
            sample_concrete_submittal["type"],
            sample_concrete_submittal["description"],
            sample_concrete_submittal["specifications"]
        )
        
        assert "decision" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "citations" in result


# ============================================================================
# Test Runner Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
