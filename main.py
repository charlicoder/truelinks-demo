"""
FastAPI Backend for Construction Submittal Review Agent
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os

from agent import review_submittal
from knowledge_base import get_knowledge_base

app = FastAPI(
    title="Construction Submittal Review Agent",
    description="AI-powered agent for reviewing construction submittals against QCS 2024 standards",
    version="1.0.0"
)

# Trusted Host middleware - allow ALB hostname, public IP access, and localhost
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "*.me-south-1.elb.amazonaws.com",
        "localhost",
        "127.0.0.1",
        "*",  # Allow all hosts (for public IP access and ALB health checks)
    ],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SubmittalRequest(BaseModel):
    type: str
    description: str
    specifications: str


class Citation(BaseModel):
    source: str
    text: str
    relevance: Optional[float] = None


class ReviewResponse(BaseModel):
    decision: str
    confidence: float
    explanation: str
    citations: List[Citation]
    recommendations: List[str]
    analysis: str


class HealthResponse(BaseModel):
    status: str
    knowledge_base_ready: bool
    chunks_count: int


# Endpoints
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health and knowledge base status."""
    try:
        kb = get_knowledge_base()
        return HealthResponse(
            status="healthy",
            knowledge_base_ready=kb.is_initialized,
            chunks_count=len(kb.chunks)
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            knowledge_base_ready=False,
            chunks_count=0
        )


@app.post("/api/review", response_model=ReviewResponse)
async def review_construction_submittal(request: SubmittalRequest):
    """Review a construction submittal against QCS 2024 standards."""
    try:
        result = review_submittal(
            submittal_type=request.type,
            description=request.description,
            specifications=request.specifications
        )
        
        return ReviewResponse(
            decision=result["decision"],
            confidence=result["confidence"],
            explanation=result["explanation"],
            citations=[Citation(**c) for c in result["citations"]],
            recommendations=result["recommendations"],
            analysis=result["analysis"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/style.css")
async def serve_css():
    """Serve the stylesheet."""
    return FileResponse(os.path.join(BASE_DIR, "style.css"), media_type="text/css")


@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript."""
    return FileResponse(os.path.join(BASE_DIR, "app.js"), media_type="application/javascript")



if __name__ == "__main__":
    import uvicorn
    
    print("Initializing knowledge base...")
    kb = get_knowledge_base()
    print(f"Knowledge base ready with {len(kb.chunks)} chunks")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
