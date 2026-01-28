# QCS 2024 Submittal Review Agent

AI-powered agent that reviews construction submittals against Qatar Construction Standards (QCS 2024) and provides approval/rejection decisions with citations.

## Features

- **LangGraph Agent**: Stateful graph-based workflow for document review
- **FAISS Vector Search**: Semantic search over QCS 2024 standards
- **Structured Output**: Approval/rejection with compliance summary, findings, and citations
- **FastAPI Backend**: RESTful API with modern async support
- **Persistent Knowledge Base**: Cached embeddings for fast startup

## Architecture

```
User Input → FastAPI → LangGraph Agent → [Retrieve → Analyze → Decide → Format] → Response
                            ↓
                    FAISS Vector Store
                            ↓
                    QCS 2024 PDFs
```

## Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API Key

### Local Setup

```bash
# Clone and setup
cd truelinks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variable
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the server
python main.py
```

Access at: http://localhost:8000

### Docker Setup

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/health` | GET | Health check |
| `/api/review` | POST | Submit for review |

### Review Request

```json
POST /api/review
{
    "type": "Concrete",
    "description": "Ready-mix concrete for structural columns",
    "specifications": "Grade: C40/50, 28-day strength: 40 MPa"
}
```

### Review Response

```json
{
    "decision": "APPROVED",
    "confidence": 0.85,
    "compliance_summary": "The submittal meets QCS 2024 requirements...",
    "key_findings": ["Finding 1", "Finding 2"],
    "issues_found": [],
    "explanation": "Detailed explanation...",
    "citations": [{"source": "05/06.pdf", "text": "..."}],
    "recommendations": ["Recommendation 1"]
}
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test class
pytest tests/test_agent.py::TestAPI
```

## Project Structure

```
truelinks/
├── main.py              # FastAPI application
├── agent.py             # LangGraph agent
├── knowledge_base.py    # FAISS vector store
├── index.html           # Frontend UI
├── style.css            # Styles
├── app.js               # Frontend JavaScript
├── tests/               # Test suite
│   └── test_agent.py
├── QCS2024/             # Knowledge base PDFs
├── kb_database/         # Cached embeddings
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 |

## License

MIT
