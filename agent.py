"""
LangGraph Agent for Construction Submittal Review
"""

import os
from typing import TypedDict, List, Dict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from knowledge_base import get_knowledge_base

load_dotenv()


# State definition
class ReviewState(TypedDict):
    """State for the submittal review agent."""
    submittal_type: str
    description: str
    specifications: str
    context: List[Dict]
    analysis: str
    decision: Literal["APPROVED", "REJECTED", "NEEDS_REVIEW"]
    confidence: float
    explanation: str
    citations: List[Dict]
    recommendations: List[str]
    compliance_summary: str
    key_findings: List[str]
    issues_found: List[str]


# Node functions
def retrieve_context(state: ReviewState) -> ReviewState:
    """Retrieve relevant context from QCS2024 knowledge base."""
    kb = get_knowledge_base()
    
    context = kb.get_context_for_review(
        state["submittal_type"],
        state["description"],
        state["specifications"]
    )
    
    return {**state, "context": context}


def analyze_compliance(state: ReviewState) -> ReviewState:
    """Analyze submittal against retrieved standards using LLM."""
    
    # Build context string from retrieved chunks
    context_text = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text'][:500]}..."
        for c in state["context"]
    ]) if state["context"] else "No relevant standards found in knowledge base."
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    system_prompt = """You are a construction standards compliance reviewer for Qatar Construction Standards (QCS 2024).
    
Your task is to analyze a construction submittal against the relevant QCS 2024 standards provided.

FOCUS ON:
1. Does the submittal meet the CORE technical requirements in QCS 2024?
2. Are the specified materials, grades, and standards appropriate?
3. Are there any CLEAR violations of minimum requirements?

IMPORTANT GUIDELINES:
- Focus on what IS provided in the submittal, not what MIGHT be missing
- If specs meet the technical requirements, that is compliant
- Administrative items (training, stock levels, etc.) are secondary concerns
- Only flag issues for CLEAR technical non-compliance

Be practical and fair in your assessment."""

    user_prompt = f"""## Submittal Details
- Type: {state["submittal_type"]}
- Description: {state["description"]}
- Specifications: {state["specifications"]}

## Relevant QCS 2024 Standards
{context_text}

Analyze this submittal for compliance with the TECHNICAL requirements. Focus on:
1. Do the specifications meet minimum QCS 2024 requirements?
2. Are there any clear violations?
3. What specific standards sections apply?"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return {**state, "analysis": response.content}


def make_decision(state: ReviewState) -> ReviewState:
    """Make approval/rejection decision based on analysis with structured output."""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    decision_prompt = f"""Based on this compliance analysis for a {state["submittal_type"]} submittal, make a decision:

SUBMITTAL DETAILS:
- Type: {state["submittal_type"]}
- Description: {state["description"]}
- Specifications: {state["specifications"]}

ANALYSIS:
{state["analysis"]}

## DECISION CRITERIA (FOLLOW STRICTLY):

**APPROVED** - Use when:
- The submittal specifications MEET the core technical requirements of QCS 2024
- Materials, grades, and standards referenced are appropriate
- Certifications and test results (if provided) meet requirements
- Minor administrative details can be noted as recommendations

**REJECTED** - Use ONLY when:
- There is a CLEAR violation of minimum technical requirements
- Specified grades/strengths are BELOW QCS 2024 minimums
- Required certifications are explicitly stated as missing/unavailable
- Materials do not meet mandatory standards

**NEEDS_REVIEW** - Use ONLY when:
- Critical technical information is completely missing (e.g., no grade specified at all)
- Cannot determine compliance due to insufficient technical data
- DO NOT use this just because administrative items aren't mentioned

## IMPORTANT:
- If specifications meet technical requirements, decision should be APPROVED
- Focus on what IS provided, not hypothetical missing items
- Training, stock levels, and other administrative items are NOT reasons to reject or flag for review
- Be practical - if a submittal has proper specs, grades, and certifications, APPROVE it

Provide a STRUCTURED decision response in EXACTLY this JSON format:
{{
    "decision": "APPROVED" or "REJECTED" or "NEEDS_REVIEW",
    "confidence": 0.0 to 1.0,
    "compliance_summary": "One paragraph summarizing overall compliance status",
    "key_findings": [
        "Finding 1: Specific requirement that was met",
        "Finding 2: Another requirement that was checked"
    ],
    "issues_found": [
        "Issue 1: Only include ACTUAL technical issues (can be empty array if none)"
    ],
    "explanation": "Clear explanation of why this decision was made",
    "recommendations": [
        "Optional improvements or best practices"
    ]
}}

Only output the JSON, nothing else."""

    response = llm.invoke([HumanMessage(content=decision_prompt)])
    
    # Parse response
    import json
    try:
        # Clean response and parse JSON
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        
        return {
            **state,
            "decision": result.get("decision", "NEEDS_REVIEW"),
            "confidence": result.get("confidence", 0.5),
            "compliance_summary": result.get("compliance_summary", ""),
            "key_findings": result.get("key_findings", []),
            "issues_found": result.get("issues_found", []),
            "explanation": result.get("explanation", "Unable to determine compliance status."),
            "recommendations": result.get("recommendations", [])
        }
    except Exception as e:
        print(f"Error parsing decision: {e}")
        return {
            **state,
            "decision": "NEEDS_REVIEW",
            "confidence": 0.3,
            "compliance_summary": "Analysis completed but structured output could not be generated.",
            "key_findings": [],
            "issues_found": [],
            "explanation": "Analysis completed but decision could not be automatically determined. Please review the full analysis.",
            "recommendations": ["Manual review recommended", "Verify submittal details are complete"]
        }


def format_citations(state: ReviewState) -> ReviewState:
    """Format citations from context for output."""
    citations = []
    
    for ctx in state.get("context", []):
        citations.append({
            "source": ctx["source"],
            "text": ctx["text"][:300] + "..." if len(ctx["text"]) > 300 else ctx["text"],
            "relevance": ctx.get("score", 0)
        })
    
    return {**state, "citations": citations}


# Build the graph
def create_review_agent():
    """Create and compile the LangGraph review agent."""
    
    workflow = StateGraph(ReviewState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("analyze", analyze_compliance)
    workflow.add_node("decide", make_decision)
    workflow.add_node("format", format_citations)
    
    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "decide")
    workflow.add_edge("decide", "format")
    workflow.add_edge("format", END)
    
    return workflow.compile()


# Global agent instance
_agent = None

def get_agent():
    """Get or create the review agent."""
    global _agent
    if _agent is None:
        _agent = create_review_agent()
    return _agent


def review_submittal(submittal_type: str, description: str, specifications: str) -> Dict:
    """Run the review agent on a submittal."""
    agent = get_agent()
    
    initial_state: ReviewState = {
        "submittal_type": submittal_type,
        "description": description,
        "specifications": specifications,
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
    
    result = agent.invoke(initial_state)
    
    return {
        "decision": result["decision"],
        "confidence": result["confidence"],
        "compliance_summary": result.get("compliance_summary", ""),
        "key_findings": result.get("key_findings", []),
        "issues_found": result.get("issues_found", []),
        "explanation": result["explanation"],
        "citations": result["citations"],
        "recommendations": result["recommendations"],
        "analysis": result["analysis"]
    }
