from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
import os
from dotenv import load_dotenv

from agent import state

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

SYSTEM_PROMPT = """You are a research relevance evaluator.
Given a user question and a paper abstract, score how directly useful 
this paper would be for ANSWERING the user's question.

A high score (7-10) means: the paper directly addresses the user's question
and contains information that would help answer it.

A low score (0-4) means: the paper mentions related terms but does not 
actually address the user's question, or applies the concept to a specific 
domain without explaining the underlying mechanism.

Be strict. A paper that USES RAG for a specific application (energy sector, 
legal, medical) scores LOW if the user asks HOW RAG works fundamentally.
Domain application papers are not the same as explanatory papers.

Respond ONLY with a JSON object:
{"score": integer 0-10, "reason": "one sentence explanation"}
"""

def _score_abstract(user_query : str, paper: dict) -> dict:
    propmt = f"Question: {user_query}\n\nAbstract: {paper['abstract']}"
    message=[
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=propmt)
    ]
    response = llm.invoke(message)
    raw = response.content.strip()

    try:
        result = json.loads(raw)
        score = int(result.get("score", 0))
        reason = result.get("reason", "")
    except (json.JSONDecodeError, ValueError, KeyError):
        score = 0
        reason = "could not parse relevance score"
    
    return {**paper, "relevance_score": score, "relevance_reason": reason}

def relevance_filter(state:dict) -> dict:
    user_query = state.get("user_query", "")
    candidate_papers = state.get("candidate_papers", [])
    threshold = 7

    scored_papers = [
        _score_abstract(user_query, paper)
        for paper in candidate_papers
    ]

    selected_papers = [
        p for p in scored_papers
        if p["relevance_score"] >= threshold
    ]

    retrieval_sufficient = len(selected_papers) >= 2

    return{
        "selected_papers": selected_papers,
        "retrieval_sufficient": retrieval_sufficient
    }