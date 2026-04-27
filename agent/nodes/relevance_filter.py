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

Score 8-10: Paper directly addresses the question with experimental
results, comparisons, or explanations relevant to the exact topic asked.

Score 5-7: Paper is related to the topic but does not directly answer
the question, or covers only part of what was asked.

Score 0-4: Paper mentions related terms but is about a different
application, domain, or problem entirely. Examples that score LOW:
- Papers about security, privacy attacks, or membership inference on
  RAG systems score LOW if the user asks how RAG works.
- Papers applying RAG to image generation, computer vision, or
  multimodal systems score LOW if the user asks about text-based RAG.
- Papers applying a concept to a specific industry domain (energy,
  legal, medical) score LOW if the user asks about the underlying
  mechanism, not the application.

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