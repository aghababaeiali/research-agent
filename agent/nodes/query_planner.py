from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

SYSTEM_PROMPT = """You are a research query specialist.
Given a user question, produce 1 to 3 precise academic search queries
suitable for searching arXiv.

IMPORTANT: Queries must be plain keyword phrases only. No boolean operators,
no arXiv IDs, no site: syntax, no quotes, no AND/OR operators.
arXiv search only supports plain keyword queries.

Rules:
- Plain keyword phrases only, 3-8 words each
- Target papers that DIRECTLY answer the user's question
- Include the specific domain if the question implies one
- For "explain X", "how does X work", or "what is X" questions, add
  "survey", "tutorial", or "overview" to at least one query
- For comparison questions, include "benchmark", "comparison", or "evaluation"
- For broad topics, add constraining terms like "empirical study" or
  "systematic analysis" to at least one query
- Avoid overly broad queries that match unrelated subfields
- Each query must be distinct

Respond ONLY with a JSON array of strings. No explanation, no markdown.
"""

def query_planner(state: dict) -> dict:
    user_query = state.get("user_query", "")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        search_queries = json.loads(raw)
        if not isinstance(search_queries, list):
            raise ValueError("Expected a list")
    except (json.JSONDecodeError, ValueError):
        search_queries = [user_query]

    return {"search_queries": search_queries}