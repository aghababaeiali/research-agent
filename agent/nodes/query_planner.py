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

Rules:
- Queries must target papers that DIRECTLY answer the user's question
- Include the specific domain if the question implies one (e.g. NLP, text, QA)
- Avoid overly broad terms that match unrelated subfields
- Each query must be distinct and target a different angle of the question

For example, if the user asks "how does RAG work for question answering":
GOOD: ["retrieval augmented generation question answering mechanism", 
       "RAG pipeline text QA dense retrieval"]
BAD:  ["retrieval augmented generation", "RAG models"]

Respond ONLY with a JSON array of strings. No explanation, no markdown.
"""

def query_planner(state:dict) -> dict:
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
        #fallback: use the original query as it is
        search_queries = [user_query]
    
    return {"search_queries": search_queries}