from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

SYSTEM_PROMPT = """You are a scientific fact-checker analyzing research paper excerpts.

Given a user question and several text passages from different papers, determine 
whether the passages contradict each other on any claim relevant to the question.

A contradiction means: passage A explicitly claims X, and passage B explicitly 
claims not-X, or claims something incompatible with X.

Do not flag differences in emphasis, scope, or methodology as contradictions.
Only flag direct factual or claim-level conflicts.

Respond ONLY with a JSON object:
{
  "contradiction_detected": true or false,
  "contradiction_note": "one sentence describing the conflict, or empty string if none"
}
"""

def contradiction_detector(state: dict) -> dict:
    user_query = state.get("user_query", "")
    relevant_chunks = state.get("relevant_chunks", [])

    if len(relevant_chunks) <2:
        return{
            "contradiction_detected": False,
            "contradiction_note": ""
        }
    
    #make a numbered list of passages for LLM
    passages = ""
    for i, chunk in enumerate(relevant_chunks):
        passages += f"\nPassage {i+1} (from '{chunk['paper_title']}'):\n"
        passages += chunk["text"][:500] #cap passage to avoid token saturation
        passages += "\n"
    
    prompt = f"User question: {user_query}\n\nPassages:\n{passages}"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        result = json.loads(raw)
        contradiction_detected = bool(result.get("contradiction_detected", False))
        contradiction_note = result.get("contradiction_note", "")
    except (json.JSONDecodeError, ValueError):
        contradiction_detected = False
        contradiction_note = ""
    
    return {
        "contradiction_detected": contradiction_detected,
        "contradiction_note": contradiction_note
    }