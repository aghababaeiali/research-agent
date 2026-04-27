from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # use the larger model for generation
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

SYSTEM_PROMPT = """You are a research assistant that answers questions based 
strictly on provided paper excerpts.

Rules:
- Answer only from the provided passages. Do not use outside knowledge.
- Cite sources inline using [Paper Title] format after each claim.
- If passages conflict, describe both positions clearly and label them.
- End your answer with a JSON block on its own line in this exact format:
  {"confidence": 0.0-1.0, "citations": ["paper title 1", "paper title 2"]}
- Confidence reflects how completely the passages answered the question.
  0.9-1.0: passages directly and fully answer the question
  0.6-0.8: passages partially answer the question
  0.3-0.5: passages are tangentially related
  0.0-0.2: passages do not answer the question
"""

def answer_synthesizer(state: dict) -> dict:
    user_query = state.get("user_query", "")
    relevant_chunks = state.get("relevant_chunks", [])
    contradiction_detected = state.get("contradiction_detected", False)
    contradiction_note = state.get("contradiction_note", "")

    if not relevant_chunks:
        return {
            "final_answer": "I could not find relevant sources to answer this question.",
            "citations": [],
            "confidence_score": 0.0
        }

    # build passages block
    passages = ""
    for i, chunk in enumerate(relevant_chunks):
        passages += f"\nPassage {i+1} (from '{chunk['paper_title']}'):\n"
        passages += chunk["text"][:800]
        passages += "\n"

    # add contradiction warning if detected
    contradiction_warning = ""
    if contradiction_detected:
        contradiction_warning = f"\nWARNING: The sources contain a conflict: {contradiction_note}\nMake sure to represent both positions in your answer.\n"

    prompt = (
        f"Question: {user_query}\n"
        f"{contradiction_warning}"
        f"\nPassages:\n{passages}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # split answer text from the JSON metadata block
    final_answer = raw
    citations = []
    confidence_score = 0.5  # default if parsing fails

    try:
        if "{" in raw:
            # find the last JSON block in the response
            json_start = raw.rfind("{")
            json_end = raw.rfind("}") + 1
            json_str = raw[json_start:json_end]
            metadata = json.loads(json_str)
            confidence_score = float(metadata.get("confidence", 0.5))
            citations = metadata.get("citations", [])
            # remove the JSON block from the answer text
            final_answer = raw[:json_start].strip()
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "final_answer": final_answer,
        "citations": citations,
        "confidence_score": confidence_score
    }