from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

ragas_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

ragas_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def evaluator(state: dict) -> dict:
    user_query = state.get("user_query", "")
    final_answer = state.get("final_answer", "")
    relevant_chunks = state.get("relevant_chunks", [])
    confidence_score = state.get("confidence_score", 0.0)

    if not final_answer or not relevant_chunks:
        return state

    contexts = [chunk["text"] for chunk in relevant_chunks]

    eval_data = {
        "question": [user_query],
        "answer": [final_answer],
        "contexts": [contexts]
    }

    dataset = Dataset.from_dict(eval_data)

    try:
        scores = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy(strictness=1)],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=1)
        )

        df = scores.to_pandas()
        faithfulness_score = round(float(df["faithfulness"].mean()), 3)
        relevancy_score = round(float(df["answer_relevancy"].mean()), 3)

    except Exception as e:
        faithfulness_score = None
        relevancy_score = None
        print(f"RAGAS evaluation failed: {e}")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "confidence": confidence_score,
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score
    }

    with open("evaluation_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"RAGAS scores: faithfulness={faithfulness_score}, "
          f"answer_relevancy={relevancy_score}")

    return state