import chromadb
import os
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.Client()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)


def _contextualize_query(user_query: str) -> str:
    messages = [
        SystemMessage(content="""Given a research question, produce a 
3-5 word academic domain prefix that describes the research area.
Respond ONLY with the prefix, nothing else.
Example input: "How does attention work in transformers?"
Example output: "transformer attention mechanisms research" """),
        HumanMessage(content=user_query)
    ]
    response = llm.invoke(messages)
    prefix = response.content.strip()
    return f"{prefix}: {user_query}"


def chunk_retriever(state: dict) -> dict:
    user_query = state.get("user_query", "")
    selected_papers = state.get("selected_papers", [])
    collection_name = state.get("collection_name", "")

    collection = chroma_client.get_collection(name=collection_name)
    contextualized_query = _contextualize_query(user_query)
    query_embedding = embedding_model.encode(contextualized_query).tolist()

    all_results = []

    for paper in selected_papers:
        paper_id = paper["id"].split("/")[-1]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"paper_id": paper_id}
        )

        for i, doc in enumerate(results["documents"][0]):
            all_results.append({
                "text": doc,
                "paper_id": paper_id,
                "paper_title": paper["title"],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = all_results[:5]

    return {"relevant_chunks": top_chunks}