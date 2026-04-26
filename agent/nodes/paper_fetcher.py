import fitz
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid

load_dotenv()

embedding_model = SentenceTransformer("all-mpnet-base-v2")
chroma_client = chromadb.Client()


def _download_and_parse_pdf(pdf_url: str) -> str:
    try:
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception:
        return ""

def _clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    skip = False
    for line in lines:
        # detect start of references section
        if line.strip().lower() in ["references", "bibliography", "works cited"]:
            skip = True
        if not skip:
            cleaned.append(line)
    return "\n".join(cleaned)

def _chunk_text(text: str, chunk_size: int = 256,
                overlap: int = 30) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def paper_fetcher(state: dict) -> dict:
    selected_papers = state.get("selected_papers", [])
    session_id = state.get("session_id") or str(uuid.uuid4())


    collection_name = f"session_{session_id}"
    collection = chroma_client.get_or_create_collection(name=collection_name)

    all_chunks = []

    for paper in selected_papers:
        paper_id = paper["id"].split("/")[-1]

        full_text = _download_and_parse_pdf(paper["pdf_url"])
        source = "full_text" if full_text else "abstract_only"
        text_to_chunk = full_text if full_text else paper["abstract"]
        text_to_chunk = _clean_text(text_to_chunk) 

        chunks = _chunk_text(text_to_chunk)

        # batch encode all chunks for this paper in one forward pass
        chunk_embeddings = embedding_model.encode(chunks).tolist()

        chunk_ids = []
        chunk_metadatas = []

        for i, chunk_text in enumerate(chunks):
            chunk_ids.append(f"{paper_id}_chunk_{i}")
            chunk_metadatas.append({
                "paper_id": paper_id,
                "paper_title": paper["title"],
                "chunk_index": i,
                "source": source
            })
            all_chunks.append({
                "chunk_id": f"{paper_id}_chunk_{i}",
                "paper_id": paper_id,
                "paper_title": paper["title"],
                "text": chunk_text,
                "source": source
            })

        # one collection.add() call per paper, not per chunk
        collection.add(
            documents=chunks,
            embeddings=chunk_embeddings,
            ids=chunk_ids,
            metadatas=chunk_metadatas
        )

    return {
        "paper_chunks": all_chunks,
        "session_id": session_id,
        "collection_name": collection_name
    }