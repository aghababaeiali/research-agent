# Research Paper Q&A Agent

An agentic AI system that answers research questions by searching arXiv in real time, retrieving relevant paper chunks, detecting contradictions between sources, and generating grounded answers with inline citations.

**Live demo:** [huggingface.co/spaces/aliabbi/research-agent](https://huggingface.co/spaces/aliabbi/research-agent)

---

## What It Does

You ask a research question in natural language. The agent:

1. Reformulates your query into precise academic search terms
2. Searches arXiv live for relevant papers
3. Loops and reformulates if initial results are insufficient
4. Downloads and parses paper PDFs, falls back to abstracts if parsing fails
5. Chunks content and stores it in a session-isolated vector database
6. Retrieves the most semantically relevant passages
7. Detects contradictions between sources and flags them explicitly
8. Generates a grounded answer with inline citations and a confidence score
9. Evaluates the answer using RAGAS (faithfulness and answer relevancy)

The key distinction from a standard RAG pipeline: the agent makes decisions. It decides whether retrieved papers are relevant enough, loops back to reformulate queries when they are not, and explicitly surfaces conflicts between sources rather than blending them into a muddled answer.

---

## Architecture

The agent is built with LangGraph and consists of 8 nodes connected by directed edges with one conditional loop.

```
START
  --> query_planner         (reformulates user query into academic search terms)
  --> arxiv_searcher        (calls arXiv API, deduplicates results)
  --> relevance_filter      (LLM scores each abstract 0-10, keeps above threshold)
  --> [DECISION]
        sufficient?  YES --> paper_fetcher
        NO, iterations < 3 --> query_planner  (loop back, reformulate)
        NO, limit reached  --> paper_fetcher  (graceful exit)
  --> paper_fetcher         (downloads PDFs, parses, chunks, stores in ChromaDB)
  --> chunk_retriever       (per-paper similarity search, global re-ranking)
  --> contradiction_detector (flags conflicting claims between sources)
  --> answer_synthesizer    (generates answer with citations and confidence score)
  --> evaluator             (RAGAS metrics, logs to evaluation_log.jsonl)
  --> END
```

The conditional loop between `relevance_filter` and `query_planner` is the core agentic behavior. A standard RAG pipeline cannot do this because it has no decision-making layer.

---

## Engineering Decisions

### State Management

LangGraph 1.1.9 requires explicit reducer functions on all state fields. Without them, node outputs are not merged correctly across conditional edges. All state fields use a `_replace` reducer via Python's `Annotated` type hint:

```python
from typing import TypedDict, Annotated

def _replace(old, new):
    return new

class AgentState(TypedDict):
    user_query: Annotated[str, _replace]
    search_queries: Annotated[list[str], _replace]
    ...
```

### Chunking Strategy

Word-based chunking at 256 words with 30-word overlap. Token-based chunking was considered but word-based avoids a dependency on a specific tokenizer and stays within the 384-token limit of `all-mpnet-base-v2`. A references section cleaning step strips bibliography content before chunking to prevent citation lists from polluting retrieval results.

### Per-Paper Retrieval

Retrieving globally across all papers in one ChromaDB query causes volume bias: a paper with 200 chunks can dominate the top-K results simply by having more chunks, regardless of relevance. The solution is to retrieve top-3 per paper independently using ChromaDB's `where` metadata filter, then merge and globally re-rank by cosine similarity, keeping the final top-5.

```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    where={"paper_id": paper_id}  # isolate per paper
)
```

### Query Contextualization

Conversational questions and academic text live in different regions of embedding space. A question like "How does RAG work?" embeds differently from a paper chunk discussing RAG mechanisms. A domain prefix is generated dynamically by the LLM and prepended to the query before embedding, reducing the semantic gap:

```python
contextualized_query = f"{llm_generated_prefix}: {user_query}"
```

### Session Isolation

Each agent run creates a uniquely named ChromaDB collection using a UUID session ID. This prevents chunk contamination across concurrent users or multiple runs. Collections are named `session_{uuid}` and are created fresh each time.

### Batch Embedding

All chunks from a single paper are encoded in one `SentenceTransformer.encode()` call rather than one call per chunk. This takes advantage of the model's internal batching and parallel processing, significantly reducing embedding time for papers with many chunks.

### Graceful Degradation

PDF parsing fails silently on some papers due to multi-column layouts, embedded equations, and complex formatting. Rather than crashing, the system falls back to abstract-only mode and logs the `source` field in chunk metadata so downstream components know what quality of content they are working with.

### RAGAS Evaluation

The evaluator node is gated by a `run_evaluation` flag in the state. In the Gradio UI this flag is `False`, keeping response time acceptable. In notebook testing it is `True`. RAGAS runs with `max_workers=1` to stay within Groq's free tier token-per-minute limits.

---

## Evaluation Results

Evaluated on 10 diverse NLP and ML research questions across two runs using RAGAS with Llama 3.1-8b-instant as the judge and all-MiniLM-L6-v2 for embeddings.

| Metric | Score |
|--------|-------|
| Average Faithfulness | 0.921 |
| Average Answer Relevancy | 0.885 |
| Queries answered | 18 / 20 |
| Queries with no papers found | 2 / 20 |

Faithfulness measures whether answer claims are traceable to retrieved sources. A score of 0.921 means the agent almost never introduces information beyond what the papers explicitly state.

### Per-query breakdown (run 2)

| Query | Faithfulness | Answer Relevancy |
|-------|-------------|------------------|
| How does RAG work? | 1.000 | 0.668 |
| Limitations of fine-tuning LLMs? | 0.444 | 1.000 |
| How does LoRA reduce memory usage? | 1.000 | 1.000 |
| How do vector databases work? | 0.750 | 1.000 |
| Sparse vs dense retrieval? | 0.900 | 1.000 |
| How does knowledge distillation work? | 1.000 | 0.997 |
| Challenges in training LLMs? | 1.000 | 1.000 |
| How does attention work in transformers? | no papers found | - |
| BERT vs GPT architecture differences? | 1.000 | 0.996 |
| How does prompt engineering affect LLMs? | 1.000 | 0.566 |

---

## Known Limitations

**PDF parsing quality.** PyMuPDF struggles with multi-column layouts, embedded equations, and scanned PDFs. Extracted text can contain artifacts. The system handles this gracefully by falling back to abstracts, but abstract-only retrieval produces lower quality answers.

**Relevance filter false positives.** The LLM-based relevance scorer occasionally passes papers that apply a concept to an unrelated domain (image generation, security research) rather than explaining the mechanism the user asked about. This is a prompt engineering problem that improves iteratively but is not fully solved.

**Embedding model semantic gap.** `all-mpnet-base-v2` is a general-purpose sentence embedding model. It was not trained on academic text. The production fix is to use `allenai/specter2`, which was trained on scientific literature with asymmetric query and document adapters. This was scoped out in favor of shipping a working system.

**Query reformulation limitations.** For very specific or niche queries (LoRA memory usage, attention mechanisms), the query planner sometimes generates queries too broad to find directly relevant papers. 2 of 20 queries returned no papers.

**No persistent storage.** ChromaDB collections are session-scoped and ephemeral. Each query starts from scratch. A production system would persist paper embeddings by paper ID to avoid re-downloading and re-chunking papers that were already processed.

**Groq free tier rate limits.** The system makes 5-6 LLM calls per query. At Groq's free tier limit of 6000 tokens per minute, back-to-back queries can hit rate limits. The Gradio UI handles this with a retry loop.

**RAGAS evaluates coherence, not factual accuracy.** RAGAS measures whether the answer is faithful to retrieved context, not whether the retrieved context itself is correct. A confidently wrong paper would score perfectly on faithfulness.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph 1.1.9 |
| LLM inference | Groq (Llama 3.3-70b for generation, Llama 3.1-8b for routing) |
| Embeddings | sentence-transformers all-mpnet-base-v2 |
| Vector database | ChromaDB (session-scoped, in-memory) |
| PDF parsing | PyMuPDF (fitz) |
| Paper search | arXiv API |
| Evaluation | RAGAS (faithfulness, answer relevancy) |
| Framework | LangChain, LangChain-Groq, LangChain-HuggingFace |
| UI | Gradio |
| Deployment | Hugging Face Spaces (CPU free tier) |
| Testing | pytest (7 unit and integration tests) |

---

## Project Structure

```
research_agent/
├── agent/
│   ├── state.py                    # AgentState TypedDict with reducers
│   ├── graph.py                    # LangGraph graph with conditional edges
│   └── nodes/
│       ├── query_planner.py        # LLM query reformulation
│       ├── arxiv_searcher.py       # arXiv API integration
│       ├── relevance_filter.py     # LLM-based abstract scoring
│       ├── paper_fetcher.py        # PDF download, parse, chunk, embed
│       ├── chunk_retriever.py      # Per-paper retrieval, global re-rank
│       ├── contradiction_detector.py  # Cross-source conflict detection
│       ├── answer_synthesizer.py   # Grounded answer generation
│       └── evaluator.py            # RAGAS metrics and logging
├── tests/
│   ├── test_query_planner.py
│   ├── test_arxiv_searcher.py
│   ├── test_relevance_filter.py
│   ├── test_answer_synthesizer.py
│   └── test_graph_integration.py
├── app.py                          # Gradio interface
├── requirements.txt
└── .env                            # GROQ_API_KEY (not committed)
```

---

## Running Locally

```bash
git clone https://github.com/aghababaeiali/research-agent.git
cd research-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```
GROQ_API_KEY=your_key_here
```

Run the Gradio UI:

```bash
python app.py
```

Run tests:

```bash
pytest tests/ -v
```

---

## Planned Improvements

- Replace `all-mpnet-base-v2` with `allenai/specter2` for domain-specific academic embeddings with asymmetric query and document encoding
- Add cross-encoder reranking (sentence-transformers CrossEncoder) as a second retrieval stage after cosine similarity
- Persist ChromaDB collections by paper ID to avoid re-processing already-seen papers
- Add LangSmith tracing for production observability
- Implement token counting per node to manage context window proactively

---

## About

Built by **Ali Aghababaei**, NLP and Generative AI Engineer based in Rotterdam, Netherlands. Orientation year visa holder, authorized to work without sponsorship.

BSc Electrical Engineering, MSc ICT at University of Padova (110/110, 2025). 2.5 years industry experience as Technical Expert. Published researcher in explainable NLP.

[GitHub](https://github.com/aghababaeiali) | [LinkedIn](https://linkedin.com/in/aliaghababaeii) | [Portfolio](https://aghababaeiali.github.io)
