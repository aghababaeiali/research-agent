from agent.nodes.relevance_filter import relevance_filter

MOCK_PAPERS = [
    {
        "id": "2305.00001",
        "title": "RAG vs Fine-tuning for Domain Adaptation",
        "abstract": "We compare retrieval augmented generation and fine-tuning approaches for adapting large language models to specialized domains. Our experiments show RAG achieves competitive performance with lower computational cost.",
        "pdf_url": "https://arxiv.org/pdf/2305.00001"
    },
    {
        "id": "2305.00002",
        "title": "History of Ancient Rome",
        "abstract": "This paper reviews the political structures of the Roman Empire from 27 BC to 476 AD, focusing on administrative reforms under Augustus.",
        "pdf_url": "https://arxiv.org/pdf/2305.00002"
    }
]

def test_relevant_paper_passes_filter():
    state = {
        "user_query": "Is RAG better than fine-tuning for domain adaptation?",
        "candidate_papers": [MOCK_PAPERS[0]]
    }
    result = relevance_filter(state)
    assert len(result["selected_papers"]) == 1
    assert result["selected_papers"][0]["relevance_score"] >= 6

def test_irrelevant_paper_blocked():
    state = {
        "user_query": "Is RAG better than fine-tuning for domain adaptation?",
        "candidate_papers": [MOCK_PAPERS[1]]
    }
    result = relevance_filter(state)
    assert len(result["selected_papers"]) == 0

def test_retrieval_sufficient_requires_two_papers():
    state = {
        "user_query": "Is RAG better than fine-tuning?",
        "candidate_papers": [MOCK_PAPERS[0]]
    }
    result = relevance_filter(state)
    assert result["retrieval_sufficient"] == False