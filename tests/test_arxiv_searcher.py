from agent.nodes.arxiv_searcher import arxiv_searcher

def test_returns_candidate_papers():
    state={
        "search_queries": ["retrieval augmented generation"],
        "search_iteration": 0
    }
    result = arxiv_searcher(state)
    assert "candidate_papers" in result
    assert isinstance(result["candidate_papers"], list)
    assert len(result["candidate_papers"]) > 0

def test_paper_has_required_fields():
    state = {
        "search_queries": ["large language models fine-tuning"],
        "search_iteration": 0
    }
    result = arxiv_searcher(state)
    paper = result["candidate_papers"][0]
    for field in ["id", "title", "abstract", "pdf_url"]:
        assert field in paper

def test_deduplications():
    state = {
        "search_queries": ["RAG retrieval augmented generation",
                           "retrieval augmented generation NLP"],
        "search_iteration": 0
    }
    result = arxiv_searcher(state)
    ids = [p["id"] for p in result["candidate_papers"]]
    assert len(ids) == len(set(ids))

def test_iteration_increments():
    state = {
    "search_queries": ["transformer attention"],
    "search_iteration": 1
    }
    result = arxiv_searcher(state)
    assert result["search_iteration"] == 2
