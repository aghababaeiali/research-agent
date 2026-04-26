from agent.nodes.query_planner import query_planner

def test_returns_list_of_strings():
    state = {"user_query": "Is RAG better than fine-tuning for domain adaptation?"}
    result = query_planner(state)
    assert "search_queries" in result
    assert isinstance(result["search_queries"], list)
    assert len(result["search_queries"]) >= 1
    assert all(isinstance (q ,str) for q in result["search_queries"])

def test_queries_are_not_empty():
    state = {"user_query": "How does attention work in transformers?"}
    result = query_planner(state)
    assert all(len(q.strip()) > 0 for q in result["search_queries"])

def test_handles_short_query():
    state = {"user_query": "LLMs"}
    result = query_planner(state)
    assert "search_queries" in result
    assert len(result["search_queries"]) >= 1

