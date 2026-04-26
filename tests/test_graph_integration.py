from agent.graph import agent

def test_graph_runs_end_to_end():
    initial_state = {
        "user_query": "How does retrieval augmented generation work?",
        "search_queries": [],
        "candidate_papers": [],
        "selected_papers": [],
        "paper_chunks": [],
        "relevant_chunks": [],
        "contradiction_detected": False,
        "contradiction_note": "",
        "confidence_score": 0.0,
        "final_answer": "",
        "citations": [],
        "search_iteration": 0,
        "max_iterations": 2,
        "retrieval_sufficient": False,
        "session_id": "test_session_001",
        "collection_name": ""
    }

    result = agent.invoke(initial_state)

    assert "relevant_chunks" in result
    assert isinstance(result["relevant_chunks"], list)
    assert result["search_iteration"] >= 1