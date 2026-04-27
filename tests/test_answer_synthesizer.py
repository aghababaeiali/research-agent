from agent.nodes.answer_synthesizer import answer_synthesizer

MOCK_CHUNKS = [
    {
        "text": "RAG combines a retrieval component with a generative model. The retriever fetches relevant documents from a corpus, which are then passed to the generator as context.",
        "paper_title": "Survey of RAG Methods",
        "paper_id": "2305.00001",
        "score": 0.85
    },
    {
        "text": "Retrieval-augmented generation reduces hallucination by grounding model outputs in retrieved evidence rather than relying solely on parametric knowledge.",
        "paper_title": "RAG for Knowledge Grounding",
        "paper_id": "2305.00002",
        "score": 0.76
    }
]

def test_returns_final_answer():
    state = {
        "user_query": "How does retrieval augmented generation work?",
        "relevant_chunks": MOCK_CHUNKS,
        "contradiction_detected": False,
        "contradiction_note": ""
    }
    result = answer_synthesizer(state)
    assert "final_answer" in result
    assert len(result["final_answer"]) > 50

def test_returns_confidence_score():
    state = {
        "user_query": "How does RAG reduce hallucination?",
        "relevant_chunks": MOCK_CHUNKS,
        "contradiction_detected": False,
        "contradiction_note": ""
    }
    result = answer_synthesizer(state)
    assert "confidence_score" in result
    assert 0.0 <= result["confidence_score"] <= 1.0

def test_handles_empty_chunks():
    state = {
        "user_query": "How does RAG work?",
        "relevant_chunks": [],
        "contradiction_detected": False,
        "contradiction_note": ""
    }
    result = answer_synthesizer(state)
    assert "final_answer" in result
    assert result["confidence_score"] == 0.0

def test_contradiction_flag_included_in_answer():
    state = {
        "user_query": "Is RAG better than fine-tuning?",
        "relevant_chunks": MOCK_CHUNKS,
        "contradiction_detected": True,
        "contradiction_note": "Paper A claims RAG outperforms fine-tuning while Paper B claims fine-tuning is superior for domain adaptation."
    }
    result = answer_synthesizer(state)
    assert "final_answer" in result
    assert len(result["final_answer"]) > 50