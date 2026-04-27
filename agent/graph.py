from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.query_planner import query_planner
from agent.nodes.arxiv_searcher import arxiv_searcher
from agent.nodes.relevance_filter import relevance_filter
from agent.nodes.paper_fetcher import paper_fetcher
from agent.nodes.chunk_retriever import chunk_retriever
from agent.nodes.contradiction_detector import contradiction_detector
from agent.nodes.answer_synthesizer import answer_synthesizer
from agent.nodes.evaluator import evaluator


def should_continue_searching(state: AgentState) -> str:
    if state["retrieval_sufficient"]:
        return "fetch_papers"
    if state["search_iteration"] >= state["max_iterations"]:
        return "fetch_papers"
    return "replan"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("query_planner", query_planner)
    graph.add_node("arxiv_searcher", arxiv_searcher)
    graph.add_node("relevance_filter", relevance_filter)
    graph.add_node("paper_fetcher", paper_fetcher)
    graph.add_node("chunk_retriever", chunk_retriever)
    graph.add_node("contradiction_detector", contradiction_detector)
    graph.add_node("answer_synthesizer", answer_synthesizer)
    graph.add_node("evaluator", evaluator)

    graph.set_entry_point("query_planner")

    graph.add_edge("query_planner", "arxiv_searcher")
    graph.add_edge("arxiv_searcher", "relevance_filter")

    graph.add_conditional_edges(
        "relevance_filter",
        should_continue_searching,
        {
            "fetch_papers": "paper_fetcher",
            "replan": "query_planner"
        }
    )

    graph.add_edge("paper_fetcher", "chunk_retriever")
    graph.add_edge("chunk_retriever", "contradiction_detector")
    graph.add_edge("contradiction_detector", "answer_synthesizer")
    graph.add_edge("answer_synthesizer", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


agent = build_graph()