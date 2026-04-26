from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.query_planner import query_planner
from agent.nodes.arxiv_searcher import arxiv_searcher
from agent.nodes.relevance_filter import relevance_filter
from agent.nodes.paper_fetcher import paper_fetcher
from agent.nodes.chunk_retriever import chunk_retriever


def should_continue_searching(state: AgentState) -> str:
    if state["retrieval_sufficient"]:
        return "fetch_papers"
    if state["search_iteration"] >= state["max_iterations"]:
        return "fetch_papers"  # graceful exit with whatever we have
    return "replan"


def build_graph():
    graph = StateGraph(AgentState)

    # register nodes
    graph.add_node("query_planner", query_planner)
    graph.add_node("arxiv_searcher", arxiv_searcher)
    graph.add_node("relevance_filter", relevance_filter)
    graph.add_node("paper_fetcher", paper_fetcher)
    graph.add_node("chunk_retriever", chunk_retriever)

    # entry point
    graph.set_entry_point("query_planner")

    # linear edges
    graph.add_edge("query_planner", "arxiv_searcher")
    graph.add_edge("arxiv_searcher", "relevance_filter")

    # conditional edge: the loop
    graph.add_conditional_edges(
        "relevance_filter",
        should_continue_searching,
        {
            "fetch_papers": "paper_fetcher",
            "replan": "query_planner"
        }
    )

    # rest of the pipeline (Week 3 nodes connect here later)
    graph.add_edge("paper_fetcher", "chunk_retriever")
    graph.add_edge("chunk_retriever", END)

    return graph.compile()


agent = build_graph()