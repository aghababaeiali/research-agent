from typing import TypedDict, Annotated

def _replace(old, new):
    return new

class AgentState(TypedDict):
    user_query:               Annotated[str,        _replace]
    search_queries:           Annotated[list[str],  _replace]
    candidate_papers:         Annotated[list[dict], _replace]
    selected_papers:          Annotated[list[dict], _replace]
    paper_chunks:             Annotated[list[dict], _replace]
    relevant_chunks:          Annotated[list[dict], _replace]
    contradiction_detected:   Annotated[bool,       _replace]
    contradiction_note:       Annotated[str,        _replace]
    confidence_score:         Annotated[float,      _replace]
    final_answer:             Annotated[str,        _replace]
    citations:                Annotated[list[str],  _replace]
    search_iteration:         Annotated[int,        _replace]
    max_iterations:           Annotated[int,        _replace]
    retrieval_sufficient:     Annotated[bool,       _replace]
    session_id:               Annotated[str,        _replace]
    collection_name:          Annotated[str,        _replace]