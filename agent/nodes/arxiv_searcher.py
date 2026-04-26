import arxiv

def arxiv_searcher(state: dict) -> dict:
    search_queries = state.get("search_queries", [])
    current_iteration = state.get("search_iteration", 0)

    client = arxiv.Client()
    
    seen_ids = set()
    candidate_papers = []

    for query in search_queries:
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for result in client.results(search):
            if result.entry_id in seen_ids:
                continue
            seen_ids.add(result.entry_id)

            candidate_papers.append({
                "id": result.entry_id,
                "title": result.title,
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "authors": [a.name for a in result.authors[:3]],
                "published": str(result.published.date())
            })
    
    return{
        "candidate_papers": candidate_papers,
        "search_iteration": current_iteration + 1
    }