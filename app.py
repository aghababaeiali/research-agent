import gradio as gr
from agent.graph import agent


def run_agent(user_query: str) -> tuple[str, str, str, str]:
    if not user_query.strip():
        return "Please enter a question.", "", "", ""

    initial_state = {
        "user_query": user_query,
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
        "session_id": "",
        "collection_name": "",
        "run_evaluation": False
    }

    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        return f"Agent error: {str(e)}", "", "", ""

    answer = result.get("final_answer", "No answer generated.")
    confidence = result.get("confidence_score", 0.0)
    citations = result.get("citations", [])
    contradiction_detected = result.get("contradiction_detected", False)
    contradiction_note = result.get("contradiction_note", "")

    # format citations
    if citations:
        citations_text = "\n".join([f"- {c}" for c in citations])
    else:
        citations_text = "No citations available."

    # format confidence
    confidence_text = f"{confidence:.0%}"

    # format contradiction
    if contradiction_detected:
        contradiction_text = f"Conflict detected: {contradiction_note}"
    else:
        contradiction_text = "No conflicts detected between sources."

    return answer, citations_text, confidence_text, contradiction_text


with gr.Blocks(title="Research Paper Q&A Agent") as demo:

    gr.Markdown("# Research Paper Q&A Agent")
    gr.Markdown(
        "Ask a research question. The agent searches arXiv, retrieves "
        "relevant papers, detects contradictions between sources, and "
        "generates a grounded answer with citations."
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research question",
                placeholder="e.g. How does retrieval augmented generation work?",
                lines=2
            )
            submit_btn = gr.Button("Search and Answer", variant="primary")

        with gr.Column(scale=1):
            confidence_output = gr.Textbox(
                label="Confidence score",
                interactive=False
            )
            contradiction_output = gr.Textbox(
                label="Source conflicts",
                interactive=False
            )

    answer_output = gr.Textbox(
        label="Answer",
        lines=10,
        interactive=False
    )

    citations_output = gr.Textbox(
        label="Sources",
        lines=4,
        interactive=False
    )

    gr.Markdown(
        "Note: Each query takes 30-90 seconds. "
        "The agent searches live arXiv papers."
    )

    submit_btn.click(
        fn=run_agent,
        inputs=[query_input],
        outputs=[answer_output, citations_output,
                 confidence_output, contradiction_output]
    )

    gr.Examples(
        examples=[
            ["How does retrieval augmented generation work?"],
            ["What are the limitations of fine-tuning large language models?"],
            ["How does LoRA reduce the cost of fine-tuning?"],
            ["What is the difference between sparse and dense retrieval?"]
        ],
        inputs=query_input
    )


if __name__ == "__main__":
    demo.launch()