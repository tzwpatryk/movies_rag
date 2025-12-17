from langgraph.graph import StateGraph, END

from nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    decide_next_step,
)
from models import GraphState

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
workflow.add_edge("rewrite_query", "retrieve")  # Pętla powrotna
workflow.add_edge("generate", END)

app = workflow.compile()

user_query = "Horror o mordercy w hokejowej osłonie twarzy"

inputs = {
    "question": user_query,
    "original_question": user_query,
    "retry_count": 0,
    "context": "",
    "is_relevant": "no",
}

final_state = app.invoke(inputs)

print("\n" + "=" * 50)
print(f"FINAL ANSWER:\n{final_state['generation']}")
