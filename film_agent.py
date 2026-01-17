from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    decide_next_step,
    web_search_node,
    route_question,
)
from models import GraphState

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)
workflow.add_node("web_search", web_search_node)

# workflow.set_entry_point("retrieve")
workflow.set_conditional_entry_point(
    route_question,
    {
        "vectorstore": "retrieve",
        "web_search": "web_search",
        "general_chat": "generate",
    },
)

workflow.add_edge("web_search", "generate")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
workflow.add_edge("rewrite_query", "retrieve")  # Pętla powrotna
workflow.add_edge("generate", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    user_query = "Horror o mordercy w hokejowej osłonie twarzy"

    inputs = {
        "question": user_query,
        "synthesized_query": user_query,
        "retry_count": 0,
        "context": "",
        "is_relevant": "no",
        "chat_history": [HumanMessage(content=user_query)],
    }

    # config is required when using checkpointer
    config = {"configurable": {"thread_id": "1"}}

    print(f"--- Starting Agent for query: {user_query} ---")

    for event in app.stream(inputs, config=config):
        for node, values in event.items():
            print(f"--- Node '{node}' finished ---")
            if node == "generate":
                print("\n" + "=" * 50)
                print(f"FINAL ANSWER:\n{values['generation']}")
