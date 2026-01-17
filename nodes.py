from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from models import GraphState
from utils import retrieve_movies
from config import grader_chain, rewriter_chain, llm_generator

template = """Jesteś ekspertem filmowym. Odpowiedz na pytanie użytkownika na podstawie poniższych fragmentów filmów.
Jeśli w kontekście nie ma odpowiedzi, powiedz, że nie wiesz. Nie wymyślaj filmów spoza kontekstu.

KONTEKST (Znalezione filmy):
{context}

PYTANIE UŻYTKOWNIKA:
{question}

ODPOWIEDŹ:"""

prompt = ChatPromptTemplate.from_template(template)


def retrieve_node(state: GraphState):
    query_to_use = state.get("synthesized_query") or state["question"]
    print(
        f"\n--- RETRIEVE: Szukam filmów dla: '{query_to_use} i historii {state['chat_history']}' ---"
    )
    documents = retrieve_movies(query_to_use, state["chat_history"])

    return {"context": documents}


def grade_documents_node(state: GraphState):
    print("--- CHECK: Sędzia ocenia wyniki... ---")
    question = state["synthesized_query"]
    context = state["context"]

    if "Nie znaleziono filmów" in context:
        print("   -> Pusty wynik z Qdranta.")
        return {"is_relevant": "no"}

    scored_result = grader_chain.invoke({"question": question, "context": context})
    print(f"   -> Decyzja: {scored_result.binary_score}")

    return {"is_relevant": scored_result.binary_score}


def rewrite_query_node(state: GraphState):
    print("--- REWRITE: Przepisuję zapytanie... ---")

    question = state["synthesized_query"]
    retry_count = state["retry_count"] + 1

    better_question = rewriter_chain.invoke({"question": question})

    print(f"   -> Nowe zapytanie (próba {retry_count}): '{better_question}'")

    return {"synthesized_query": better_question, "retry_count": retry_count}


def generate_node(state: GraphState):
    print("--- GENERATE: Generuję odpowiedź końcową... ---")

    generation_chain = prompt | llm_generator | StrOutputParser()
    response = generation_chain.invoke(
        {"context": state["context"], "question": state["synthesized_query"]}
    )

    return {"generation": response, "chat_history": [AIMessage(content=response)]}


def decide_next_step(state):
    if state["is_relevant"] == "yes":
        return "generate"
    else:
        if state["retry_count"] >= 3:
            print("--- MAX RETRIES: Poddaję się, generuję z tym co mam. ---")
            return "generate"
        return "rewrite_query"
