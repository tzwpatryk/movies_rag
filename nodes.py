from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_community.tools import DuckDuckGoSearchRun

from models import GraphState, RouteQuery
from utils import retrieve_movies
from config import grader_chain, rewriter_chain, llm_generator, llm_router

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
    documents, synthesized_query = retrieve_movies(query_to_use, state["chat_history"])

    return {"context": documents, "synthesized_query": synthesized_query}


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

    context = state.get("context", "")
    question = state["question"]

    # 1. TRYB CHAT (Brak kontekstu z bazy/internetu -> luźna rozmowa)
    if not context:
        print("   -> Tryb: General Chat")
        chat_template = """Jesteś ekspertem filmowym. Rozmawiaj swobodnie z użytkownikiem. 
        Bądź pomocny, uprzejmy i wykazuj się wiedzą o kinie, ale nie zmyślaj faktów.
        
        PYTANIE UŻYTKOWNIKA: {question}
        """
        chat_prompt = ChatPromptTemplate.from_template(chat_template)

        chat_chain = chat_prompt | llm_generator | StrOutputParser()
        response = chat_chain.invoke({"question": question})

    else:
        print("   -> Tryb: RAG / Context QA")
        rag_chain = prompt | llm_generator | StrOutputParser()
        query_to_use = state.get("synthesized_query") or question

        response = rag_chain.invoke({"context": context, "question": query_to_use})

    return {"generation": response, "chat_history": [AIMessage(content=response)]}


def decide_next_step(state):
    if state["is_relevant"] == "yes":
        return "generate"
    else:
        if state["retry_count"] >= 3:
            print("--- MAX RETRIES: Poddaję się, generuję z tym co mam. ---")
            return "generate"
        return "rewrite_query"


def route_question(state):
    print("--- ROUTE QUESTION ---")
    question = state["question"]

    system = """Jesteś ekspertem kierującym ruchem w asystencie filmowym.
    - Jeśli użytkownik prosi o rekomendację filmu, szuka fabuły, gatunku -> 'vectorstore'.
    - Jeśli pyta o aktualności, box office, premiery z tego roku, repertuar kin -> 'web_search'.
    - Jeśli użytkownik pyta o aktorów lub reżyserów lub jeśli to powitanie, pytanie o wiedzę ogólną (np. 'Kim jest Nolan?'), podziękowanie -> 'general_chat'.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    router = prompt | llm_router.with_structured_output(RouteQuery)
    decision = router.invoke({"question": question})

    return decision.destination


def web_search_node(state):
    print("--- WEB SEARCH ---")
    question = state["question"]

    search = DuckDuckGoSearchRun()
    results = search.invoke(question)

    return {"context": results, "is_relevant": "yes"}
