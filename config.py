from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from fastembed import SparseTextEmbedding
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import GradeDocuments, MovieSearchIntent

from dotenv import load_dotenv

load_dotenv()
COLLECTION_NAME = "movies_db_final"

# ===== MODELS =====

dense_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device="mps"
)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps")

client = QdrantClient(url="http://localhost:6333")

llm_grader = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_translator = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=80)
llm_generator = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024
)
llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=80)

# ===== GRADER =====

structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

system_grader_prompt = """Jesteś pomocnym asystentem, który weryfikuje trafność wyników wyszukiwania.

Twoim zadaniem jest sprawdzenie, czy znalezione filmy SĄ TEMATYCZNIE ZWIĄZANE z pytaniem.
Nie oceniaj daty ani oceny filmu (to zostało już przefiltrowane przez bazę danych).

Zasady:
1. Jeśli pytanie dotyczy "filmu wojennego", a w kontekście są filmy wojenne -> zwróć 'yes'.
2. Jeśli pytanie dotyczy "komedii", a w kontekście są horrory -> zwróć 'no'.
3. Bądź wyrozumiały. Jeśli wynik jest chociaż trochę pasujący -> zwróć 'yes'.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader_prompt),
        ("human", "PYTANIE: {question}\n\nKONTEKST:\n{context}"),
    ]
)

grader_chain = grade_prompt | structured_llm_grader


# ===== REWRITER =====

system_rewriter_prompt = """Jesteś ekspertem od wyszukiwania informacji (Google Search Expert).
Twoim zadaniem jest przekształcenie nieudanego zapytania użytkownika w IDEALNE zapytanie tekstowe (w języku angielskim), które zmaksymalizuje szansę znalezienia filmu w bazie wektorowej.

ZASADY:
1. Przetłumacz sens zapytania na angielski.
2. Skup się na SŁOWACH KLUCZOWYCH fabuły, tematu i gatunku (np. "hockey mask killer" -> "slasher Jason Voorhees summer camp horror").
3. NIE dodawaj filtrów jakościowych, jeśli użytkownik o nie nie prosił! (NIE dodawaj: "high rating", "best movies", "plot twists", "IMDB top 250"). To bardzo ważne - takie słowa mogą wykluczyć dobre filmy.
4. NIE zwracaj JSON-a. Zwróć tylko ciąg tekstu do wpisania w wyszukiwarkę.

Przykład:
Input: "Film o gościu co ucieka z więzienia rurą"
Output: "Prison escape movie sewage pipe redemption shawshank drama"
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_rewriter_prompt),
        (
            "human",
            "Oryginalne pytanie: {question}\nPoprzednie wyniki były błędne. Spróbuj sformułować lepsze zapytanie.",
        ),
    ]
)

# Tutaj używamy parsera tekstowego, bo chcemy po prostu string
rewriter_chain = rewrite_prompt | llm_translator | StrOutputParser()


system_prompt_text = """
Jesteś ekspertem wyszukiwarki filmowej. Twoim zadaniem jest przeanalizowanie pytania użytkownika (w języku polskim) 
i wyodrębnienie precyzyjnych filtrów oraz tematu wyszukiwania (w języku angielskim).

Zinterpretuj potrzeby użytkownika na podstawie JEGO OSTATNIEGO PYTANIA ORAZ HISTORII ROZMOWY.
Musisz zwrócić OBIEKT STANU, który zawiera łączne filtry (stare + nowe).

ZASADY ANALIZY HISTORII:
1. "State Management": Twoja odpowiedź musi zawierać KOMPLETNY zestaw filtrów potrzebny do wyszukania filmu w tym momencie.
   - PRZYKŁAD: 
     Tura 1: User="Chcę horror" -> Output: {{ "genres": ["Horror"], "year_min": null }}
     Tura 2: User="Coś nowszego" -> Output: {{ "genres": ["Horror"], "year_min": 2015 }} (Zauważ: gatunek Horror został zachowany!)
2. ZMIANY: Nadpisuj stare filtry tylko wtedy, gdy użytkownik wyraźnie o to prosi (np. "jednak wolę komedię" -> usuń Horror, dodaj Comedy).
3. KONTEKST: Jeśli użytkownik pisze "coś innego", "podobne", "nowsze" - musisz wiedzieć, do czego się odnosi, patrząc na historię.

ZASADY POLA 'synthesized_query':
- Stwórz jedno zdanie po angielsku, które łączy wszystkie aktywne filtry i temat. To jest Twoje "synthesized_query".

ZASADY FILTRÓW:
1. GATUNKI: Używaj standardowych nazw: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western.
   UWAGA! Dobieraj gatunki ostrożnie, możesz podać je w formie listy, jeśli zapytanie dotyczy większej liczby lub domyślasz się zbliżonych gatunków. Jeśli nie jesteś pewien - zostaw pole puste
2. SUBIEKTYWNOŚĆ:
   - "Dobry film" -> min_score: 7.0
   - "Bardzo dobry/Hit" -> min_score: 8.0
   - "Krótki film" -> max_runtime: 100
   - "Długi film" -> brak limitu (lub min_runtime, jeśli byśmy go mieli).
3. DATY:
   - "Lata 90" -> 1990-1999
   - "Po 2010" -> 2011-2025
   - "Stary" -> zazwyczaj przed 1980 (zależy od kontekstu, ale bądź rozsądny).


PYTANIE UŻYTKOWNIKA: {query}
"""
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_text),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
    ]
)

llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


query_analyzer = router_prompt | llm_router.with_structured_output(MovieSearchIntent)
