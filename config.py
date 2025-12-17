from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from fastembed import SparseTextEmbedding
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models import GradeDocuments, MovieSearchIntent

from dotenv import load_dotenv

load_dotenv()
COLLECTION_NAME = "movies_db"

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
structured_llm_grader = llm_grader.with_structured_output(GradeDocuments)

system_grader_prompt = """Jesteś surowym krytykiem. Oceniasz, czy znalezione fragmenty filmów (KONTEKST) 
zawierają informacje potrzebne do odpowiedzi na pytanie użytkownika.

Jeśli filmy pasują tematycznie lub zawierają odpowiedź -> zwróć 'yes'.
Jeśli filmy są zupełnie nie na temat (np. pytanie o horror, a filmy to komedie) -> zwróć 'no'.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_grader_prompt),
        ("human", "PYTANIE: {question}\n\nKONTEKST:\n{context}"),
    ]
)

grader_chain = grade_prompt | structured_llm_grader

system_rewriter_prompt = """Jesteś ekspertem od wyszukiwania.
Twoim zadaniem jest przetłumaczenie pytania na język angielski (query_english) i wyciągnięcie jawnych filtrów.

BARDZO WAŻNE ZASADY DOTYCZĄCE GATUNKÓW:
1. NIE ZGADUJ gatunku. Jeśli użytkownik nie napisał wprost "horror", "komedia", "film akcji" - pole 'genres' ma zostać PUSTE (None).
2. Opis fabuły ("facet ucieka", "strzelanina") to NIE jest gatunek. To jest temat (query_english).
3. Lepiej nie dać żadnego filtra gatunkowego, niż dać błędny i ukryć właściwy film.
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

ZASADY:
1. TŁUMACZENIE: Temat (query_english) musi być po angielsku, ale NIE może zawierać słów, które trafiły do filtrów (np. nie wpisuj 'horror', jeśli dodałeś to do genres).
2. GATUNKI: Używaj standardowych nazw: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western.
   UWAGA! Dobieraj gatunki ostrożnie, możesz podać je w formie listy, jeśli zapytanie dotyczy większej liczby lub domyślasz się zbliżonych gatunków. Jeśli nie jesteś pewien - zostaw pole puste
3. SUBIEKTYWNOŚĆ:
   - "Dobry film" -> min_score: 7.0
   - "Bardzo dobry/Hit" -> min_score: 8.0
   - "Krótki film" -> max_runtime: 100
   - "Długi film" -> brak limitu (lub min_runtime, jeśli byśmy go mieli).
4. DATY:
   - "Lata 90" -> 1990-1999
   - "Po 2010" -> 2011-2025
   - "Stary" -> zazwyczaj przed 1980 (zależy od kontekstu, ale bądź rozsądny).

PYTANIE UŻYTKOWNIKA: {query}
"""
router_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_text), ("human", "{query}")]
)

llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


query_analyzer = router_prompt | llm_router.with_structured_output(MovieSearchIntent)
