from typing import List

from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

from pydantic import BaseModel, Field
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv

COLLECTION_NAME = "movies_db"

load_dotenv()

dense_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, device="mps"
)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps")

client = QdrantClient(url="http://localhost:6333")
llm_translator = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=80)
llm_generator = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024
)

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


class MovieSearchIntent(BaseModel):
    """
    Struktura interpretacji pytania o film.
    """

    query_english: str = Field(
        ...,
        description="Temat fabuły przetłumaczony na angielski. Np. dla 'komedia o psach' -> 'funny dogs'. Zostaw przymiotniki (straszny, zabawny), usuń tylko techniczne określenia lat i oceny.",
    )
    genres: Optional[List[str]] = Field(
        None,
        description="Lista gatunków filmowych (np. Action, Comedy, Horror, Drama, Sci-Fi). Tylko standardowe nazwy angielskie.",
    )
    year_min: Optional[int] = Field(
        None, description="Minimalny rok premiery. Np. dla 'lata 90' = 1990."
    )
    year_max: Optional[int] = Field(
        None, description="Maksymalny rok premiery. Np. dla 'lata 90' = 1999."
    )
    min_score: Optional[float] = Field(
        None,
        description="Minimalna ocena (0-10). Jeśli użytkownik pisze 'dobry' lub 'polecany', ustaw 7.0. Jeśli 'wybitny'/'hit', ustaw 8.0.",
    )
    max_runtime: Optional[int] = Field(
        None,
        description="Maksymalny czas trwania w minutach. Jeśli użytkownik pisze 'krótki', ustaw 100. Jeśli 'bardzo krótki', ustaw 85.",
    )


router_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_text), ("human", "{query}")]
)

llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
query_analyzer = router_prompt | llm_router.with_structured_output(MovieSearchIntent)


def build_qdrant_filter(intent: MovieSearchIntent) -> Optional[models.Filter]:
    must_conditions = []

    if intent.year_min or intent.year_max:
        range_params = {}
        if intent.year_min:
            range_params["gte"] = intent.year_min
        if intent.year_max:
            range_params["lte"] = intent.year_max

        must_conditions.append(
            models.FieldCondition(key="year", range=models.Range(**range_params))
        )

    if intent.min_score:
        must_conditions.append(
            models.FieldCondition(
                key="vote_average", range=models.Range(gte=intent.min_score)
            )
        )

    if intent.max_runtime:
        must_conditions.append(
            models.FieldCondition(
                key="runtime", range=models.Range(lte=intent.max_runtime)
            )
        )

    if intent.genres:
        if len(intent.genres) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="genres", match=models.MatchValue(value=intent.genres[0])
                )
            )
        else:
            genre_should = [
                models.FieldCondition(
                    key="genres", match=models.MatchValue(value=genre)
                )
                for genre in intent.genres
            ]
            must_conditions.append(models.Filter(should=genre_should))

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


def rerank_qdrant_hits(
    query: str, hits: List[models.ScoredPoint], top_k: int = 5
) -> List[models.ScoredPoint]:
    """
    Funkcja bierze wyniki z Qdranta (hits), ocenia je rerankerem i zwraca najlepsze obiekty.
    """
    passages = [f"{hit.payload['title']} {hit.payload['overview']}" for hit in hits]
    rerank_pairs = [[query, passage] for passage in passages]

    scores = reranker.predict(rerank_pairs)

    scored_hits = list(zip(hits, scores))
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    return [hit for hit, score in scored_hits[:top_k]]


def retrieve_movies(query: str) -> List[str]:
    """
    Funkcja zamienia pytanie na intencję (filtry + temat), tworzy wektory
    i pyta Qdranta używając Hybrid Search z filtrowaniem metadanych.
    """

    print(f"\n🧠 Analizuję intencję zapytania: '{query}'...")

    intent = query_analyzer.invoke({"query": query})

    english_query = intent.query_english
    qdrant_filter = build_qdrant_filter(intent)

    print(f"   -> Temat (EN): '{english_query}'")
    print(f"   -> Wykryte filtry: {intent.model_dump(exclude={'query_english'})}")

    if not english_query:
        english_query = query

    print(f"\n🔍 Szukam w Qdrant (Hybrid + Filters)...")

    query_dense = dense_model.encode(english_query).tolist()

    raw_sparse_output = list(sparse_model.embed([english_query]))[0]
    query_sparse = models.SparseVector(
        indices=raw_sparse_output.indices.tolist(),
        values=raw_sparse_output.values.tolist(),
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="text-dense",
                filter=qdrant_filter,
                limit=50,
            ),
            models.Prefetch(
                query=query_sparse,
                using="text-sparse",
                filter=qdrant_filter,
                limit=50,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=10,
    )

    top_hits = rerank_qdrant_hits(english_query, results.points, top_k=5)

    formatted_docs = []
    for hit in top_hits:
        doc_content = (
            f"Title: {hit.payload['title']}\n"
            f"Year: {hit.payload['year']}\n"
            f"Genres: {hit.payload['genres']}\n"
            f"Rating: {hit.payload.get('vote_average', 'N/A')}\n"
            f"Overview: {hit.payload['overview']}\n"
            f"Score: {hit.score:.4f}\n"
            "---"
        )
        formatted_docs.append(doc_content)

    if not formatted_docs:
        return "Nie znaleziono filmów spełniających kryteria."

    return "\n\n".join(formatted_docs)


template = """Jesteś ekspertem filmowym. Odpowiedz na pytanie użytkownika na podstawie poniższych fragmentów filmów.
Jeśli w kontekście nie ma odpowiedzi, powiedz, że nie wiesz. Nie wymyślaj filmów spoza kontekstu.

KONTEKST (Znalezione filmy):
{context}

PYTANIE UŻYTKOWNIKA:
{question}

ODPOWIEDŹ:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": RunnableLambda(retrieve_movies), "question": RunnablePassthrough()}
    | prompt
    | llm_generator
    | StrOutputParser()
)

queries = [
    # "Jakiś horror z lat 90 o duchach?",
    # "Film gdzie gra są statki kosmiczne",
    # "Smutny klaun",
    # "Film z początku XXI wieku. Surrealistyczny, tajemniczy. Najlepiej dobrze oceniony przez krytyków.",
    "Niszowy film, najlepiej z lat 50-60. Najlepiej przybijający i niezrozumiały. Długi i doceniony przez krytyków."
]

for q in queries:
    print(f"\n🔹 PYTANIE: {q}")
    response = rag_chain.invoke(q)
    print(f"🔸 ODPOWIEDŹ: {response}")
    print("-" * 50)
