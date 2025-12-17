from typing import Optional, List

from qdrant_client import models

from config import (
    client,
    query_analyzer,
    dense_model,
    reranker,
    sparse_model,
    COLLECTION_NAME,
)
from models import MovieSearchIntent


def rerank_qdrant_hits(
    query: str, hits: List[models.ScoredPoint], top_k: int = 5
) -> List[models.ScoredPoint]:
    """
    Funkcja bierze wyniki z Qdranta (hits), ocenia je rerankerem i zwraca najlepsze obiekty.
    """

    # print(hits)
    passages = [
        f"{hit.payload['title']} {hit.payload['overview']} {" ".join(hit.payload['keywords'])}"
        for hit in hits
    ]
    rerank_pairs = [[query, passage] for passage in passages]

    scores = reranker.predict(rerank_pairs)

    scored_hits = list(zip(hits, scores))
    scored_hits.sort(key=lambda x: x[1], reverse=True)

    return [hit for hit, score in scored_hits[:top_k]]


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
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20,
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
