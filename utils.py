from typing import Optional, List
from langchain_core.messages import BaseMessage

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
    query: str,
    hits: List[models.ScoredPoint],
    specific_title: Optional[str] = None,
    top_k: int = 5,
) -> List[models.ScoredPoint]:
    """
    Funkcja bierze wyniki z Qdranta (hits), ocenia je rerankerem i zwraca najlepsze obiekty.
    """
    passages = []
    for hit in hits:
        title = hit.payload.get("title", "")
        # orig_title = hit.payload.get("original_title", "")
        overview = hit.payload.get("overview", "")
        tagline = hit.payload.get("tagline", "")
        keywords = ", ".join(hit.payload.get("keywords", []))

        passage = f"{title} {tagline} {overview} {keywords}"
        passages.append(passage)

    rerank_pairs = [[query, passage] for passage in passages]

    scores = reranker.predict(rerank_pairs)

    scored_hits = list(zip(hits, scores))

    if specific_title:
        boosted_hits = []
        for hit, score in scored_hits:
            title = hit.payload.get("title", "").lower()
            target = specific_title.lower()
            is_match = target == title or target in title
            final_score = score + 10.0 if is_match else score
            boosted_hits.append((hit, final_score))

        scored_hits = boosted_hits

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

    if intent.production_companies:
        if len(intent.production_companies) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="production_companies",
                    match=models.MatchValue(value=intent.production_companies[0]),
                )
            )
        else:
            p_comp_should = [
                models.FieldCondition(
                    key="production_companies", match=models.MatchValue(value=p_comp)
                )
                for p_comp in intent.production_companies
            ]
            must_conditions.append(models.Filter(should=p_comp_should))

    if intent.production_countries:
        if len(intent.production_countries) == 1:
            must_conditions.append(
                models.FieldCondition(
                    key="production_countries",
                    match=models.MatchValue(value=intent.production_countries[0]),
                )
            )
        else:
            p_count_should = [
                models.FieldCondition(
                    key="production_countries", match=models.MatchValue(value=p_count)
                )
                for p_count in intent.production_countries
            ]
            must_conditions.append(models.Filter(should=p_count_should))

    if intent.original_language:
        must_conditions.append(
            models.FieldCondition(
                key="original_language",
                match=models.MatchValue(value=intent.original_language),
            )
        )

    if intent.min_vote_count:
        must_conditions.append(
            models.FieldCondition(
                key="vote_count", range=models.Range(gte=intent.min_vote_count)
            )
        )

    if intent.include_adult is False:
        must_conditions.append(
            models.FieldCondition(key="adult", match=models.MatchValue(value=False))
        )

    if not must_conditions:
        return None

    return models.Filter(must=must_conditions)


def run_qdrant_search(
    english_query: str, qdrant_filter: Optional[models.Filter], limit: int = 20
):
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
        limit=limit,
    )
    return results.points


def relax_intent(intent: MovieSearchIntent) -> MovieSearchIntent:
    new_intent = intent.model_copy()
    if new_intent.min_score:
        new_intent.min_score = max(0.0, new_intent.min_score - 1.5)

    if new_intent.year_min:
        new_intent.year_min = new_intent.year_min - 5
    if new_intent.year_max:
        new_intent.year_max = new_intent.year_max + 5

    if new_intent.max_runtime:
        new_intent.max_runtime = None

    if new_intent.min_vote_count:
        new_intent.min_vote_count = None

    if new_intent.original_language:
        new_intent.original_language = None

    return new_intent


def retrieve_movies(query: str, chat_history: List[BaseMessage] = []) -> List[str]:
    """
    Zwraca: (sformatowane_dokumenty, zsyntezowane_zapytanie_angielskie)
    """

    print(f"\n🧠 Analizuję intencję zapytania: '{query}'...")

    MAX_HISTORY_LENGHT = 6

    if len(chat_history) > MAX_HISTORY_LENGHT:
        chat_history_for_llm = chat_history[-MAX_HISTORY_LENGHT:]
    else:
        chat_history_for_llm = chat_history

    intent = query_analyzer.invoke(
        {"query": query, "chat_history": chat_history_for_llm}
    )
    english_query = intent.synthesized_query

    if intent.specific_title:
        print(f"Wykryto konkretny film: {intent.specific_title}")
        english_query = f"{english_query} | Movie title: {intent.specific_title}"

    print(f"\n Obecne zsyntezowane zapytanie: '{english_query}'")

    qdrant_filter = build_qdrant_filter(intent)
    active_filters = {
        k: v
        for k, v in intent.model_dump().items()
        if v is not None and k not in ["query_english", "synthesized_query"]
    }

    print(f"   -> Temat (EN): '{english_query}'")
    print(f"   -> Wykryte filtry: {active_filters}")

    if not english_query:
        english_query = query

    print(f"\n🔍 Szukam w Qdrant (Hybrid + Filters)...")

    hits = run_qdrant_search(english_query, qdrant_filter)
    top_hits = rerank_qdrant_hits(english_query, hits, intent.specific_title, top_k=5)

    filters_info = ""

    if len(top_hits) < 3:
        print("\n⚠️  Mało wyników. Uruchamiam 'Lekkie Luzowanie' filtrów...")
        relaxed_intent = relax_intent(intent)
        relaxed_filter = build_qdrant_filter(relaxed_intent)

        active_relaxed = {
            k: v
            for k, v in relaxed_intent.model_dump().items()
            if v is not None and k not in ["query_english", "synthesized_query"]
        }
        print(f"   -> Nowe filtry (Relaxed): {active_relaxed}")

        relaxed_hits = run_qdrant_search(english_query, relaxed_filter)
        relaxed_top_hits = rerank_qdrant_hits(english_query, relaxed_hits, top_k=5)

        if len(relaxed_top_hits) > len(top_hits):
            top_hits = relaxed_top_hits
            filters_info = (
                "UWAGA DLA MODELU: Nie znaleziono idealnych dopasowań dla ścisłych filtrów (np. konkretny rok czy wysoka ocena). "
                "Filtry zostały lekko poluzowane (rozszerzono zakres lat lub obniżono minimalną ocenę), "
                "aby znaleźć najbardziej zbliżone filmy. Poinformuj o tym użytkownika.\n\n"
            )
        else:
            print("   -> Luzowanie nie pomogło (nadal brak wyników).")

    formatted_docs = []
    for hit in top_hits:
        p = hit.payload
        genres = ", ".join(p.get("genres", []))
        keywords = ", ".join(p.get("keywords", []))
        production = ", ".join(p.get("production_companies", []))
        countries = ", ".join(p.get("production_countries", []))
        doc_content = (
            f"Title: {p.get('title')}\n"
            f"Original Title: {p.get('original_title')}\n"
            f"Year: {p.get('year')}\n"
            f"Genres: {genres}\n"
            f"Language: {p.get('original_language')} (Spoken: {p.get('spoken_languages')})\n"
            f"Origin: {countries}\n"
            f"Production: {production}\n"
            f"Rating: {p.get('vote_average', 'N/A')} (Votes: {p.get('vote_count')})\n"
            f"Runtime: {p.get('runtime')} min\n"
            f"Tagline: {p.get('tagline')}\n"
            f"Keywords: {keywords}\n"
            f"Overview: {p.get('overview')}\n"
            f"Relevance Score: {hit.score:.4f}\n"
            "---"
        )
        formatted_docs.append(doc_content)

    if not formatted_docs:
        return "Nie znaleziono filmów spełniających kryteria.", english_query

    return filters_info + "\n\n".join(formatted_docs), english_query
