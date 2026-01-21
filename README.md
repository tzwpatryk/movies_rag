# Raport Końcowy: System Rekomendacji Filmowej z wykorzystaniem Agentic RAG i Wyszukiwania Hybrydowego

## 1. Skład grupy projektowej i podział zadań

### **Olga Śmichurska** (Data Science & Database Engineering)

- **Pozyskanie i eksploracyjna analiza zbioru danych TMDB.**
- **Czyszczenie danych:** usuwanie rekordów bez opisów, filtracja po liczbie głosów i długości tekstu.
- **Inżynieria cech:** stworzenie bogatego reprezentanta tekstowego filmu (`text_to_embed`) łączącego tytuł, opis, słowa kluczowe i metadane.
- **Generowanie embeddingów:**
- _Dense:_ Model `Qwen/Qwen3-Embedding-0.6B`.
- _Sparse:_ Algorytm `BM25 (FastEmbed)`.

- **Konfiguracja i populacja bazy wektorowej Qdrant.**

### **Patryk Danielewicz** (AI Engineering & Application Development)

- **Implementacja architektury agenta** w oparciu o **LangGraph** (graf stanów).
- **Logika biznesowa RAG:**
- _Query Analysis:_ Ekstrakcja filtrów metadanych (rok, gatunek) z języka naturalnego.
- _Hybrid Search & Reranking:_ Integracja Qdrant z modelem Cross-Encoder.
- _Self-Correction:_ Pętle weryfikacji i przepisywania zapytań.

- **Integracja z zewnętrznymi narzędziami** (DuckDuckGo Search) i modelem Llama 3 via Groq.
- **Budowa interfejsu użytkownika** w Streamlit.

---

## 2. Opis problemu do rozwiązania

Celem projektu było rozwiązanie ograniczeń klasycznych wyszukiwarek słów kluczowych oraz standardowych systemów RAG w domenie filmowej. Główne problemy adresowane przez rozwiązanie to:

1. **Niejednoznaczność zapytań:** Użytkownicy często łączą w jednym zdaniu filtry "twarde" (np. _"lata 90."_, _"horror"_) z "miękkimi" opisami fabuły (np. _"o kosmitach"_). Standardowy vector search często ignoruje twarde ograniczenia na rzecz podobieństwa semantycznego.
2. **Halucynacje LLM:** Modele językowe mają tendencję do zmyślania fabuły lub przypisywania filmów do nieistniejących aktorów/reżyserów.
3. **Brak aktualnej wiedzy:** Baza wektorowa jest statyczna, więc system nie znałby odpowiedzi na pytania o repertuar kinowy na bieżący tydzień.

---

## 3. Zbiór danych i przygotowanie (Preprocessing)

Wykorzystano zbiór **TMDB Movie Dataset** (wersja ~930k filmów, przefiltrowana do ~72k najbardziej jakościowych rekordów).

### Proces przygotowania danych (`embed_films.ipynb`):

- **Filtracja jakościowa:**
- Usunięcie filmów bez daty premiery.
- Odrzucenie filmów z bardzo krótkimi opisami (< 50 znaków) oraz małą liczbą głosów (`vote_count <= 10`), aby wyeliminować szum i filmy amatorskie.

- **Feature Engineering (`text_to_embed`):**
  Stworzono jeden spójny blok tekstu dla każdego filmu, który służy do wyszukiwania semantycznego. Dzięki temu model embeddingów "widzi" pełen kontekst filmu.
  Format:

```text
"Movie title: {Title} ({Year}). Original language: {Lang}. Genres: {Genres}. Plot summary: {Overview}. Keywords: {Tags}. Produced by: {Companies}..."

```

- **Baza Wektorowa:**
- Wykorzystano obraz Docker `qdrant/qdrant`.
- Kolekcja `movies_db_final` przechowuje wektory gęste (Qwen) oraz rzadkie (BM25) wraz z bogatym payloadem (rok, ocena, gatunki) umożliwiającym filtrowanie.

---

## 4. Zastosowane metody i architektura Agentic RAG

System opiera się na bibliotece **LangGraph**, tworząc cykliczny graf decyzyjny.

### A. Query Analysis & Metadata Extraction

Zamiast wrzucać surowe pytanie użytkownika do bazy, system wykorzystuje LLM (`query_analyzer`), aby rozbić je na:

- **Filtry twarde:** Rok (min/max), gatunek, ocena, język.
- **Zapytanie semantyczne:** Przetłumaczone na język angielski zdanie opisujące fabułę.

> **Przykład:** "Polska komedia z lat 90"
> **Filter:** `{country: PL, genre: Comedy, year: 1990-1999}`

### B. Hybrid Search z Rerankingiem

Wyszukiwanie odbywa się dwuetapowo:

1. **Retrieval:** Równoległe zapytanie Dense (znaczenie) i Sparse (słowa kluczowe) do Qdrant z nałożonymi filtrami metadanych.
2. **Reranking:** Wykorzystanie modelu `cross-encoder/ms-marco-MiniLM-L-6-v2` do precyzyjnej oceny top-k wyników pod kątem zgodności z pytaniem.

### C. Mechanizmy kontroli jakości (Zapobieganie halucynacjom)

Wdrożono zaawansowane techniki "Self-Correction" w celu zapewnienia wiarygodności:

- **Węzeł Sędziego (Grade Documents):** Po pobraniu filmów, wyspecjalizowany prompt (LLM Grader) ocenia, czy znalezione dokumenty są faktycznie relewantne do pytania. Zwraca decyzję binarną: `yes` lub `no`.
- **Pętla przepisywania (Rewrite-Retrieve Loop):** Jeśli Sędzia oceni wyniki jako nietrafne (np. użytkownik pytał o horror, a baza zwróciła komedie przez błąd w zapytaniu), system nie generuje odpowiedzi. Zamiast tego:

1. Uruchamia węzeł `rewrite_query`, który przeformułowuje zapytanie na lepsze (np. usuwa zbędne przymiotniki, tłumaczy na angielski).
2. Ponawia wyszukiwanie w bazie (maksymalnie 3 próby).

- **Relaksacja filtrów (`relax_intent`):** W kodzie zaimplementowano logikę "Luzowania filtrów". Jeśli użytkownik poda zbyt restrykcyjne kryteria (np. _"Film akcji z 1995 roku z oceną 10.0"_) i baza zwróci 0 wyników, system automatycznie poszerza zakres lat (+/- 5 lat) i obniża wymaganą ocenę, aby znaleźć cokolwiek zbliżonego.

### D. Routing

LLM decyduje, czy pytanie wymaga:

- **Bazy wiedzy (Vector Store):** Pytania o fabułę, rekomendacje.
- **Wyszukiwania w sieci (DuckDuckGo):** Pytania o premiery "w tym roku", repertuar kin.
- **General Chat:** Pytania luźne, niezwiązane z filmami.

---

## 5. Wyniki i wnioski

Przeprowadzone eksperymenty wykazały skuteczność przyjętej architektury:

- **Przewaga Hybrid Search nad Dense Search:** Samo wyszukiwanie wektorowe (Dense) często gubiło się przy pytaniach o konkretne nazwy własne lub niszowe słowa kluczowe. Dodanie wektorów rzadkich (BM25) znacząco poprawiło trafność dla zapytań specyficznych (np. nazwiska reżyserów, unikalne rekwizyty).
- **Kluczowa rola filtrów metadanych:** Bez wstępnej analizy zapytania (Query Analysis), model wektorowy często zwracał filmy semantycznie podobne, ale z niewłaściwej epoki lub gatunku. "Uziemienie" zapytania w metadanych (rok, gatunek) okazało się najważniejszym czynnikiem sukcesu.
- **Skuteczność pętli Self-Correction:** W przypadku zapytań nieprecyzyjnych (np. _"ten film o gościu w masce"_), pierwsza iteracja często była błędna. Mechanizm Rewrite zazwyczaj w 2. turze "wcelowywał" w odpowiednie słowa kluczowe (np. dodając _"Jason Voorhees"_ lub _"Halloween"_), co eliminowało halucynacje w finalnej odpowiedzi.

**Rekomendacja:**
Najlepsze rezultaty osiąga metoda Agentic RAG z silnym naciskiem na ekstrakcję metadanych. Pozwala ona połączyć elastyczność LLM z precyzją bazy danych SQL-podobnej (Qdrant filtering), eliminując większość błędów merytorycznych typowych dla prostych systemów RAG.
