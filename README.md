**System Rekomendacji Filmowej z wykorzystaniem Agentic RAG i Wyszukiwania Hybrydowego**

## Skład grupy projektowej i podział zadań

- **Olga Śmichurska**
  - Wyszukanie i selekcja odpowiedniego zbioru danych filmowych (TMDB Dataset).
  - Obróbka i czyszczenie danych.
  - Dobór zmiennych i inżynieria cech pod kątem wyszukiwania.
  - Stworzenie i przetworzenie kolumn z embeddingami:
    - Generowanie embeddingów gęstych (Dense) przy użyciu modelu `Qwen`.
    - Generowanie wektorów rzadkich (Sparse) dla algorytmu BM25.
  - Konfiguracja bazy wektorowej **Qdrant**

- **Patryk Danielewicz** (Budowa Agenta AI i Aplikacja)
  - Projekt architektury agenta w oparciu o **LangGraph** (graf stanów, węzły decyzyjne).
  - Implementacja węzłów funkcjonalnych:
    - **Retrieve:** Integracja z Qdrant i mechanizm Hybrid Search.
    - **Grade & Rewrite:** Logika "Sędziego" (ocena trafności) i pętla autokorekty zapytań.
    - **Routing:** Inteligentne kierowanie pytań (baza vs. web search vs. chat).
  - Wdrożenie mechanizmu Rerankingu poprawiającego precyzję wyników.
  - Stworzenie interfejsu użytkownika w technologii **Streamlit**.

## Opis planowanych prac

Celem projektu jest stworzenie zaawansowanego asystenta filmowego, który rozwiązuje problemy klasycznych wyszukiwarek poprzez zastosowanie architektury **Agentic RAG**. System nie tylko wyszukuje filmy, ale aktywnie "rozumie" potrzeby użytkownika, potrafi dopytać o szczegóły lub samodzielnie poprawić swoje zapytanie, jeśli pierwsze wyniki są niesatysfakcjonujące.

Głównym elementem prac jest budowa grafu, który integruje bazę wektorową z modelem językowym. Agent będzie analizował historię rozmowy, wyodrębniał filtry (rok, gatunek) z języka naturalnego oraz decydował o strategii wyszukiwania. W przypadku braku trafnych wyników w bazie, system automatycznie przeformułuje zapytanie na język angielski i ponowi próbę (pętla _Self-Correction_). Dodatkowo, dla zapytań o bieżący repertuar, agent skorzysta z wyszukiwania internetowego.

### Opis problemu do rozwiązania

Standardowe systemy rekomendacji często zawodzą przy zapytaniach nieprecyzyjnych lub opisowych. Użytkownicy oczekują interakcji w języku naturalnym, gdzie mogą łączyć "miękkie" opisy fabuły z "twardymi" filtrami (np. "tylko filmy po 2010 roku"). Projekt rozwiązuje problem łączenia tych dwóch światów oraz eliminuje halucynacje modeli LLM poprzez uziemienie odpowiedzi w faktycznej bazie danych (RAG). Do tego posiada dostęp do filmów bardziej niszowych, które nie zawsze byłyby zwracane przez przeglądarke albo LLM.

### Lista metod planowanych do zastosowania

1.  **Hybrid Search (Wyszukiwanie Hybrydowe):** Połączenie wyszukiwania semantycznego (Dense Retrieval) z wyszukiwaniem słów kluczowych (Sparse BM25) dla zwiększenia kompletności wyników.
2.  **Agentic Workflow (LangGraph):** Implementacja cyklicznego grafu decyzyjnego z mechanizmami pętli zwrotnej (Rewrite-Retrieve Loop).
3.  **Semantic Routing:** Klasyfikacja intencji użytkownika w celu wyboru odpowiedniego źródła wiedzy (Vector Store / Web Search).
4.  **Reranking:** Zastosowanie modelu Cross-Encoder do ponownej oceny i uszeregowania wyników zwróconych przez bazę wektorową.
5.  **Metadata Extraction:** Wykorzystanie LLM do strukturyzacji zapytań (wyciąganie filtrów: rok, gatunek, ocena) z tekstu rozmowy.

### Wskazanie zbioru danych do treningu i testowania rozwiązania

Projekt wykorzysta zbiór danych **TMDB 5000 Movie Dataset** (wersja rozszerzona 2023):

- Źródło: [TMDB Movies Dataset 2023 (930k movies)](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
