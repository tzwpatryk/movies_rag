from typing import TypedDict, Optional, List, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str  # Aktualne pytanie (może być zmienione przez rewriter)
    synthesized_query: str  # Pelne informacje o preferencjach uzytkownika
    context: str  # Znalezione filmy (tekst sformatowany)
    is_relevant: str  # Decyzja sędziego: "yes" lub "no"
    retry_count: int  # Licznik prób, żeby uniknąć nieskończonej pętli
    generation: str
    chat_history: Annotated[List[BaseMessage], add_messages]  # Historia rozmowy


class RouteQuery(BaseModel):
    """Kieruje zapytanie w odpowiednie miejsce"""

    destination: Literal["vectorstore", "web_search", "general_chat"] = Field(
        ...,
        description="Gdzie skierować pytanie: 'vectorstore' dla rekomendacji filmowych, 'web_search' dla aktualnych wydarzeń/repertuaru, 'general_chat' dla zwykłej rozmowy.",
    )


class GradeDocuments(BaseModel):
    """Ocena trafności dokumentów."""

    binary_score: str = Field(
        description="Czy dokumenty są trafne względem pytania? 'yes' lub 'no'"
    )


class MovieSearchIntent(BaseModel):
    """
    Struktura interpretacji pytania o film.
    """

    synthesized_query: str = Field(
        ...,
        description="Pełne, opisowe zdanie w języku angielskim podsumowujące, czego szuka użytkownik, biorąc pod uwagę CAŁĄ historię rozmowy. Np. 'Horror movies from the 90s regarding space travel'.",
    )
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
