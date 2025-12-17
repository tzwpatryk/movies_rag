# Film Agent (LangGraph RAG)

This project is an experimental implementation of an Agentic RAG (Retrieval-Augmented Generation) system for movie recommendations. It uses **LangGraph** to create a control flow where the AI evaluates the quality of search results and self-corrects by rewriting queries if the initial retrieval is insufficient.

**Status:** Work in Progress / Educational Prototype

## Architecture

The system does not follow a linear path. It operates as a state graph with the following logic:

1. **Retrieve:** Analyzes the user query to extract metadata filters (year, genre) and performs a Hybrid Search (Dense + Sparse) in Qdrant.
2. **Grade:** An LLM acts as a judge to evaluate if the retrieved documents are relevant to the user's question.
3. **Rewrite (Loop):** If the documents are irrelevant, the agent reformulates the query and retries the search (up to a fixed limit).
4. **Generate:** Once relevant context is found, the LLM generates the final answer.

## Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM Inference:** Groq API (Llama 3.3 70B)
- **Vector Database:** Qdrant (Local Docker instance)
- **Embeddings:** Qwen (Dense) + BM25 (Sparse)
- **Reranking:** Cross-Encoder (ms-marco-MiniLM)

## Project Structure

The code is modularized into the following components:

- `film_agent.py` - **Entry point.** Defines the StateGraph workflow, conditional edges, and runs the application.
- `nodes.py` - **Graph Nodes.** Contains the core functions for each step: `retrieve`, `grade`, `rewrite`, and `generate`.
- `utils.py` - **Utilities.** Handles Qdrant search logic, Pydantic models for intent extraction, and result reranking.
- `config.py` - **Configuration.** Initializes models, clients, and environment variables.

## Setup and Usage

**1. Prerequisites**

- Python 3.10+
- Docker (for Qdrant)
- Groq API Key

**2. Installation**
Install the required dependencies:

```bash
pip install -r requirements.txt

```

**3. Environment**
Create a `.env` file in the root directory:

```text
GROQ_API_KEY=your_api_key_here

```

**4. Run Qdrant**
Start a local Qdrant instance:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

```

_Note: You must have a collection named `movies_db` indexed with movie data before running the agent._

**5. Execution**
Run the agent:

```bash
python film_agent.py

```
