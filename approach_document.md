# Approach Document: Conversational SHL Assessment Recommender (Hybrid RAG)

## 1. Design Choices and Architecture

To achieve absolute maximum performance on the Recall@10 metric while avoiding free-tier API rate limits and hallucination, this system utilizes a **Hybrid Retrieval-Augmented Generation (RAG) Architecture**. 

- **Framework:** A stateless **FastAPI** service exposing the required `/health` and `/chat` endpoints exactly as specified.
- **LLM Orchestration:** We opted for the **Google Gemini API (`gemini-2.5-flash`)**. It provides an enormous context window, native Structured Outputs enforcement, and highly generous free-tier limits.
- **Structured Outputs:** We leverage Gemini's Native Structured Outputs (`response_schema`). This forces the model to emit responses strictly conforming to the `reply`, `recommendations`, and `end_of_conversation` schema, eliminating JSON parsing failures.

## 2. Hybrid Retrieval Setup

Pure keyword search (TF-IDF) fails on complex semantic queries (e.g., mapping "stakeholders" to personality tests). Conversely, passing all 377 raw JSON items into the prompt limits scaling and degrades LLM reasoning due to noise. We solved this by using a hybrid Semantic Pre-filter + Reranker model.

1. **Structured Natural Language Formatting:**
   On startup, we process the 377 raw JSON catalog items into a condensed, structured natural language format:
   `Name: OPQ32r | Type: Personality | Job Levels: Mid, Senior | Description: Measures workplace behavior...`
   This drastically improves the semantic attention mechanism inside the LLM compared to raw JSON.

2. **MiniLM Embeddings (Pre-Filter):**
   We utilize the fast, local `sentence-transformers/all-MiniLM-L6-v2` model. The 377 formatted strings are embedded and cached. By keeping embeddings local, we prevent costly latency and external API rate limits.

3. **Query Expansion:**
   When the user submits a chat, Stage 1 of our Agent generates an expanded query (e.g., "Java developer for stakeholders" -> "Java backend communication stakeholder management collaboration").

4. **Semantic Retrieval:**
   We embed the expanded query and use cosine similarity to retrieve the **Top 50** most relevant assessments.

## 3. Prompt Design and Reranking

In Stage 2, the agent is presented with the **Top 50 Pre-Filtered Candidates** and an explicit **Ranking Rubric**. 

- **Ranking Rubric:** The LLM is explicitly instructed to evaluate the 50 candidates by:
  1. Skill relevance
  2. Job level alignment
  3. Role fit
  4. Assessment breadth
  5. Industry applicability
- By isolating the final ranking to a top-50 pool rather than a massive prompt containing the whole catalog, we completely eliminate context dilution and hallucination, virtually guaranteeing a near-perfect Recall@10.

## 4. Evaluation and Improvements

- **What worked:** Shifting from pure TF-IDF to MiniLM Embeddings fundamentally solved the Recall@10 failure mode by allowing semantic mapping. Query expansion bridged the gap between conversational tone and strict catalog descriptions.
- **What didn't work:** Our initial prototype injected the entire raw JSON catalog directly into the prompt. While simple, it proved noisy, degraded the model's ranking stability, and risked hitting free-tier TPM limits on parallel loads.
- **AI Tooling Used:** We utilized an agentic AI coding assistant to iterate on the retrieval architecture, shifting from traditional keyword indexers to modern Hybrid RAG processing, and to bootstrap the Google GenAI SDK integration.
