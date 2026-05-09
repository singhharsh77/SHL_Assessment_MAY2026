# Conversational SHL Assessment Recommender

This repository contains a stateless FastAPI microservice that powers an AI conversational agent designed to help hiring managers seamlessly select SHL assessments.

## The Problem: Passing the Evaluation Harness

Our primary goal was to build an agent that could successfully pass an automated evaluation harness spanning multiple turns of conversation while adhering strictly to a JSON output schema.

During development, we faced **two major challenges**:

1. **The Recall@10 Failure (Semantic Mismatch):** 
   Our initial approach relied on traditional TF-IDF keyword search. However, this failed hard on complex, semantic queries. For example, if a user requested an assessment for "stakeholder interaction," TF-IDF would strictly look for the word "stakeholders" and fail to recommend the OPQ32r personality test (which inherently measures stakeholder skills but lacks the exact keyword in its description).
   
2. **LLM Over-Clarification & Rate Limits:** 
   We initially tried dumping the entire 377-item JSON catalog into the LLM prompt. While it solved the semantic issue, it caused context bloat, increased latency, degraded the model's ranking stability, and risked hitting free-tier token limits (TPM) on the evaluation harness. Furthermore, the LLM would stubbornly generate "preliminary" recommendations while asking clarifying questions, violating the strict `recommendations: []` rule.

## The Solution: A Hybrid RAG Architecture

To solve these challenges and build an enterprise-grade AI system, we completely overhauled the architecture:

### 1. Miniature Semantic Pre-Filtering (`sentence-transformers`)
We removed TF-IDF and integrated `sentence-transformers/all-MiniLM-L6-v2`. On startup, the system translates the raw JSON catalog into condensed **Structured Natural Language** and caches local embeddings. This entirely solved the Recall@10 issue by allowing true semantic matching (e.g., mapping conversational requirements to formal test competencies) without external API overhead.

### 2. Two-Stage Pipeline (Gemini 2.5 Flash)
- **Stage 1 (Query Expansion):** The agent reads the conversation history and condenses it into a powerful, dense semantic query string.
- **Stage 2 (Reranking):** The pre-filter grabs the Top 50 matches. We inject *only* those 50 into the Gemini prompt along with an explicit **Ranking Rubric**. By shrinking the context window, we eliminated hallucination and stabilized the ranking.

### 3. Programmatic Output Enforcement
To prevent the LLM from leaking recommendations during the clarification phase, we implemented a strict programmatic intercept. The LLM natively generates an `is_clarifying` boolean. If true, the FastAPI backend forcefully wipes the recommendations array clean before returning the JSON, guaranteeing 100% compliance with the grading schema.

## Interactive Chat Demonstration

Below is a live demonstration of the Hybrid RAG agent parsing vague requirements, asking for clarification (while strictly outputting no recommendations), and finally committing to a highly accurate shortlist.


## Screenshots

## Local Env
  <img width="699" height="502" alt="Screenshot 2026-05-09 at 1 18 25 PM" src="https://github.com/user-attachments/assets/70b7d16b-bf1f-41e4-944c-bc0feffd29d5" />


## Live Env : [https://shlassessment-production-cc08.up.railway.app/](https://shlassessment-production-cc08.up.railway.app/) 
### use ```/chat , /health``` for testing

### /chats

- Request
  <img width="614" height="485" alt="Screenshot 2026-05-09 at 1 49 28 PM" src="https://github.com/user-attachments/assets/48315e5f-9bef-4fd0-8ad1-2d5fb40e734d" />

- Response
  <img width="616" height="567" alt="Screenshot 2026-05-09 at 1 49 46 PM" src="https://github.com/user-attachments/assets/146aa686-5803-4875-aa49-ad6bdfd6f93c" />


### /health

<img width="450" height="200" alt="Screenshot 2026-05-09 at 1 50 02 PM" src="https://github.com/user-attachments/assets/aa7ea25b-d4fd-4ad5-b636-d2772736864b" />

  

## Running the Project

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Export your Gemini API Key
export GEMINI_API_KEY="your_api_key_here"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```
