import os
import json
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

from retriever import retriever

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "mock_key"))

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class AgentResponse(BaseModel):
    reply: str
    is_clarifying: bool = Field(description="Set to true if your reply ends with a question to gather more constraints from the user.")
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

EXPANSION_PROMPT = """You are a search query expansion bot for SHL assessments.
Analyze the user's conversation history and output ONLY a single line of expanded keywords, synonyms, and semantic concepts related to the user's latest request. 
Do not include conversational text, just the raw query string to feed into an embedding engine.
Example input: "Java developer for stakeholders"
Example output: Java backend communication stakeholder management collaboration client-facing influence teamwork"""

def process_chat(messages: List[dict]) -> dict:
    try:
        # ----------------------------------------------------
        # STAGE 1: Query Expansion
        # ----------------------------------------------------
        expansion_contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=EXPANSION_PROMPT)])
        ]
        
        # Feed the conversation to generate the query
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            expansion_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
        expansion_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=expansion_contents,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        expanded_query = expansion_response.text.strip()
        print(f"[Query Expansion] {expanded_query}")
        
        # ----------------------------------------------------
        # STAGE 2: Semantic Pre-filter
        # ----------------------------------------------------
        top_candidates = retriever.search(expanded_query, top_k=50)
        candidates_text = "\n\n".join(top_candidates)
        
        # ----------------------------------------------------
        # STAGE 3: LLM Reranking & Final Generation
        # ----------------------------------------------------
        SYSTEM_PROMPT = f"""You are the SHL Assessment Recommender Assistant. Your task is to guide hiring managers to select appropriate assessments from the SHL catalog through a conversational interface.

Follow these strict rules:
1. CLARIFY: If a query is too vague (e.g., "I need a test"), ask clarifying questions. 
2. DO NOT OVER-CLARIFY: If the user has provided at least 2 or 3 solid constraints (e.g., a role, a seniority, and a specific skill), that is enough context. Do not get stuck in an endless loop of asking questions. Make a recommendation.
3. EMPTY RECOMMENDATIONS WHILE CLARIFYING: If you are asking a clarifying question or gathering context, you MUST return an empty array [] for `recommendations`. Do NOT provide "initial" or "preliminary" recommendations.
4. RECOMMEND: ONLY when you have enough constraints, evaluate the TOP 50 PRE-FILTERED CANDIDATES provided below and commit to a shortlist. 
5. REFINE: React to changed constraints fluidly.
6. COMPARE: Explain differences based purely on the provided descriptions.
7. STAY IN SCOPE: Only discuss SHL assessments. Refuse out-of-scope requests with an empty recommendations array.

Ranking Rubric:
When recommending (and NOT clarifying), evaluate the TOP 50 candidates using this strict rubric and return the absolute best 1 to 10 tests:
1. Skill relevance (Are the required skills explicitly measured?)
2. Job level alignment (Does it fit the seniority?)
3. Role fit
4. Assessment breadth
5. Industry applicability

Output Format:
Always reply matching the required JSON schema.
- `reply`: Your conversational message.
- `recommendations`: EXACTLY [] if clarifying, gathering context, or refusing. Otherwise, the Top 1 to 10 matched tests (extract `Name`, `URL`, and `TestType_Code` exactly as written).
- `end_of_conversation`: true ONLY if a solid final shortlist is provided and the user is satisfied.

TOP 50 PRE-FILTERED CANDIDATES:
{candidates_text}
"""
        
        final_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            final_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
        final_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=AgentResponse
        )

        final_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_contents,
            config=final_config,
        )
        
        response_dict = json.loads(final_response.text)
        
        # Programmatically enforce empty recommendations if the model is still clarifying
        if response_dict.get("is_clarifying", False):
            response_dict["recommendations"] = []
            
        # Remove the internal flag before returning to the user
        response_dict.pop("is_clarifying", None)
        
        return response_dict

    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return {
            "reply": "I'm sorry, I encountered an internal error processing your request.",
            "recommendations": [],
            "end_of_conversation": False
        }
