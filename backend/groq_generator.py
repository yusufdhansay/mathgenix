"""
groq_generator.py — Groq cloud LPU inference for MathGenix.

Uses Groq's OpenAI-compatible REST API to generate math questions
at ~300 tok/s using large 70B+ models. Dramatically faster than
local Ollama on consumer hardware.

Available models:
  - llama-3.1-70b-versatile   (best balance of speed + quality)
  - llama-3.3-70b-versatile   (newest, strong math reasoning)
  - mixtral-8x7b-32768        (fast, good for structured output)
"""

import os
import re
import json
import time
import requests
from .document_processor import get_text_chunks
from .question_generator import (
    BLOOMS_INSTRUCTIONS,
    retrieve_relevant_chunks,
    extract_json,
    sanitize_question_batch,
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Models available on Groq's free tier
GROQ_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "description": "Flagship model with strong math reasoning"},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "description": "Blazing-fast low-latency completion"},
]


def get_groq_api_key() -> str:
    """Returns the Groq API key from environment."""
    return os.environ.get("GROQ_API_KEY", "")


def check_groq_connection(api_key: str) -> dict:
    """
    Validates the Groq API key by making a lightweight models list request.
    Returns status dict with connection info.
    """
    if not api_key or api_key == "your_key_here":
        return {"status": "No API Key", "available": False}

    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        if response.status_code == 200:
            return {"status": "Connected", "available": True}
        elif response.status_code == 401:
            return {"status": "Invalid API Key", "available": False}
        else:
            return {"status": f"Error ({response.status_code})", "available": False}
    except requests.exceptions.ConnectionError:
        return {"status": "Network Error", "available": False}
    except Exception as e:
        return {"status": f"Error: {str(e)}", "available": False}


def generate_questions_groq(
    text: str,
    taxonomy_level: str,
    model_name: str = "llama-3.3-70b-versatile",
    api_key: str = "",
    previous_topics: list = None,
) -> dict:
    """
    Generates math questions using Groq's cloud LPU inference.
    Returns 5 questions (cloud can handle the larger payload easily).
    """
    if not api_key or api_key == "your_key_here":
        return {"error": "Groq API key not configured. Go to Settings and enter your key from console.groq.com"}

    level_instruction = BLOOMS_INSTRUCTIONS.get(taxonomy_level, "Generate questions based on the text.")

    # Prepare context from document
    try:
        chunks = get_text_chunks(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            return {"error": "No valid text found in the document to generate questions."}

        query = f"Mathematical concepts, formulas, and problems suitable for {taxonomy_level} questions."
        relevant_chunks = retrieve_relevant_chunks(chunks, query, k=4)
        context_text = "\n".join(relevant_chunks)[:2000]  # Cloud can handle more context
        print(f">>> [GROQ] Context text length: {len(context_text)} chars. First 200 chars: {repr(context_text[:200])}")
    except Exception as e:
        return {"error": f"Error processing document context: {str(e)}"}

    # Build exclusion instruction if previous topics exist
    exclusion_instruction = ""
    if previous_topics:
        topic_list = "; ".join(f'"{t}"' for t in previous_topics)
        exclusion_instruction = (
            f"\nCRITICAL UNIQUENESS RULE: The user has already generated questions on these topics: [{topic_list}]. "
            f"You MUST generate questions on COMPLETELY DIFFERENT topics, sub-topics, and formulas from the context. "
            f"Do NOT repeat, rephrase, or reuse any of the above topics. Pick fresh mathematical concepts.\n"
        )

    # Richer prompt for cloud — 70B models can follow complex instructions reliably
    prompt = f"""Generate exactly 5 math questions at Bloom's Taxonomy level "{taxonomy_level}".

Return ONLY a raw JSON object (no markdown, no code fences).

JSON format:
{{"questions":[{{"id":1,"topic":"Short Topic","question":"Question text with $inline\\ math$ and $$display\\ math$$","answer":"$x=3$","solution_steps":["Step 1 (1 sentence)","Step 2 (1 sentence)"]}}]}}

CRITICAL LaTeX formatting rules:
- Wrap ONLY pure math expressions in $ delimiters: $x^2 + 3x = 0$, $\\frac{{1}}{{2}}$
- NEVER use $ as a currency symbol. Write currency using its name, e.g., "100 dollars" or "100 USD"
- NEVER mix plain English text inside $ delimiters. WRONG: "$x = 3 is the solution$". RIGHT: "$x = 3$ is the solution"
- Each $ must have a matching closing $. Every $...$ block must contain ONLY valid LaTeX math
- Use \\frac{{a}}{{b}} for fractions, \\sqrt{{x}} for roots, $inline$ for inline, $$display$$ for block

Content rules:
- Generate exactly 5 questions, each on a DIFFERENT mathematical topic from the context
- Every question must be mathematically correct — verify your arithmetic before outputting
- Include all numerical values AND parameters needed to solve inside the question text (e.g. state n for nth-derivative problems)
- Never combine mismatched types in one expression (e.g. do not add a scalar to a vector, do not equate a single item to a fraction of a count)
- The question's actual solve step, not just its wording, must match the target Bloom's level below
- Provide exactly 2 concise solution steps per question (1 sentence each)
- {level_instruction}
- CRITICAL: Base questions strictly on the mathematical topics/formulas in the Context. If the Context lacks enough material, output fewer than 5 questions rather than inventing formulas.
{exclusion_instruction}
Context:
{context_text}"""

    try:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise mathematics question generator. Output ONLY valid JSON. CRITICAL: Base all questions strictly and exclusively on the provided context. In LaTeX, use $ only around pure math expressions (e.g. $x^2$). NEVER use $ as a currency symbol — write out the currency name (e.g., '100 dollars' instead of '$100'). Never put English words inside $ delimiters."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.5,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }

        print(f">>> [GROQ] Sending request ({model_name})...")
        start_time = time.time()

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,  # Groq should respond in <10s
        )

        elapsed = time.time() - start_time

        if response.status_code == 401:
            return {"error": "Invalid Groq API key. Please check your key in Settings."}
        elif response.status_code == 429:
            return {"error": "Groq rate limit exceeded. Please wait a moment and try again (free tier: 30 req/min)."}
        elif response.status_code != 200:
            return {"error": f"Groq API error ({response.status_code}): {response.text[:200]}"}

        result = response.json()

        # Extract the response content
        raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Log performance
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_time = usage.get("total_time", elapsed)
        if completion_tokens > 0:
            tps = completion_tokens / elapsed if elapsed > 0 else 0
            print(f">>> [GROQ] Complete: {elapsed:.1f}s, {completion_tokens} tokens, {tps:.0f} tok/s")

        print("====== GROQ RAW RESPONSE ======")
        print(raw_text[:500])
        print("===============================")

        # Parse JSON
        parsed_data = extract_json(raw_text)

        if parsed_data and "questions" in parsed_data and len(parsed_data["questions"]) > 0:
            # Sanitize LaTeX (fix broken matrices, missing backslashes, etc.)
            parsed_data["questions"] = sanitize_question_batch(parsed_data["questions"])
            return parsed_data

        # Try direct JSON parse as fallback (Groq's json_object mode is very reliable)
        try:
            parsed_data = extract_json(raw_text)
            if parsed_data and "questions" in parsed_data and len(parsed_data["questions"]) > 0:
                parsed_data["questions"] = sanitize_question_batch(parsed_data["questions"])
                return parsed_data
        except Exception:
            pass

        print("!!! [GROQ] JSON PARSING FAILED !!!")
        return {
            "error": "Failed to parse structured JSON from Groq response.",
            "raw_response": raw_text[:300],
        }

    except requests.exceptions.Timeout:
        print("!!! [GROQ] REQUEST TIMEOUT !!!")
        return {"error": "Groq request timed out after 30 seconds. Please try again."}
    except requests.exceptions.ConnectionError:
        print("!!! [GROQ] CONNECTION FAILED !!!")
        return {"error": "Cannot connect to Groq API. Please check your internet connection."}
    except Exception as e:
        print(f"!!! [GROQ] EXCEPTION: {str(e)} !!!")
        return {"error": f"Groq generation error: {str(e)}"}