from .document_processor import get_text_chunks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json
import os
import requests

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434")

# Dictionary mapping Bloom's taxonomy levels to compact cognitive tier blueprints
BLOOMS_INSTRUCTIONS = {
    "Remember": "REMEMBER: Ask recall-only questions about definitions, formulas, or theorems explicitly stated in the context. E.g., state a formula or identify a term from the text. Do not ask for multi-step calculations.",
    "Understand": "UNDERSTAND: Ask conceptual questions requiring explanation or interpretation of ideas from the context. E.g., explain what a term means in the context of the text, or translate a relation described in the text into an equation.",
    "Apply": "APPLY: Ask procedural questions where the user must apply a formula or algorithm present in the context to solve a problem with given values. E.g., compute a value using a formula from the text. Do not generate unrelated topics like compound interest or quadratic equations unless they are in the context.",
    "Analyze": "ANALYZE: Ask breakdown questions based on the context. E.g., identify an error in a worked solution of a concept from the text, or compare two methods described in the text.",
    "Evaluate": "EVALUATE: Ask justification or assessment questions based on the context. E.g., justify why a theorem from the text holds under certain conditions, or evaluate which method from the text is more efficient.",
    "Create": "CREATE: Ask synthesis questions where the user must construct a new model, word problem, or system based on the rules and concepts in the context."
}


def retrieve_relevant_chunks(chunks, query, k=3):
    """
    Uses TF-IDF + cosine similarity to find the most relevant text chunks for a query.
    """
    if not chunks:
        return []
    
    all_texts = chunks + [query]
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    top_k = min(k, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

def extract_json(text: str):
    """
    Attempts to extract and parse a JSON block from the LLM output.
    Preprocesses LaTeX backslashes to prevent JSON decode errors and control character stripping.
    """
    text = text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Preprocess text to escape LaTeX backslashes.
    # We want to double any backslash that is NOT followed by '"' or '\'.
    # This prevents json.loads from parsing LaTeX commands like \begin, \frac, \theta, \nabla as JSON escapes.
    fixed = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text):
            next_char = text[i + 1]
            if next_char in ('"', '\\'):
                # Keep escaped quotes and escaped backslashes as-is
                fixed.append(text[i])
                fixed.append(text[i + 1])
                i += 2
            else:
                # Double the backslash for everything else (LaTeX commands, etc.)
                fixed.append('\\')
                fixed.append('\\')
                fixed.append(text[i + 1])
                i += 2
        else:
            fixed.append(text[i])
            i += 1
    processed_text = ''.join(fixed)

    # Strategy 1: Direct parse
    try:
        return json.loads(processed_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object boundaries with brace matching
    start_idx = processed_text.find('{"questions"')
    if start_idx == -1:
        start_idx = processed_text.find('{')
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False
    json_str = ""

    for i in range(start_idx, len(processed_text)):
        char = processed_text[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = processed_text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break

    return None



def sanitize_latex(text: str) -> str:
    """
    Backend post-processor for LaTeX text coming from LLM JSON output.
    
    Focuses on reliable, deterministic fixes:
      1. Double-escaped backslashes from JSON (\\\\frac → \\frac)
      2. Missing backslashes on known LaTeX commands (frac{ → \\frac{)
      3. Matrix environment row separators
      4. Matrix environments missing $$ wrappers
    
    NOTE: Smart $ delimiter fixing (currency detection, mixed content splitting,
    bare command wrapping) is handled by the frontend MathRenderer, which has
    access to KaTeX and can make render-time decisions.
    """
    if not text:
        return text

    # ── Phase 0: Fix double-escaped backslashes from JSON parsing ────
    text = re.sub(r'\\\\(frac|sqrt|sum|prod|int|lim|sin|cos|tan|cot|sec|csc|'
                  r'alpha|beta|gamma|delta|theta|lambda|pi|infty|partial|nabla|'
                  r'cdot|times|div|text|mathrm|mathbf|left|right|log|ln|det|max|min|'
                  r'begin|end|over|quad|qquad|hat|bar|vec|dot|ddot|tilde|pm|mp|leq|geq|'
                  r'neq|approx|equiv|pmod|bmod|binom|displaystyle)', 
                  r'\\\1', text)

    # ── Phase 1: Fix begin/end environment commands ──────────────────
    text = re.sub(r'(?<!\\)begin\{', r'\\begin{', text)
    text = re.sub(r'(?<!\\)end\{', r'\\end{', text)

    # ── Phase 2: Fix matrix row separators ───────────────────────────
    matrix_envs = ['bmatrix', 'pmatrix', 'vmatrix', 'matrix', 'Bmatrix', 'Vmatrix', 'cases']
    for env in matrix_envs:
        pattern = r'(\\begin\{' + env + r'\})([\s\S]*?)(\\end\{' + env + r'\})'
        
        def replace_matrix(match):
            begin_tag = match.group(1)
            content = match.group(2)
            end_tag = match.group(3)
            
            def replace_slashes(m):
                val = m.group(0)
                end_pos = m.end()
                full_string = m.string
                next_char = full_string[end_pos:end_pos+1] if end_pos < len(full_string) else ''
                n = len(val)
                if n % 2 == 1:
                    return '\\' if (n == 1 and next_char.isalpha()) else '\\\\'
                return '\\\\'
            sanitized_content = re.sub(r'\\+', replace_slashes, content)
            return begin_tag + sanitized_content + end_tag
        text = re.sub(pattern, replace_matrix, text)

    # ── Phase 3: Fix missing backslash on bare LaTeX commands ────────
    latex_commands = [
        'frac', 'sqrt', 'sum', 'prod', 'int', 'lim',
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'pi',
        'infty', 'partial', 'nabla', 'cdot', 'times', 'div',
        'text', 'mathrm', 'mathbf', 'left', 'right',
        'log', 'ln', 'det', 'max', 'min',
        'over', 'pm', 'mp', 'leq', 'geq', 'neq', 'approx', 'equiv',
    ]
    for cmd in latex_commands:
        text = re.sub(r'(?<!\\)\b' + cmd + r'(?=[\s{(])', lambda m, c=cmd: '\\' + c, text)

    # ── Phase 4: Ensure matrix environments have $$ delimiters ───────
    for env in matrix_envs:
        text = re.sub(
            r'(?<!\$)(\\begin\{' + env + r'\}[\s\S]*?\\end\{' + env + r'\})(?!\$)',
            lambda m: '$$' + m.group(1) + '$$',
            text
        )

    return text


def sanitize_question_batch(questions: list) -> list:
    """
    Applies LaTeX sanitization to all text fields in a batch of questions.
    """
    for q in questions:
        for field in ['question', 'answer']:
            if field in q and isinstance(q[field], str):
                q[field] = sanitize_latex(q[field])
        if 'solution_steps' in q and isinstance(q['solution_steps'], list):
            q['solution_steps'] = [
                sanitize_latex(step) if isinstance(step, str) else step
                for step in q['solution_steps']
            ]
    return questions

def generate_questions(text: str, taxonomy_level: str, model_name: str = "llama3", previous_topics: list = None):
    """
    Generates questions based on the provided text and Bloom's Taxonomy level.
    Uses direct Ollama REST API calls — zero LangChain overhead.
    Optimized for fast inference on Apple Silicon (M1/M2) with constrained memory.
    """
    level_instruction = BLOOMS_INSTRUCTIONS.get(taxonomy_level, "Generate questions based on the text.")

    try:
        chunks = get_text_chunks(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            return {"error": "No valid text found in the document to generate questions."}
        
        query = f"Mathematical concepts, formulas, and problems suitable for {taxonomy_level} questions."
        relevant_chunks = retrieve_relevant_chunks(chunks, query, k=3)
        # Trim total context to ~1200 chars max to keep prompt under 2048 tokens
        context_text = "\n".join(relevant_chunks)[:1200]
        print(f">>> [OLLAMA] Context text length: {len(context_text)} chars. First 200 chars: {repr(context_text[:200])}")
    except Exception as e:
        return {"error": f"Error processing document context: {str(e)}"}

    # Build exclusion instruction if previous topics exist
    exclusion_instruction = ""
    if previous_topics:
        topic_list = "; ".join(f'"{t}"' for t in previous_topics)
        exclusion_instruction = (
            f"\nDo NOT generate questions on these already-used topics: [{topic_list}]. "
            f"Pick DIFFERENT topics from the context.\n"
        )

    # Ultra-compact prompt — every token saved = faster generation on M1
    prompt = f"""Generate exactly 3 math questions at Bloom's level "{taxonomy_level}". Return ONLY raw JSON, no markdown.

JSON format:
{{"questions":[{{"id":1,"topic":"Short Topic","question":"Question with $inline\\ math$","answer":"$x=3$","solution_steps":["Step 1","Step 2"]}}]}}

LaTeX rules:
- Wrap ONLY pure math in $ delimiters: $x^2 + 3x = 0$, $\\frac{{a}}{{b}}$
- NEVER use $ for currency. Write out the currency name, e.g., "100 dollars"
- NEVER put English text inside $...$. WRONG: "$x = 3 is the solution$". RIGHT: "$x = 3$ is the solution"
- Use \\frac{{a}}{{b}}, \\sqrt{{x}}, $inline$, $$block$$

Content rules:
- 3 questions, each on a different topic from the context
- Verify arithmetic correctness before outputting
- Include all values needed to solve inside the question
- Exactly 2 short solution steps per question (1 sentence each)
- {level_instruction}
{exclusion_instruction}
Context:
{context_text}"""

    try:
        # Direct Ollama REST API call — no LangChain middleware
        payload = {
            "model": model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_ctx": 2048,        # Small context window → stays in Metal VRAM
                "num_predict": 1024,    # Cap output tokens (3 questions ≈ 600 tokens)
            }
        }
        
        print(f">>> Sending request to Ollama ({model_name})...")
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload,
            timeout=120  # 2 min hard timeout
        )
        response.raise_for_status()
        
        result = response.json()
        raw_text = result.get("response", "")
        
        # Log timing info from Ollama's response metadata
        total_duration = result.get("total_duration", 0)
        eval_count = result.get("eval_count", 0)
        if total_duration > 0:
            secs = total_duration / 1e9
            tps = eval_count / secs if secs > 0 else 0
            print(f">>> Generation complete: {secs:.1f}s, {eval_count} tokens, {tps:.1f} tok/s")
        
        # Debug printing to uvicorn task logs
        print("====== OLLAMA RAW RESPONSE ======")
        print(raw_text[:500])  # Only print first 500 chars to avoid log bloat
        print("=================================")
        
        # Parse the JSON response
        parsed_data = extract_json(raw_text)
        
        if parsed_data and "questions" in parsed_data and len(parsed_data["questions"]) > 0:
            # Sanitize LaTeX (fix broken matrices, missing backslashes, etc.)
            parsed_data["questions"] = sanitize_question_batch(parsed_data["questions"])
            return parsed_data
        
        print("!!! JSON PARSING FAILED !!!")
        return {
            "error": "Failed to parse structured JSON from model response.",
            "raw_response": raw_text[:300]
        }
        
    except requests.exceptions.Timeout:
        print("!!! REQUEST TIMEOUT (120s) !!!")
        return {"error": "Generation timed out after 120 seconds. Try a smaller document or a lighter model like llama3."}
    except requests.exceptions.ConnectionError:
        print("!!! OLLAMA CONNECTION FAILED !!!")
        return {"error": "Cannot connect to Ollama. Please ensure it is running (open Ollama app or run 'ollama serve')."}
    except Exception as e:
        print(f"!!! GENERATION EXCEPTION: {str(e)} !!!")
        return {"error": f"Error generating questions: {str(e)}"}
