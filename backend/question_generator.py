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
    "Understand": (
        "UNDERSTAND: Questions must test CONCEPTUAL COMPREHENSION — the student should explain, interpret, paraphrase, or compare ideas from the context. "
        "STRICTLY FORBIDDEN patterns: "
        "(a) 'Find the probability of...' or 'Calculate the value of...' — that is Apply level, NOT Understand. "
        "(b) 'Who is credited with...' or 'In what year...' — that is Remember level, NOT Understand. "
        "CORRECT Understand-level patterns: "
        "(1) 'Explain in your own words why [theorem/formula] holds under [condition]' "
        "(2) 'Describe the geometric interpretation of [concept] in the context of [topic]' "
        "(3) 'Compare and contrast [method A] and [method B] as described in the text' "
        "(4) 'What happens to [result] if [parameter/condition] is changed? Explain conceptually' "
        "The question must require the student to EXPLAIN or INTERPRET, never to compute a numerical answer."
    ),
    "Apply": "APPLY: Ask procedural questions where the user must apply a formula or algorithm present in the context to solve a problem with given values. E.g., compute a value using a formula from the text. Do not generate unrelated topics like compound interest or quadratic equations unless they are in the context.",
    "Analyze": (
        "ANALYZE: Questions must require DECOMPOSITION, ERROR DETECTION, or STRUCTURAL COMPARISON — the student should break a problem into components, identify flaws, or distinguish between approaches. "
        "STRICTLY FORBIDDEN patterns: "
        "(a) 'Find the derivative/transform/integral of...' — that is Apply level, NOT Analyze. "
        "(b) 'Solve the differential equation...' — that is Apply level, NOT Analyze. "
        "(c) Any question that can be answered by plugging values into a single formula is Apply, NOT Analyze. "
        "CORRECT Analyze-level patterns: "
        "(1) 'The following worked solution contains an error: [show steps]. Identify the mistake and explain why it is incorrect' "
        "(2) 'Break down the solution of [complex problem] into its component sub-problems and explain the role of each step' "
        "(3) 'Compare method A vs method B for solving [problem type]. Under what conditions does each method fail or succeed?' "
        "(4) 'Given that [result], determine which theorem or property from the context was applied and justify your reasoning' "
        "The question must force the student to DECOMPOSE, DISTINGUISH, or DETECT errors — not just compute."
    ),
    "Evaluate": "EVALUATE: Ask justification or assessment questions based on the context. E.g., justify why a theorem from the text holds under certain conditions, or evaluate which method from the text is more efficient.",
    "Create": (
        "CREATE: Questions MUST require the student to CONSTRUCT, DERIVE, DESIGN, or SYNTHESIZE something new. "
        "STRICTLY FORBIDDEN pattern: 'Find the [transform/derivative/integral] of [function]' — that is Apply level, NOT Create. "
        "CORRECT Create-level patterns: "
        "(1) 'Derive the relationship between [concept A] and [concept B] starting from their definitions' "
        "(2) 'Construct a function f(x) that satisfies BOTH [constraint 1] AND [constraint 2] simultaneously' "
        "(3) 'Combine [theorem 1] with [theorem 2] to develop a general formula for [new result]' "
        "(4) 'Design a [system/signal/function] that achieves [goal] using concepts from the context' "
        "The question must force the student to BUILD something original, not just compute a known formula."
    ),
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
    Preprocesses LaTeX backslashes, trailing commas, smart quotes, embedded
    newlines, unescaped inner quotes, and truncation to prevent JSONDecodeErrors.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Fix 1: Normalize smart/curly quotes to straight quotes.
    _SMART_QUOTES = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'"}
    for smart, straight in _SMART_QUOTES.items():
        text = text.replace(smart, straight)

    # Fix 2+3 (combined state-machine pass): escape bare backslashes that
    # aren't already valid JSON escapes (LaTeX commands like \frac, \theta,
    # \nabla), escape literal newlines/tabs/CRs found INSIDE a string value,
    # and escape a `"` that appears inside a string but isn't really closing
    # it (heuristic: if what follows past whitespace isn't , : } ] then it's
    # literal content, not a real closing quote).
    #
    # IMPORTANT: only '"' and '\\' are treated as "already a valid escape,
    # leave alone" — NOT n/t/r/b/f/u. Those letters are exactly the first
    # letter of extremely common LaTeX commands (\nabla, \tan, \frac,
    # \right, \beta, \upsilon), so treating them as pass-through JSON
    # escapes would silently corrupt LaTeX into literal control characters.
    fixed = []
    i = 0
    in_string = False
    n = len(text)
    while i < n:
        ch = text[i]

        if ch == '\\' and i + 1 < n:
            nxt = text[i + 1]
            if nxt in ('"', '\\'):
                fixed.append(ch); fixed.append(nxt); i += 2
            else:
                fixed.append('\\'); fixed.append('\\'); fixed.append(nxt); i += 2
            continue

        if ch == '"':
            if in_string:
                j = i + 1
                while j < n and text[j] in ' \t\r\n':
                    j += 1
                if j < n and text[j] in ',:}]':
                    in_string = False
                    fixed.append(ch)
                else:
                    fixed.append('\\"')
            else:
                in_string = True
                fixed.append(ch)
            i += 1
            continue

        if in_string and ch == '\n':
            fixed.append('\\n'); i += 1; continue
        if in_string and ch == '\r':
            fixed.append('\\r'); i += 1; continue
        if in_string and ch == '\t':
            fixed.append('\\t'); i += 1; continue

        fixed.append(ch)
        i += 1

    processed_text = ''.join(fixed)

    # Fix 4: Strip trailing commas before a closing brace/bracket.
    processed_text = re.sub(r',(\s*[}\]])', r'\1', processed_text)

    # Strategy 1: Direct parse
    try:
        return json.loads(processed_text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Brace-matching boundary extraction
    start_idx = processed_text.find('{"questions"')
    if start_idx == -1:
        start_idx = processed_text.find('{')
    if start_idx == -1:
        return None

    brace_count = 0
    in_str = False
    json_str = ""
    i = start_idx
    while i < len(processed_text):
        char = processed_text[i]
        if char == '\\' and i + 1 < len(processed_text):
            i += 2
            continue
        if char == '"':
            in_str = not in_str
        if not in_str:
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
        i += 1

    # Fix 5: Truncation recovery. If the model hit max_tokens mid-question,
    # salvage every COMPLETE question object instead of discarding the whole
    # batch. Find the last fully-closed "}" belonging to a question entry,
    # cut there, and re-close the array/object.
    q_start = processed_text.find('"questions"')
    if q_start != -1:
        arr_start = processed_text.find('[', q_start)
        if arr_start != -1:
            depth = 0
            last_complete_end = -1
            in_str2 = False
            k = arr_start
            while k < len(processed_text):
                c = processed_text[k]
                if c == '\\' and k + 1 < len(processed_text):
                    k += 2
                    continue
                if c == '"':
                    in_str2 = not in_str2
                if not in_str2:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            last_complete_end = k
                k += 1
            if last_complete_end != -1:
                salvaged = processed_text[:last_complete_end + 1] + "]}"
                try:
                    return json.loads(salvaged)
                except json.JSONDecodeError:
                    pass

    return None



# Single source of truth for every LaTeX command sanitize_latex knows about.
# Previously this list was duplicated (and inconsistently maintained) across
# Phase 0 and Phase 3 separately, which is why fixes kept failing to stick —
# a command added in one phase was often missing from the other. Defining it
# once here means every phase automatically stays in sync.
#
# Deliberately EXCLUDES tokens that collide with common English words even
# though they are valid LaTeX commands (e.g. 'to', 'in', 'big', 'over', 'top',
# 'star', 'square', 'triangle', 'angle', 'dim') — auto-inserting a backslash
# into ordinary prose containing those words would corrupt the question text.
LATEX_COMMANDS_MASTER = [
    # Greek (lower) — none of these collide with real English words
    'varepsilon', 'vartheta', 'varsigma', 'varrho', 'varphi', 'varpi',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota',
    'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau',
    'upsilon', 'phi', 'chi', 'psi', 'omega',
    # Greek (upper)
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
    # calculus / operators
    'frac', 'sqrt', 'sum', 'prod', 'oint', 'iiint', 'iint', 'int', 'lim',
    'max', 'min', 'det', 'gcd', 'lcm', 'deg', 'log', 'ln', 'exp',
    'sinh', 'cosh', 'tanh', 'coth', 'arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
    'cot', 'sec', 'csc', 'partial', 'nabla', 'curl',
    'cdot', 'times', 'circ', 'oplus', 'ominus', 'otimes', 'odot',
    # relations
    'leq', 'geq', 'neq', 'approx', 'equiv', 'propto', 'simeq', 'sim', 'cong', 'perp',
    'parallel', 'subseteq', 'subset', 'supseteq', 'supset', 'notin', 'ni',
    'forall', 'exists', 'emptyset', 'varnothing',
    # arrows (only the distinctive multi-letter forms; bare "to" is excluded)
    'rightarrow', 'leftrightarrow', 'leftarrow', 'Rightarrow', 'Leftrightarrow',
    'Leftarrow', 'mapsto',
    # delimiters / sizing
    'left', 'right',
    # formatting
    'mathrm', 'mathbf', 'mathit', 'mathcal', 'mathbb', 'boldsymbol', 'text',
    'overline', 'underline', 'widehat', 'widetilde', 'hat', 'bar', 'vec', 'ddot', 'dot', 'tilde',
    # misc — distinctive tokens, low collision risk
    'binom', 'infty', 'aleph', 'hbar', 'ell', 'pm', 'mp',
    'begin', 'end', 'displaystyle', 'quad', 'qquad',
]
# Longest-first so alternation/lookahead prefers e.g. "sinh" over "sin"
LATEX_COMMANDS_SORTED = sorted(set(LATEX_COMMANDS_MASTER), key=len, reverse=True)
LATEX_CMD_ALTERNATION = '|'.join(re.escape(c) for c in LATEX_COMMANDS_SORTED)


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
    text = re.sub(r'\\\\(' + LATEX_CMD_ALTERNATION + r')', r'\\\1', text)

    # ── Phase 1: Fix begin/end environment commands ──────────────────
    text = re.sub(r'(?<!\\)begin\{', r'\\begin{', text)
    text = re.sub(r'(?<!\\)end\{', r'\\end{', text)

    # ── Phase 2: Fix matrix row separators ───────────────────────────
    # 'array' was missing here — \begin{array}{ll}...\end{array} (a very
    # common construct for piecewise/Laplace functions in this project's
    # content) was silently never getting its row separators repaired.
    matrix_envs = ['bmatrix', 'pmatrix', 'vmatrix', 'matrix', 'Bmatrix', 'Vmatrix',
                   'cases', 'array', 'align', 'align*', 'aligned', 'gathered', 'split']
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

    # ── Phase 2b: Escape bare curly-brace delimiters after \left/\right ──
    # \left{ and \right} are a DIFFERENT bug class from a missing command
    # backslash: \left itself is present and correct, but KaTeX requires
    # the delimiter argument to carry its own backslash (\left\{) — a bare
    # { is parsed as a TeX group-opener, not a delimiter, and errors out.
    # This is easy to miss because \left/\right look completely correct
    # at a glance; only the brace right after them is wrong.
    text = re.sub(r'\\left\{', r'\\left\\{', text)
    text = re.sub(r'\\right\}', r'\\right\\}', text)

    # ── Phase 3: Fix missing backslash on bare LaTeX commands ────────
    # Lookahead accepts normal delimiters/punctuation/end-of-string OR the
    # start of ANOTHER known command. That second option is what makes this
    # catch "sinleft(" — without it, "sin" is never matched because nothing
    # follows it except the letter "l", which used to look like plain text.
    _lookahead = (r'(?=[\s{}()\[\]$^_.,;:!?\\]|(?:' + LATEX_CMD_ALTERNATION + r')|$)')
    for cmd in LATEX_COMMANDS_SORTED:
        pattern = r'(?<!\\)\b' + re.escape(cmd) + _lookahead
        text = re.sub(pattern, lambda m, c=cmd: '\\' + c, text)

    # ── Phase 3b: Split glued commands, e.g. "\sinleft" → "\sin\left",  ──
    # "\nablaphi" → "\nabla\phi". Phase 3 above can only match the FIRST
    # command in a glued run (it has a real word boundary before it); the
    # second command has no boundary of its own since it's glued directly
    # onto the first. This anchors on "immediately after an already-
    # backslashed command" instead, which never fires on ordinary prose.
    _chain_pattern = re.compile(
        r'\\(' + LATEX_CMD_ALTERNATION + r')(' + LATEX_CMD_ALTERNATION + r')(?![a-zA-Z])'
    )
    for _ in range(4):  # fixpoint loop for chains of 3+ glued commands
        new_text = _chain_pattern.sub(r'\\\1\\\2', text)
        if new_text == text:
            break
        text = new_text

    # ── Phase 3c: Single-letter function name glued to left(/right) ──────
    # Handles "uleft(t)" → "u\left(t)" and "uright)" → "u\right)" — the
    # letter is a variable name (e.g. the unit step function u(t)), not a
    # command, so it must NOT get a backslash itself. The lookahead requires
    # an immediately-adjacent paren (no space), which real English words
    # essentially never do, so this can't misfire on prose like "Wright".
    text = re.sub(r'(?<!\\)\b([a-zA-Z])left(?=\()', r'\1\\left', text)
    text = re.sub(r'(?<!\\)\b([a-zA-Z])right(?=\))', r'\1\\right', text)

    # ── Phase 3d: Single-letter coefficient glued to an accent command ───
    # Handles vector-calculus notation like "xhat{i}" -> "x\hat{i}" and
    # "yvec{r}" -> "y\vec{r}" (a coefficient directly followed by \hat{},
    # \vec{}, \bar{}, \tilde{} etc). Requiring an immediate "{" right after
    # the command name is the safety net — no English word is ever directly
    # followed by "{", so this cannot misfire on prose.
    for _accent in ('widehat', 'widetilde', 'hat', 'bar', 'vec', 'ddot', 'dot', 'tilde'):
        text = re.sub(
            r'(?<!\\)\b([a-zA-Z])' + _accent + r'(?=\{)',
            lambda m, a=_accent: m.group(1) + '\\' + a,
            text,
        )

    # ── Phase 3e: Digit glued directly to a command name, e.g. "2pi" ─────
    # "\b" never fires between a digit and a letter, so standalone "pi" gets
    # fixed by Phase 3 but "2pi" (extremely common: 2π, 3σ, 10θ...) does not.
    # Safe to de-boundary here because no real English word is ever a digit
    # immediately followed by a math command name (no false-positive risk).
    text = re.sub(
        r'(?<=\d)(' + LATEX_CMD_ALTERNATION + r')' + _lookahead,
        lambda m: '\\' + m.group(1),
        text,
    )

    # ── Phase 3f: Non-standard commands KaTeX has no symbol for ──────────
    # \curl is not a real (KaTeX or standard LaTeX) macro — no amount of
    # backslash-repair fixes it, since the problem isn't a missing backslash,
    # it's that the command itself doesn't exist. Substitute it for something
    # KaTeX can actually render instead of letting it fail and fall back to
    # raw text on-screen.
    text = re.sub(r'\\curl\b', r'\\operatorname{curl}', text)

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

Question-writing rules:
- The question must ONLY state the problem. Do NOT include the solving formula or method in the question.
  WRONG: "Find the Laplace transform of $f(t)$ using the formula $L{{f(t)}} = ...$"
  RIGHT: "Find the Laplace transform of $f(t) = t^2 \\sin 3t$"
- Give ONLY the function/expression/values to work with. Formulas go in solution_steps, NOT the question.

Content rules:
- 3 questions, each on a DIFFERENT sub-topic from the context (never repeat the same theorem/transform/method)
- Verify arithmetic correctness before outputting
- Include all values and parameters needed to define the problem (e.g. state n for nth-derivative)
- Never combine mismatched types (no adding a scalar to a vector)
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