"""
answer_verifier.py — Local SymPy-powered answer verification engine for MathGenix.

Intercepts LLM-generated math questions and independently verifies the
numerical/symbolic correctness of each answer using SymPy. Runs entirely
offline with zero external API dependencies.

Verification statuses:
  - "verified"   → SymPy independently confirmed the answer is correct
  - "unverified" → SymPy computed a different answer (possible LLM error)
  - "skipped"    → Question is conceptual / non-computable (e.g. Remember, Understand)
  - "error"      → LaTeX parsing or SymPy solving failed
"""

import re
import concurrent.futures
import sympy
from sympy import (
    symbols, solve, simplify, N, Eq, oo,
    sqrt, Rational, pi, E, log, sin, cos, tan,
    factorial, binomial, Abs
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Bloom's levels where answers are typically computable
COMPUTABLE_LEVELS = {"Apply", "Analyze"}

# Timeout for a single verification (seconds)
VERIFY_TIMEOUT = 3

# ─── LaTeX cleaning & parsing ───────────────────────────────────────────────

def clean_latex(raw: str) -> str:
    """
    Strips LaTeX delimiters and normalizes common LaTeX commands into
    SymPy-parseable plain-text expressions.
    """
    s = raw.strip()

    # Remove $ delimiters
    s = re.sub(r'^\$\$?|\$\$?$', '', s)
    s = s.strip()

    # Common LaTeX → plain text replacements
    replacements = [
        (r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))'),  # \frac{a}{b} → ((a)/(b))
        (r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)'),                   # \sqrt{x} → sqrt(x)
        (r'\\sqrt\[([^]]+)\]\{([^{}]+)\}', r'(\2)**(1/(\1))'),  # \sqrt[n]{x} → x**(1/n)
        (r'\\left\(', '('),
        (r'\\right\)', ')'),
        (r'\\left\[', '('),
        (r'\\right\]', ')'),
        (r'\\cdot', '*'),
        (r'\\times', '*'),
        (r'\\div', '/'),
        (r'\\pi', 'pi'),
        (r'\\infty', 'oo'),
        (r'\\ln', 'log'),
        (r'\\log', 'log'),
        (r'\\sin', 'sin'),
        (r'\\cos', 'cos'),
        (r'\\tan', 'tan'),
        (r'\\pm', '+'),         # Take positive branch for verification
        (r'\\approx', '='),
        (r'\\leq', '<='),
        (r'\\geq', '>='),
        (r'\\neq', '!='),
        (r'\\le', '<='),
        (r'\\ge', '>='),
        (r'\\%', '/100'),
        (r'\^', '**'),
        (r'\\text\{[^{}]*\}', ''),  # Strip \text{...} labels
        (r'\\mathrm\{[^{}]*\}', ''),
        (r'\\,', ''),
        (r'\\;', ''),
        (r'\\quad', ''),
        (r'\\qquad', ''),
        (r'\\\\', ''),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    # Remove any remaining backslashes
    s = s.replace('\\', '')

    return s.strip()


def parse_answer_value(answer_str: str):
    """
    Attempts to extract a numeric or symbolic value from the LLM's answer string.
    Returns a SymPy expression or None.
    
    Handles formats like:
      "$x = 3$"  →  3
      "$\\frac{1}{2}$"  →  1/2
      "$x = -2, x = 3$"  →  [-2, 3]  (set of solutions)
      "5"  →  5
      "$42$"  →  42
    """
    cleaned = clean_latex(answer_str)

    # Try to extract value after "=" sign (e.g., "x = 3")
    eq_match = re.search(r'=\s*(.+)', cleaned)
    if eq_match:
        value_part = eq_match.group(1).strip()
    else:
        value_part = cleaned

    # Handle multiple solutions separated by commas or "and"/"or"
    # e.g., "x = 2, x = -3" or "x = 2 and x = -3"
    multi_match = re.split(r',\s*(?:x\s*=\s*)?|(?:\s+and\s+|\s+or\s+)(?:x\s*=\s*)?', value_part)

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

    results = []
    for val in multi_match:
        val = val.strip()
        if not val:
            continue
        try:
            expr = parse_expr(val, transformations=transformations)
            results.append(expr)
        except Exception:
            continue

    if len(results) == 0:
        return None
    elif len(results) == 1:
        return results[0]
    else:
        return sorted(results, key=lambda x: complex(N(x)).real if x.is_number else 0)


def extract_equation(question_text: str):
    """
    Attempts to extract a solvable equation from the question text.
    Looks for patterns like "solve ... = ...", "find x if ...", etc.
    Returns a SymPy Eq object or None.
    """
    cleaned = clean_latex(question_text)

    # Find equation patterns: look for "expression = expression"
    eq_patterns = [
        r'solve[:\s]+(.+?=.+?)(?:\.|$|,|\s+for)',
        r'find\s+\w+\s+(?:if|when|where|given)\s+(.+?=.+?)(?:\.|$)',
        r'equation\s+(.+?=.+?)(?:\.|$|,)',
        r'(.+?=\s*0)',  # Standard form: f(x) = 0
    ]

    for pattern in eq_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            eq_str = match.group(1).strip()
            parts = eq_str.split('=')
            if len(parts) == 2:
                transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
                try:
                    lhs = parse_expr(parts[0].strip(), transformations=transformations)
                    rhs = parse_expr(parts[1].strip(), transformations=transformations)
                    return Eq(lhs, rhs)
                except Exception:
                    continue

    return None


# ─── Single question verification ───────────────────────────────────────────

def _verify_single_question_core(question_obj: dict, taxonomy_level: str) -> str:
    """
    Core verification logic that executes inside a thread.
    """
    answer_str = question_obj.get("answer", "")
    question_str = question_obj.get("question", "")

    if not answer_str or not question_str:
        return "error"

    try:
        # 1. Parse the claimed answer
        claimed = parse_answer_value(answer_str)
        if claimed is None:
            return "error"

        # 2. Try to extract and solve the equation from the question
        equation = extract_equation(question_str)

        if equation is not None:
            # Identify the free variable (default to x)
            free_vars = equation.free_symbols
            if free_vars:
                var = list(free_vars)[0]  # Pick the first variable
                solutions = solve(equation, var)

                if solutions:
                    # Compare claimed answer against SymPy's solutions
                    if isinstance(claimed, list):
                        # Multiple claimed answers — check set equality
                        claimed_set = set(simplify(c) for c in claimed)
                        solution_set = set(simplify(s) for s in solutions)
                        if claimed_set == solution_set:
                            return "verified"
                    else:
                        # Single claimed answer — check if it's in the solution set
                        for sol in solutions:
                            # Try symbolic equality first
                            if simplify(claimed - sol) == 0:
                                return "verified"
                            # Fall back to numerical comparison
                            try:
                                if abs(complex(N(claimed)) - complex(N(sol))) < 0.01:
                                    return "verified"
                            except (TypeError, ValueError):
                                continue

                    return "unverified"

        # 3. If no equation found, try pure arithmetic verification
        #    (e.g., "What is 15% of 240?" → answer "36")
        if claimed is not None and hasattr(claimed, 'is_number') and claimed.is_number:
            # Try to find and evaluate arithmetic expressions in the question
            arith_patterns = [
                r'(?:calculate|compute|find|evaluate|what is)\s+(.+?)(?:\?|$)',
            ]
            for pattern in arith_patterns:
                match = re.search(pattern, clean_latex(question_str), re.IGNORECASE)
                if match:
                    expr_str = match.group(1).strip()
                    # Clean up percentage patterns
                    expr_str = re.sub(r'(\d+)%\s*(?:of)\s*(\d+)', r'(\1/100)*\2', expr_str)
                    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
                    try:
                        expr = parse_expr(expr_str, transformations=transformations)
                        result = N(expr)
                        if abs(complex(result) - complex(N(claimed))) < 0.01:
                            return "verified"
                        else:
                            return "unverified"
                    except Exception:
                        continue

        # 4. If we couldn't extract anything solvable, mark as error (not unverified)
        return "error"

    except Exception:
        return "error"


def verify_single_question(question_obj: dict, taxonomy_level: str) -> str:
    """
    Verifies a single question's answer using SymPy.
    Runs the verification core inside a threadpool with a timeout, ensuring
    it is completely thread-safe and cross-platform (works outside main thread).
    
    Returns one of: "verified", "unverified", "skipped", "error"
    """
    # Skip non-computable Bloom's levels
    if taxonomy_level not in COMPUTABLE_LEVELS:
        return "skipped"

    # Run core solver inside a threadpool executor with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_verify_single_question_core, question_obj, taxonomy_level)
        try:
            return future.result(timeout=VERIFY_TIMEOUT)
        except concurrent.futures.TimeoutError:
            print(f"⚠️ SymPy Verification timed out after {VERIFY_TIMEOUT}s")
            return "error"
        except Exception as e:
            print(f"⚠️ SymPy Verification raised unexpected exception: {e}")
            return "error"


# ─── Batch verification ─────────────────────────────────────────────────────

def verify_question_batch(questions: list, taxonomy_level: str) -> list:
    """
    Runs verification on a batch of questions. Attaches a 'verification_status'
    field to each question dict in-place and returns the annotated list.
    
    This function NEVER raises — verification failures are gracefully absorbed
    so question delivery is never blocked.
    """
    for q in questions:
        try:
            status = verify_single_question(q, taxonomy_level)
            q["verification_status"] = status
        except Exception:
            q["verification_status"] = "error"

    # Log summary
    statuses = [q.get("verification_status", "error") for q in questions]
    verified = statuses.count("verified")
    unverified = statuses.count("unverified")
    skipped = statuses.count("skipped")
    errors = statuses.count("error")
    print(f">>> Verification complete: {verified} verified, {unverified} unverified, {skipped} skipped, {errors} errors")

    return questions
