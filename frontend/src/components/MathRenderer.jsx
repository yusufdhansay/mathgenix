import React, { useMemo } from 'react';
import katex from 'katex';

/**
 * MathRenderer — Robust LaTeX + text renderer for LLM-generated math content.
 *
 * This is the permanent, comprehensive solution. Instead of relying on the LLM
 * to perfectly format LaTeX (it won't), this renderer:
 *
 *   1. PRE-PROCESSES the text to fix common LLM formatting issues
 *   2. TOKENIZES into display math ($$...$$), inline math ($...$),
 *      bare LaTeX commands, and plain text
 *   3. RENDERS each token with KaTeX, falling back gracefully on errors
 *
 * All edge cases from known LLM outputs are handled:
 *   - Bare \frac{}{} outside any $...$ delimiters → auto-wrapped
 *   - $100 (currency) → rendered as plain "100"
 *   - Mixed English + math inside $...$ → split properly
 *   - Unmatched $ signs → stripped
 *   - Double-escaped \\frac → fixed to \frac
 */
export default function MathRenderer({ text = '', className = '' }) {
  const rendered = useMemo(() => renderMathText(text), [text]);

  return <span className={className}>{rendered}</span>;
}


// ─── Rendering Pipeline ──────────────────────────────────────────────

function renderMathText(text) {
  if (typeof text !== 'string' || !text) {
    return [<React.Fragment key="empty">{String(text || '')}</React.Fragment>];
  }

  // Phase 1: Pre-process to fix common LLM issues
  let processed = preprocess(text);

  // Phase 2: Tokenize into math and text segments
  const tokens = tokenize(processed);

  // Phase 3: Render each token
  return tokens.map((token, idx) => {
    if (token.type === 'display') {
      return renderKatex(token.content, true, idx);
    }
    if (token.type === 'inline') {
      return renderKatex(token.content, false, idx);
    }
    // Plain text
    return <React.Fragment key={idx}>{token.content}</React.Fragment>;
  });
}


// ─── Phase 1: Pre-processing ─────────────────────────────────────────

function preprocess(text) {
  // Fix double-escaped backslashes: \\frac → \frac (common JSON artifact)
  text = text.replace(/\\\\(frac|sqrt|sum|prod|int|lim|sin|cos|tan|cot|sec|csc|alpha|beta|gamma|delta|theta|lambda|pi|infty|partial|nabla|cdot|times|div|text|mathrm|mathbf|left|right|log|ln|det|max|min|begin|end|over|quad|hat|bar|vec|dot|tilde|pm|mp|leq|geq|neq|approx|equiv|pmod|bmod|binom|displaystyle)/g, '\\$1');

  // Fix bare LaTeX commands missing their backslash (e.g., "frac{" → "\frac{")
  // Only if not already preceded by a backslash
  text = text.replace(/(?<!\\)\b(frac|sqrt|sum|prod|int|lim)\s*\{/g, '\\$1{');

  return text;
}


// ─── Phase 2: Tokenization ──────────────────────────────────────────

/**
 * LATEX_CMD_RE matches a LaTeX command like \frac{a}{b} including its
 * brace arguments, or simple commands like \pi, \theta, etc.
 */
const LATEX_CMDS = [
  'frac', 'sqrt', 'sum', 'prod', 'int', 'lim',
  'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
  'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'pi',
  'infty', 'partial', 'nabla', 'cdot', 'times', 'div',
  'text', 'mathrm', 'mathbf', 'left', 'right',
  'log', 'ln', 'det', 'max', 'min',
  'over', 'pm', 'mp', 'leq', 'geq', 'neq', 'approx', 'equiv',
  'hat', 'bar', 'vec', 'dot', 'tilde', 'begin', 'end',
];

const LATEX_CMD_PATTERN = new RegExp(
  '\\\\(?:' + LATEX_CMDS.join('|') + ')(?![a-zA-Z])'
);

const MATH_CHAR_RE = /[\\^_{}=+\-*/|<>∑∫∞√∂∇≤≥≠±×÷]/;

const ENGLISH_WORD_RE = /\b(?:Find|find|Solve|solve|Calculate|calculate|Determine|determine|Evaluate|evaluate|Compute|compute|Show|show|Prove|prove|What|what|the|The|value|of|is|are|given|by|using|If|if|Then|then|where|Where|for|For|units?|rod|length|beam|string|vibration|conduction|deflection|temperature|distribution|along|heated|elastic|forced|transverse|membrane)\b/;

function tokenize(text) {
  const tokens = [];
  let i = 0;

  while (i < text.length) {
    // ── 1. Display math $$...$$ ──
    if (text[i] === '$' && text[i + 1] === '$') {
      const close = text.indexOf('$$', i + 2);
      if (close !== -1) {
        tokens.push({ type: 'display', content: text.slice(i + 2, close).trim() });
        i = close + 2;
        continue;
      }
    }

    // ── 2. Inline math $...$ ──
    if (text[i] === '$') {
      const close = findClosingDollar(text, i + 1);
      if (close !== -1) {
        const inner = text.slice(i + 1, close);

        // Case A: Currency ($100, $50.00) → plain text
        if (/^\d/.test(inner)) {
          tokens.push({ type: 'text', content: inner });
          i = close + 1;
          continue;
        }

        const hasEnglish = ENGLISH_WORD_RE.test(inner);
        const hasMath = MATH_CHAR_RE.test(inner) || LATEX_CMD_PATTERN.test(inner);

        // Case B: Pure math → render as inline math
        if (hasMath && !hasEnglish) {
          tokens.push({ type: 'inline', content: inner.trim() });
          i = close + 1;
          continue;
        }

        // Case C: Mixed English + math → split and process recursively
        if (hasEnglish && hasMath) {
          const subTokens = splitMixedContent(inner);
          tokens.push(...subTokens);
          i = close + 1;
          continue;
        }

        // Case D: Pure English inside $ → plain text
        if (hasEnglish && !hasMath) {
          tokens.push({ type: 'text', content: inner });
          i = close + 1;
          continue;
        }

        // Case E: Short, ambiguous → treat as math
        if (inner.trim().length <= 30 && inner.trim().length > 0) {
          tokens.push({ type: 'inline', content: inner.trim() });
          i = close + 1;
          continue;
        }

        // Case F: Long, no math → plain text
        tokens.push({ type: 'text', content: inner });
        i = close + 1;
        continue;
      }

      // Unmatched $ → skip it
      i += 1;
      continue;
    }

    // ── 3. Bare \begin{env}...\end{env} ──
    const envMatch = text.slice(i).match(/^\\begin\{(\w+)\}([\s\S]*?)\\end\{\1\}/);
    if (envMatch) {
      tokens.push({ type: 'display', content: envMatch[0] });
      i += envMatch[0].length;
      continue;
    }

    // ── 4. Bare LaTeX command outside delimiters ──
    if (text[i] === '\\' && LATEX_CMD_PATTERN.test(text.slice(i))) {
      const expr = extractBareExpression(text, i);
      tokens.push({ type: 'inline', content: expr });
      i += expr.length;
      continue;
    }

    // ── 5. Plain text — consume until next special character ──
    let end = i + 1;
    while (end < text.length) {
      // Stop at $ delimiter
      if (text[end] === '$') break;
      // Stop at bare LaTeX command
      if (text[end] === '\\' && LATEX_CMD_PATTERN.test(text.slice(end))) break;
      // Stop at bare \begin
      if (text.slice(end).startsWith('\\begin{')) break;
      end++;
    }
    tokens.push({ type: 'text', content: text.slice(i, end) });
    i = end;
  }

  return tokens;
}


// ─── Helper: Find the matching closing $ ─────────────────────────────

function findClosingDollar(text, start) {
  // Find the next $ that isn't part of $$
  for (let j = start; j < text.length; j++) {
    if (text[j] === '$') {
      // Make sure it's not $$ (display math opener)
      if (j + 1 < text.length && text[j + 1] === '$') {
        return -1; // This is $$, not a closing $
      }
      return j;
    }
  }
  return -1;
}


// ─── Helper: Extract a bare math expression starting at \ ────────────

function extractBareExpression(text, start) {
  let i = start;
  let braceDepth = 0;
  let result = '';

  while (i < text.length) {
    const ch = text[i];

    if (ch === '{') {
      braceDepth++;
      result += ch;
      i++;
      continue;
    }
    if (ch === '}') {
      braceDepth--;
      result += ch;
      i++;
      if (braceDepth <= 0) {
        // After closing a top-level brace group, peek ahead
        const peek = text.slice(i).replace(/^\s*/, '');
        if (peek && /^[\\^_{}+\-*/=(<>a-zA-Z0-9]/.test(peek[0])) {
          // Check if it's an English word (stop) or math (continue)
          const wordMatch = peek.match(/^[a-zA-Z]{3,}/);
          if (wordMatch && !isMathWord(wordMatch[0])) {
            break; // English word — stop
          }
          continue;
        }
        break;
      }
      continue;
    }

    if (braceDepth > 0) {
      result += ch;
      i++;
      continue;
    }

    if (ch === '\\') {
      result += ch;
      i++;
      continue;
    }

    if (ch === '$' || ch === '\n') break;

    // End of expression: stop at periods followed by space+uppercase (sentence boundary)
    if (ch === '.' && i + 1 < text.length && text[i + 1] === ' ') {
      const afterDot = text.slice(i + 2);
      if (/^[A-Z]/.test(afterDot)) break; // New sentence
    }

    if (/[a-zA-Z]/.test(ch)) {
      const wordMatch = text.slice(i).match(/^[a-zA-Z]+/);
      if (wordMatch) {
        const word = wordMatch[0];
        // If previous char is \, this is a command name
        if (result.endsWith('\\')) {
          result += word;
          i += word.length;
          continue;
        }
        // Single letter variable or known math function — include
        if (word.length <= 2 || isMathWord(word)) {
          result += word;
          i += word.length;
          continue;
        }
        // English word — stop
        break;
      }
    }

    if (/[0-9^_+\-*/=(),.<>]/.test(ch) || ch === ' ') {
      if (ch === ' ') {
        // Only include space if math continues after it
        const afterSpace = text.slice(i).replace(/^\s*/, '');
        if (afterSpace && /^[\\^_{}+\-*/=<>0-9a-z(]/.test(afterSpace[0])) {
          result += ch;
          i++;
          continue;
        }
        break;
      }
      result += ch;
      i++;
      continue;
    }

    break;
  }

  return result.trim() || text[start];
}


function isMathWord(word) {
  const mathWords = new Set([
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln', 'exp',
    'det', 'lim', 'max', 'min', 'sup', 'inf', 'mod', 'gcd', 'lcm',
    'pi', 'alpha', 'beta', 'gamma', 'theta', 'delta', 'lambda',
    'sigma', 'omega', 'phi', 'psi', 'frac', 'sqrt',
  ]);
  return mathWords.has(word.toLowerCase());
}


// ─── Helper: Split mixed English + math into separate tokens ─────────

function splitMixedContent(inner) {
  const tokens = [];
  let pos = 0;

  while (pos < inner.length) {
    // Find the next LaTeX command
    const cmdMatch = inner.slice(pos).match(LATEX_CMD_PATTERN);

    if (!cmdMatch) {
      // No more LaTeX — rest is plain text (but check for simple math)
      const remainder = inner.slice(pos);
      if (remainder.trim()) {
        tokens.push(...tokenizeSimpleMath(remainder));
      }
      break;
    }

    const cmdStart = pos + cmdMatch.index;

    // Output any text before the math
    if (cmdStart > pos) {
      const before = inner.slice(pos, cmdStart);
      if (before.trim()) {
        tokens.push(...tokenizeSimpleMath(before));
      }
    }

    // Extract the full expression starting at this command
    const expr = extractBareExpression(inner, cmdStart);
    tokens.push({ type: 'inline', content: expr.trim() });
    pos = cmdStart + expr.length;
  }

  return tokens;
}


// ─── Helper: Find simple math in otherwise-plain text ────────────────

function tokenizeSimpleMath(text) {
  const tokens = [];

  // Match function calls like f(x), g(2), F(pi/4)
  const parts = text.split(/(\b[a-zA-Z]\([^)]+\))/g);

  for (const part of parts) {
    if (!part) continue;

    // Check if it's a function call like f(x)
    const funcMatch = part.match(/^([a-zA-Z])\(([^)]+)\)$/);
    if (funcMatch) {
      tokens.push({ type: 'inline', content: part });
      continue;
    }

    // Check for expressions with ^ or _ (like x^2, a_n)
    if (/[a-zA-Z][_^]/.test(part)) {
      // Split around the math expression
      const subParts = part.split(/(\b[a-zA-Z]\w*[_^][^\s,;.!?]*)/g);
      for (const sub of subParts) {
        if (!sub) continue;
        if (/[_^]/.test(sub) && /^[a-zA-Z]/.test(sub)) {
          tokens.push({ type: 'inline', content: sub.trim() });
        } else {
          tokens.push({ type: 'text', content: sub });
        }
      }
      continue;
    }

    tokens.push({ type: 'text', content: part });
  }

  return tokens;
}


// ─── Phase 3: KaTeX Rendering ────────────────────────────────────────

function renderKatex(math, displayMode, key) {
  try {
    const html = katex.renderToString(math, {
      displayMode,
      throwOnError: false,
      trust: true,
      strict: false,
    });
    return (
      <span
        key={key}
        className={displayMode ? 'math-block-wrapper' : 'math-inline-wrapper'}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  } catch {
    // KaTeX failed — try cleaning up common issues and retry
    const cleaned = cleanForKatex(math);
    try {
      const html = katex.renderToString(cleaned, {
        displayMode,
        throwOnError: false,
        trust: true,
        strict: false,
      });
      return (
        <span
          key={key}
          className={displayMode ? 'math-block-wrapper' : 'math-inline-wrapper'}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    } catch {
      // Complete failure — render as styled code so it's at least readable
      return (
        <code key={key} className="math-error" style={{ color: 'var(--text-primary)', opacity: 0.9 }}>
          {math}
        </code>
      );
    }
  }
}


function cleanForKatex(math) {
  // Remove any remaining double-escaped backslashes
  let cleaned = math.replace(/\\\\/g, '\\');

  // Fix \frac without proper braces: \frac12 → \frac{1}{2}
  cleaned = cleaned.replace(/\\frac\s*([^{])\s*([^{])/g, '\\frac{$1}{$2}');

  // Fix \sqrt without braces: \sqrtx → \sqrt{x}
  cleaned = cleaned.replace(/\\sqrt\s*([^{[\s])/g, '\\sqrt{$1}');

  return cleaned;
}
