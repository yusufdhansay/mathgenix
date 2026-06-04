import React from 'react';
import katex from 'katex';

/**
 * MathRenderer — Renders mixed text + LaTeX math with KaTeX.
 *
 * Parsing rules:
 *   - $$...$$ → KaTeX display (block) math
 *   - $...$   → KaTeX inline math, BUT only when:
 *       1. The opening $ is NOT followed by a digit (to avoid currency like $1000)
 *       2. The content between $ delimiters is short enough to be a formula (< 200 chars)
 *       3. The content contains at least one "math-like" character (^, _, \, =, {, }, etc.)
 *          OR is purely numeric/symbolic (e.g., "$x$", "$3$")
 *   - Everything else → plain text
 */
export default function MathRenderer({ text = '', className = '' }) {
  if (typeof text !== 'string') {
    return <span className={className}>{String(text)}</span>;
  }

  const rendered = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // 1. Try to match block math $$...$$  (highest priority)
    const blockMatch = remaining.match(/^\$\$([\s\S]+?)\$\$/);
    if (blockMatch && remaining.indexOf('$$') === 0) {
      const math = blockMatch[1].trim();
      try {
        const html = katex.renderToString(math, {
          displayMode: true,
          throwOnError: false,
          trust: true,
        });
        rendered.push(
          <span
            key={key++}
            className="math-block-wrapper"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      } catch {
        rendered.push(<code key={key++} className="math-error">{blockMatch[0]}</code>);
      }
      remaining = remaining.slice(blockMatch[0].length);
      continue;
    }

    // 1.5. Try to match bare \begin{env}...\end{env} blocks (not wrapped in $$)
    const envMatch = remaining.match(/^\\begin\{(\w+)\}([\s\S]*?)\\end\{\1\}/);
    if (envMatch) {
      const math = envMatch[0];
      try {
        const html = katex.renderToString(math, {
          displayMode: true,
          throwOnError: false,
          trust: true,
        });
        rendered.push(
          <span
            key={key++}
            className="math-block-wrapper"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      } catch {
        rendered.push(<code key={key++} className="math-error">{math}</code>);
      }
      remaining = remaining.slice(envMatch[0].length);
      continue;
    }

    // 2. Try to match inline math $...$
    //    Skip if $ is followed by a digit (currency like $1000)
    if (remaining[0] === '$' && remaining.length > 1 && !/\d/.test(remaining[1])) {
      // Find the closing $
      const closeIdx = remaining.indexOf('$', 1);
      if (closeIdx > 0 && closeIdx < 300) {
        const inner = remaining.slice(1, closeIdx).trim();

        // Validate: must look like math, not a plain English sentence
        // Math-like if it contains typical LaTeX chars or is very short (single variable)
        const looksLikeMath =
          inner.length <= 80 &&
          (
            /[\\^_{}=+\-*/|<>∑∫∞√∂∇≤≥≠±×÷∈∉⊂⊃∪∩]/.test(inner) ||  // LaTeX operators
            /^[a-zA-Z0-9.,\s()]+$/.test(inner) && inner.length <= 20 || // Short like "x = 3"
            /\\frac|\\sqrt|\\sum|\\int|\\lim|\\log|\\sin|\\cos|\\tan|\\pi|\\alpha|\\beta|\\theta|\\infty|\\cdot|\\times/.test(inner)  // LaTeX commands
          );

        if (looksLikeMath && inner.length > 0) {
          try {
            const html = katex.renderToString(inner, {
              displayMode: false,
              throwOnError: false,
              trust: true,
            });
            rendered.push(
              <span
                key={key++}
                className="math-inline-wrapper"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            );
          } catch {
            // KaTeX failed — render as plain text instead of breaking
            rendered.push(<React.Fragment key={key++}>{remaining.slice(0, closeIdx + 1)}</React.Fragment>);
          }
          remaining = remaining.slice(closeIdx + 1);
          continue;
        }
      }
    }

    // 3. No math match — consume plain text up to the next $ or end
    const nextDollar = remaining.indexOf('$', 1);
    if (nextDollar === -1) {
      // No more $ signs — rest is plain text
      rendered.push(<React.Fragment key={key++}>{remaining}</React.Fragment>);
      remaining = '';
    } else {
      // Output text up to (but not including) the next $
      rendered.push(<React.Fragment key={key++}>{remaining.slice(0, nextDollar)}</React.Fragment>);
      remaining = remaining.slice(nextDollar);
    }
  }

  return <span className={className}>{rendered}</span>;
}
