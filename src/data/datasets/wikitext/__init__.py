"""WikiText-103 dataset processing.

Tools:
- normalize: Fix WikiText formatting artifacts (@ tokens, punctuation spacing, headers)

Document boundary pattern for extract_text.py:
    r'^\\s*=\\s+[^=]+\\s+=\\s*$'  (top-level article headers, e.g. ' = Title = ')
"""

__all__: list[str] = []
