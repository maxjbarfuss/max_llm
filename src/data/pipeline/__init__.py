"""Common dataset-agnostic pipeline steps.

Tools:
- tokenize: Convert normalized text to a cached .npy token array
- extract_tokens: Extract a token subset from a cached .npy array
- extract_text: Extract a text subset respecting configurable document boundaries
"""

__all__: list[str] = []
