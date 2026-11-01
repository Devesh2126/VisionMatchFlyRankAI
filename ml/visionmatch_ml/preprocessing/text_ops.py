import re


def sanitize_query_text(text: str, max_length: int = 500) -> str:
    """
    Normalizes whitespace, strips control characters,
    and enforces length bounds before tokenization.
    """
    if not isinstance(text, str):
        raise ValueError("Query text must be a string.")

    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        raise ValueError("Query text cannot be empty after sanitization.")

    return cleaned[:max_length]