"""Lightweight LOCAL retriever (mini-RAG) — costs ZERO API calls.

WHY NOT embeddings/FAISS? Embedding every chunk of your notes costs API
calls. For a small notes file, simple keyword-overlap scoring works well
and is free. The retrieved chunks are injected into agent prompts, which
also keeps prompts SHORT -> fewer tokens -> cheaper/faster LLM calls.
"""

import re
from typing import List


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split notes into ~chunk_size character chunks on paragraph borders."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_size and current:
            chunks.append(current.strip())
            current = ""
        current += p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))


def retrieve(query: str, text: str, top_k: int = 3) -> List[str]:
    """Return the top_k chunks most relevant to the query."""
    chunks = chunk_text(text)
    if not chunks:
        return []
    q = _tokens(query)
    scored = sorted(chunks, key=lambda c: len(q & _tokens(c)), reverse=True)
    return scored[:top_k]
