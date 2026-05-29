import re
from typing import List


def clean_legal_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Case \d+:\d+-cv-[\w-]+ Document \d+.*?\n", "", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 2500, overlap: int = 300):
    text = clean_legal_text(text)

    if len(text) <= chunk_size:
        return [{
            "chunk_id": 0,
            "text": text,
            "start": 0,
            "end": len(text)
        }]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start": start,
                "end": end
            })
            chunk_id += 1

        start = end - overlap

    return chunks


def get_chunk_stats(text: str) -> dict:
    chunks = chunk_text(text)

    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks),
        "max_chunk_length": max(len(c) for c in chunks),
        "min_chunk_length": min(len(c) for c in chunks),
    }

